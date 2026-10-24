"""05_build_index.py

Constructs Module A: Structural Vulnerability Index under the IPCC Framework:
Vulnerability = f(Exposure, Sensitivity, Adaptive Capacity).

Indicators:
- Exposure: Rainfall CV, drought frequency, dry spell duration, hot days, heavy rain frequency
- Sensitivity: Ag workforce share, ag labourer share, rainfed area share, smallholder share, crop concentration
- Adaptive Capacity: Irrigation coverage, female literacy, overall literacy, fertilizer intensity

Methods:
1. Min-max normalization with directional orientation (higher = more vulnerable)
2. Inversion of Adaptive Capacity (1 - norm)
3. Both Equal Weighting (EW) and PCA-based Weighting (PCA)
4. Weighting sensitivity analysis (Spearman rank correlation between EW and PCA)
5. Tertile classification into Low, Moderate, High vulnerability classes
6. Independent validation against NITI Aayog Multidimensional Poverty Index (MPI)
7. Explainability Random Forest classifier with feature importances
8. Saves vulnerability_index_districts.parquet and .csv
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def min_max_scale(series):
    """Min-max normalizes a pandas Series to [0, 1]."""
    s_min = series.min()
    s_max = series.max()
    if s_max == s_min:
        return pd.Series(0.5, index=series.index)
    return (series - s_min) / (s_max - s_min)


def main():
    config = load_config()
    processed_dir = config["paths"]["processed_dir"]

    print("[INIT] Loading climate, agricultural, and static panels...")
    df_clim = pd.read_parquet(os.path.join(processed_dir, "climate_district_year.parquet"))
    df_agri = pd.read_parquet(os.path.join(processed_dir, "agri_district_season_year.parquet"))
    df_stat = pd.read_parquet(os.path.join(processed_dir, "static_district_features.parquet"))

    # 1. Compute Long-Term Climate Exposure Indicators (1990-2020) per district
    print("[MODULE A] Aggregating long-term climate exposure indicators...")
    clim_summary = df_clim.groupby("censuscode").agg(
        rainfall_cv=("monsoon_rain_cv", "mean"),
        drought_frequency=("drought_flag", "mean"),
        mean_longest_dry_spell=("longest_dry_spell_days", "mean"),
        mean_hot_days=("monsoon_hot_days", "mean"),
        mean_heavy_rain_days=("heavy_rain_days", "mean"),
    ).reset_index()

    # 2. Compute Long-Term Agricultural Sensitivity & Adaptive Indicators per district
    print("[MODULE A] Aggregating agricultural sensitivity & adaptive indicators...")
    agri_summary = df_agri.groupby("censuscode").agg(
        mean_irrigation_coverage=("irrigation_coverage_ratio", "mean"),
        mean_rainfed_share=("rainfed_area_share", "mean"),
        mean_crop_diversification=("crop_diversification_index", "mean"),
        mean_yield_kgha=("kharif_composite_yield_kgha", "mean"),
        yield_std=("kharif_composite_yield_kgha", "std"),
        mean_fertilizer_kgha=("fertilizer_kg_per_ha_nca", "mean")
    ).reset_index()

    agri_summary["yield_cv"] = np.where(
        agri_summary["mean_yield_kgha"] > 0,
        agri_summary["yield_std"] / agri_summary["mean_yield_kgha"],
        0.0
    )
    # Crop concentration = 1 - crop diversification
    agri_summary["crop_concentration"] = 1.0 - agri_summary["mean_crop_diversification"]

    # 3. Merge All Indicators on Master Census Code
    df_idx = df_stat.merge(clim_summary, on="censuscode", how="left")
    df_idx = df_idx.merge(agri_summary, on="censuscode", how="left")

    # Focus on the analytical coverage districts (ICRISAT agricultural states + Census + Climate)
    valid_mask = df_idx["mean_rainfed_share"].notna() & df_idx["rainfall_cv"].notna()
    df_model = df_idx[valid_mask].copy().reset_index(drop=True)
    print(f"[MODULE A] Analysis cohort: {len(df_model)} districts with complete multi-sector indicators.")

    # 4. Indicator Definitions and Orientation (Higher = More Vulnerable)
    # Exposure:
    # - rainfall_cv (+)
    # - drought_frequency (+)
    # - mean_longest_dry_spell (+)
    # - mean_hot_days (+)
    # - mean_heavy_rain_days (+)
    exposure_vars = [
        "rainfall_cv", "drought_frequency", "mean_longest_dry_spell",
        "mean_hot_days", "mean_heavy_rain_days"
    ]
    for var in exposure_vars:
        df_model[f"{var}_norm"] = min_max_scale(df_model[var].fillna(df_model[var].median()))

    # Sensitivity:
    # - ag_workforce_share_pct (+)
    # - ag_labourer_share_pct (+)
    # - mean_rainfed_share (+)
    # - small_marginal_holdings_pct (+)
    # - crop_concentration (+)
    # - yield_cv (+)
    sensitivity_vars = [
        "ag_workforce_share_pct", "ag_labourer_share_pct", "mean_rainfed_share",
        "small_marginal_holdings_pct", "crop_concentration", "yield_cv"
    ]
    for var in sensitivity_vars:
        df_model[f"{var}_norm"] = min_max_scale(df_model[var].fillna(df_model[var].median()))

    # Adaptive Capacity:
    # - female_literacy_rate (-) -> Inverted: (1 - norm)
    # - overall_literacy_rate (-) -> Inverted: (1 - norm)
    # - mean_irrigation_coverage (-) -> Inverted: (1 - norm)
    # - mean_fertilizer_kgha (-) -> Inverted: (1 - norm)
    adaptive_vars = [
        "female_literacy_rate", "overall_literacy_rate",
        "mean_irrigation_coverage", "mean_fertilizer_kgha"
    ]
    for var in adaptive_vars:
        norm_val = min_max_scale(df_model[var].fillna(df_model[var].median()))
        # Invert so higher value = lack of adaptive capacity = higher vulnerability
        df_model[f"{var}_lack_norm"] = 1.0 - norm_val

    # 5. Dimension Sub-Indices
    exp_cols = [f"{v}_norm" for v in exposure_vars]
    sens_cols = [f"{v}_norm" for v in sensitivity_vars]
    adapt_cols = [f"{v}_lack_norm" for v in adaptive_vars]

    df_model["exposure_subindex"] = df_model[exp_cols].mean(axis=1).round(4)
    df_model["sensitivity_subindex"] = df_model[sens_cols].mean(axis=1).round(4)
    df_model["lack_adaptive_capacity_subindex"] = df_model[adapt_cols].mean(axis=1).round(4)

    # 6. Weighting Schemes: Equal Weights vs. OECD PCA Weights
    # A) Equal Weights (EW)
    df_model["vulnerability_index_equal"] = (
        (df_model["exposure_subindex"] + df_model["sensitivity_subindex"] + df_model["lack_adaptive_capacity_subindex"]) / 3.0
    ).round(4)

    # B) OECD PCA Weights (squared factor loadings weighted by variance explained)
    all_oriented_cols = exp_cols + sens_cols + adapt_cols
    X_mat = df_model[all_oriented_cols].values
    pca = PCA(n_components=3, random_state=42)
    pca.fit(X_mat)

    # Weight per indicator j = sum_k (explained_variance_ratio_k * (|loading_jk| / sum_m |loading_mk|))
    oecd_weights = np.zeros(len(all_oriented_cols))
    for k in range(3):
        comp_abs = np.abs(pca.components_[k])
        comp_norm = comp_abs / np.sum(comp_abs)
        oecd_weights += pca.explained_variance_ratio_[k] * comp_norm
    oecd_weights = oecd_weights / np.sum(oecd_weights)

    pca_scores = X_mat.dot(oecd_weights)
    df_model["vulnerability_index_pca"] = min_max_scale(pd.Series(pca_scores)).round(4)
    pca_weights = pd.Series(oecd_weights, index=all_oriented_cols)

    # 7. Sensitivity Analysis on Weighting
    ew_pca_corr, _ = spearmanr(df_model["vulnerability_index_equal"], df_model["vulnerability_index_pca"])
    rank_ew = df_model["vulnerability_index_equal"].rank()
    rank_pca = df_model["vulnerability_index_pca"].rank()
    mean_rank_diff = np.abs(rank_ew - rank_pca).mean()

    print("\n=======================================================")
    print(" Weighting Sensitivity Analysis (EW vs. PCA)")
    print("=======================================================")
    print(f"Spearman Rank Correlation (EW vs. PCA): r_s = {ew_pca_corr:.4f}")
    print(f"Mean Absolute District Rank Shift: {mean_rank_diff:.1f} ranks out of {len(df_model)}")
    print(f"PCA Variance Explained (PC1): {pca.explained_variance_ratio_[0] * 100:.2f}%")

    # 8. Categorization into Tertiles (Low, Moderate, High)
    tertile_cuts = df_model["vulnerability_index_equal"].quantile([0.333, 0.667]).values
    labels = ["Low", "Moderate", "High"]
    df_model["vulnerability_class"] = pd.qcut(
        df_model["vulnerability_index_equal"], q=3, labels=labels
    )

    print("\n=======================================================")
    print(" Structural Vulnerability Class Distribution")
    print("=======================================================")
    print(df_model["vulnerability_class"].value_counts().to_string())

    # 9. Independent External Validation against NITI Aayog MPI
    print("\n=======================================================")
    print(" Independent Validation against NITI Aayog MPI")
    print("=======================================================")
    valid_mpi = df_model.dropna(subset=["mpi_headcount_ratio"])
    spearman_mpi, p_val_spearman = spearmanr(
        valid_mpi["vulnerability_index_equal"], valid_mpi["mpi_headcount_ratio"]
    )
    pearson_mpi, p_val_pearson = pearsonr(
        valid_mpi["vulnerability_index_equal"], valid_mpi["mpi_headcount_ratio"]
    )

    print(f"Validation Sample Size: {len(valid_mpi)} districts")
    print(f"Spearman Rank Correlation vs NITI MPI: r_s = {spearman_mpi:.4f} (p = {p_val_spearman:.2e})")
    print(f"Pearson Correlation vs NITI MPI:       r   = {pearson_mpi:.4f} (p = {p_val_pearson:.2e})")
    assert spearman_mpi > 0.25, "Validation failed: Structural vulnerability should be positively correlated with Multidimensional Poverty!"

    # 10. Explainability Classifier (RF as an Interpretation Tool, NOT predictive accuracy)
    print("\n=======================================================")
    print(" Explainability Model (Driver Attribution)")
    print("=======================================================")
    print("[NOTE] This classifier explains which indicators drive the IPCC index class.")
    print("[NOTE] It reproduces the formula for driver attribution, NOT predictive validity.")

    X_explain = df_model[all_oriented_cols]
    y_explain = df_model["vulnerability_class"]
    rf_explain = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_explain.fit(X_explain, y_explain)

    feature_importances = pd.Series(rf_explain.feature_importances_, index=all_oriented_cols).sort_values(ascending=False)
    print("\nTop 7 Drivers of Structural Vulnerability:")
    for feat, imp in feature_importances.head(7).items():
        clean_feat = feat.replace("_norm", "").replace("_lack", "").replace("_", " ").title()
        print(f" - {clean_feat:32s}: {imp * 100:.2f}% relative contribution")

    # 11. Final Dataset Assembly
    output_cols = [
        "censuscode", "DISTRICT", "ST_NM", "ST_CEN_CD", "DT_CEN_CD",
        "exposure_subindex", "sensitivity_subindex", "lack_adaptive_capacity_subindex",
        "vulnerability_index_equal", "vulnerability_index_pca", "vulnerability_class",
        "rainfall_cv", "drought_frequency", "mean_longest_dry_spell",
        "ag_workforce_share_pct", "mean_rainfed_share", "small_marginal_holdings_pct",
        "crop_concentration", "female_literacy_rate", "mean_irrigation_coverage",
        "mpi_headcount_ratio"
    ]
    df_output = df_model[output_cols].copy()

    # Save
    out_parquet = os.path.join(processed_dir, "vulnerability_index_districts.parquet")
    out_csv = os.path.join(processed_dir, "vulnerability_index_districts.csv")
    df_output.to_parquet(out_parquet, index=False)
    df_output.to_csv(out_csv, index=False)

    print("\n=======================================================")
    print(f" [SUCCESS] Module A Vulnerability Index Complete:")
    print(f"  - Parquet: {out_parquet} ({os.path.getsize(out_parquet) / 1024:.1f} KB)")
    print(f"  - CSV:     {out_csv} ({os.path.getsize(out_csv) / 1024:.1f} KB)")
    print(f"  - Districts: {len(df_output)}")
    print(f"  - Independent Spearman correlation vs MPI: r_s = {spearman_mpi:.4f}")
    print("=======================================================")


if __name__ == "__main__":
    main()
