"""03_build_agri.py

Processes ICRISAT Apportioned Database (1966-2017) to:
1. Aggregate seasonal Kharif crop production, area, and composite yield (kg/ha).
2. Compute crop diversification index (Simpson's Index).
3. Integrate irrigation coverage, rainfed share, fertilizer intensity, and cropping intensity.
4. Detrend district yield time-series (1990-2017) to construct residuals and z-scores.
5. Create binary yield shock labels at multiple thresholds (-1.0 SD, -20% drop, -1.28 SD severe).
6. Engineer temporal lag features (previous year's yield anomaly and shock status).
7. Save to data/processed/agri_district_season_year.parquet and .csv.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_simpson_diversity(row, crop_area_cols):
    """Computes Simpson's Crop Diversity Index: 1 - sum(p_i^2)."""
    areas = row[crop_area_cols].values.astype(float)
    areas = np.nan_to_num(areas, nan=0.0)
    total = np.sum(areas)
    if total <= 0:
        return 0.0
    shares = areas / total
    return float(1.0 - np.sum(shares ** 2))


def map_special_icrisat_codes(df):
    """
    Maps ICRISAT special 3-digit apportioned codes (multiplied by 10)
    to official Census 2011 district codes:
    990 -> 99 (Ganganagar)
    740 -> 74 (Hissar)
    860 -> 86 (Gurgaon)
    770 -> 77 (Jind)
    820 -> 82 (Mahendragarh)
    660 -> 66 (Ambala)
    700 -> 70 (Karnal)
    800 -> 80 (Rohtak)
    990/67 -> 67 (Nainital)
    """
    code_map = {
        800: 80,
        770: 77,
        740: 74,
        860: 86,
        840: 84,
        660: 66,
        820: 82,
        700: 70,
        990: 99,
    }
    df["censuscode"] = df["Indian census District Code 2011"].replace(code_map).astype(int)
    return df


def main():
    config = load_config()
    raw_icrisat_dir = config["paths"]["raw_icrisat"]
    processed_dir = config["paths"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    print("[INIT] Loading ICRISAT agricultural tables...")
    df_yield = pd.read_parquet(os.path.join(raw_icrisat_dir, "area-production-yield.parquet"))
    df_irr = pd.read_parquet(os.path.join(raw_icrisat_dir, "sourcewise-irrigated-area.parquet"))
    df_land = pd.read_parquet(os.path.join(raw_icrisat_dir, "landuse.parquet"))
    df_fert = pd.read_parquet(os.path.join(raw_icrisat_dir, "fertilizer-consumption.parquet"))

    print(f" - Crop records: {len(df_yield):,}")
    print(f" - Irrigation records: {len(df_irr):,}")
    print(f" - Land use records: {len(df_land):,}")
    print(f" - Fertilizer records: {len(df_fert):,}")

    # Map census codes
    df_yield = map_special_icrisat_codes(df_yield)
    df_irr = map_special_icrisat_codes(df_irr)
    df_land = map_special_icrisat_codes(df_land)
    df_fert = map_special_icrisat_codes(df_fert)

    # 1. Aggregate Kharif Crops
    kharif_crops = [
        "RICE", "KHARIF SORGHUM", "PEARL MILLET", "MAIZE", 
        "PIGEONPEA", "GROUNDNUT", "SOYABEAN", "COTTON"
    ]
    kharif_area_cols = [f"{c} AREA" for c in kharif_crops if f"{c} AREA" in df_yield.columns]
    kharif_prod_cols = [f"{c} PRODUCTION" for c in kharif_crops if f"{c} PRODUCTION" in df_yield.columns]

    df_yield["kharif_total_area_1000ha"] = df_yield[kharif_area_cols].sum(axis=1)
    df_yield["kharif_total_prod_1000t"] = df_yield[kharif_prod_cols].sum(axis=1)
    
    # Calculate composite yield: (Prod * 1000 tonnes -> kg) / (Area * 1000 ha -> ha) = kg/ha
    valid_area = df_yield["kharif_total_area_1000ha"] > 0.05
    df_yield["kharif_composite_yield_kgha"] = np.where(
        valid_area,
        (df_yield["kharif_total_prod_1000t"] * 1000.0) / df_yield["kharif_total_area_1000ha"],
        np.nan
    )

    # Compute crop diversification index (Simpson's Diversity)
    print("[AGRI] Calculating Simpson's Crop Diversification Index...")
    all_crop_area_cols = [c for c in df_yield.columns if c.endswith(" AREA") and not c.startswith("kharif_")]
    df_yield["crop_diversification_index"] = df_yield.apply(
        lambda row: compute_simpson_diversity(row, all_crop_area_cols), axis=1
    )

    # Select core agricultural columns
    agri_core = df_yield[[
        "Dist Code", "Year", "censuscode", "State Name", "Dist Name",
        "kharif_total_area_1000ha", "kharif_total_prod_1000t",
        "kharif_composite_yield_kgha", "crop_diversification_index"
    ]].copy()

    # 2. Merge Irrigation Coverage
    irr_subset = df_irr[[
        "Dist Code", "Year", "CANALS AREA", "TUBE WELLS AREA", "TANKS AREA", "NET AREA", "GROSS AREA"
    ]].rename(columns={
        "CANALS AREA": "canal_irrigated_area_1000ha",
        "TUBE WELLS AREA": "tubewell_irrigated_area_1000ha",
        "TANKS AREA": "tank_irrigated_area_1000ha",
        "NET AREA": "net_irrigated_area_1000ha",
        "GROSS AREA": "gross_irrigated_area_1000ha"
    })
    agri_core = agri_core.merge(irr_subset, on=["Dist Code", "Year"], how="left")

    # 3. Merge Land Use
    land_subset = df_land[[
        "Dist Code", "Year", "NET CROPPED AREA", "GROSS CROPPED AREA", "CROPING INTENSITY"
    ]].rename(columns={
        "NET CROPPED AREA": "net_cropped_area_1000ha",
        "GROSS CROPPED AREA": "gross_cropped_area_1000ha",
        "CROPING INTENSITY": "cropping_intensity"
    })
    agri_core = agri_core.merge(land_subset, on=["Dist Code", "Year"], how="left")

    # 4. Merge Fertilizer Consumption
    fert_subset = df_fert[[
        "Dist Code", "Year", "TOTAL CONSUMPTION", "TOTAL PER HA OF NCA"
    ]].rename(columns={
        "TOTAL CONSUMPTION": "total_fertilizer_1000t",
        "TOTAL PER HA OF NCA": "fertilizer_kg_per_ha_nca"
    })
    agri_core = agri_core.merge(fert_subset, on=["Dist Code", "Year"], how="left")

    # Calculate irrigation share and rainfed share
    nca_valid = agri_core["net_cropped_area_1000ha"] > 0.05
    agri_core["irrigation_coverage_ratio"] = np.where(
        nca_valid,
        np.clip(agri_core["net_irrigated_area_1000ha"] / agri_core["net_cropped_area_1000ha"], 0.0, 1.0),
        np.nan
    )
    agri_core["rainfed_area_share"] = 1.0 - agri_core["irrigation_coverage_ratio"]

    # 5. Detrending and Shock Labeling (1990-2017)
    print("[DETREND] Fitting district-specific yield trends (1990-2017)...")
    agri_recent = agri_core[agri_core["Year"] >= 1990].copy().sort_values(["censuscode", "Year"])

    detrended_records = []
    threshold_sd = config["agri_shock"]["shock_threshold_sd"]       # -1.0
    threshold_pct = config["agri_shock"]["shock_threshold_pct"]     # -0.20 (-20%)
    threshold_sev = config["agri_shock"]["secondary_shock_threshold_sd"] # -1.28

    for c_code, group in agri_recent.groupby("censuscode"):
        sub = group.copy().sort_values("Year")
        valid_mask = sub["kharif_composite_yield_kgha"].notna()
        valid_sub = sub[valid_mask]

        if len(valid_sub) >= 10:
            # Linear trend fitting on valid years
            poly = np.polyfit(valid_sub["Year"], valid_sub["kharif_composite_yield_kgha"], 1)
            fitted_trend = np.polyval(poly, sub["Year"])
            
            # Prevent negative or unrealistically near-zero trends
            min_floor = max(valid_sub["kharif_composite_yield_kgha"].quantile(0.05), 100.0)
            fitted_trend = np.maximum(fitted_trend, min_floor)

            residuals = sub["kharif_composite_yield_kgha"] - fitted_trend
            valid_residuals = residuals[valid_mask]
            res_std = valid_residuals.std() if len(valid_residuals) > 1 else 1.0
            if res_std <= 0:
                res_std = 1.0

            sub["yield_trend_kgha"] = fitted_trend.round(2)
            sub["yield_residual_kgha"] = residuals.round(2)
            sub["yield_anomaly_z"] = (residuals / res_std).round(3)
            sub["yield_deviation_pct"] = ((residuals / fitted_trend) * 100.0).round(2)

            # Define binary shock targets
            sub["shock_label_1sd"] = np.where(valid_mask, (sub["yield_anomaly_z"] <= threshold_sd).astype(int), np.nan)
            sub["shock_label_20pct"] = np.where(valid_mask, (sub["yield_deviation_pct"] <= (threshold_pct * 100)).astype(int), np.nan)
            sub["shock_label_severe"] = np.where(valid_mask, (sub["yield_anomaly_z"] <= threshold_sev).astype(int), np.nan)
        else:
            sub["yield_trend_kgha"] = np.nan
            sub["yield_residual_kgha"] = np.nan
            sub["yield_anomaly_z"] = np.nan
            sub["yield_deviation_pct"] = np.nan
            sub["shock_label_1sd"] = np.nan
            sub["shock_label_20pct"] = np.nan
            sub["shock_label_severe"] = np.nan

        # Temporal lag features (1 year lag)
        sub["yield_kgha_lag1"] = sub["kharif_composite_yield_kgha"].shift(1)
        sub["yield_anomaly_z_lag1"] = sub["yield_anomaly_z"].shift(1)
        sub["shock_label_1sd_lag1"] = sub["shock_label_1sd"].shift(1)

        detrended_records.append(sub)

    df_agri_final = pd.concat(detrended_records, ignore_index=True)

    # 6. Quality Checks and Class Balance Report
    print("\n=======================================================")
    print(" Agricultural Panel Quality & Target Distribution")
    print("=======================================================")
    valid_shocks = df_agri_final["shock_label_1sd"].dropna()
    shock_count_1sd = int(valid_shocks.sum())
    total_valid = len(valid_shocks)
    shock_rate_1sd = (shock_count_1sd / total_valid) * 100.0

    shock_count_20pct = int(df_agri_final["shock_label_20pct"].dropna().sum())
    shock_rate_20pct = (shock_count_20pct / total_valid) * 100.0

    shock_count_sev = int(df_agri_final["shock_label_severe"].dropna().sum())
    shock_rate_sev = (shock_count_sev / total_valid) * 100.0

    print(f"Total district-years analyzed (1990-2017): {total_valid:,}")
    print(f"Primary Target (Shock < -1.0 SD): {shock_count_1sd} shocks ({shock_rate_1sd:.2f}% base rate)")
    print(f"Secondary Target (Drop > 20%):     {shock_count_20pct} shocks ({shock_rate_20pct:.2f}% base rate)")
    print(f"Severe Target (Shock < -1.28 SD):  {shock_count_sev} shocks ({shock_rate_sev:.2f}% base rate)")
    print(f"Mean Kharif Yield: {df_agri_final['kharif_composite_yield_kgha'].mean():.1f} kg/ha")
    print(f"Mean Irrigation Coverage: {df_agri_final['irrigation_coverage_ratio'].mean() * 100:.1f}%")
    print(f"Mean Crop Diversification: {df_agri_final['crop_diversification_index'].mean():.3f}")

    # Export
    out_parquet = os.path.join(processed_dir, "agri_district_season_year.parquet")
    out_csv = os.path.join(processed_dir, "agri_district_season_year.csv")
    df_agri_final.to_parquet(out_parquet, index=False)
    df_agri_final.to_csv(out_csv, index=False)

    print("\n=======================================================")
    print(f" [SUCCESS] Agricultural panel exported successfully:")
    print(f"  - Parquet: {out_parquet} ({os.path.getsize(out_parquet) / 1024:.1f} KB)")
    print(f"  - CSV:     {out_csv} ({os.path.getsize(out_csv) / 1024:.1f} KB)")
    print(f"  - Rows: {len(df_agri_final):,}, Columns: {len(df_agri_final.columns)}")
    print(f"  - Districts covered: {df_agri_final['censuscode'].nunique()}")
    print("=======================================================")


if __name__ == "__main__":
    main()

