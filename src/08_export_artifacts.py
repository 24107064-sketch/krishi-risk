"""08_export_artifacts.py

Prepares and verifies all static and precomputed artifacts consumed by the Streamlit application:
1. Generates simplified districts.geojson for high-performance interactive Plotly choropleths.
2. Embeds precomputed vulnerability classes, shock probabilities, and priority scores into geojson properties.
3. Generates app/model_card.md documenting intended use, data, honest metrics, limitations, and versioning.
4. Verifies all required artifacts exist and are ready for app consumption (no retraining in app).
"""

import os
import json
import yaml
import geopandas as gpd
import pandas as pd


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model_card(output_path="app/model_card.md"):
    content = """# Model Card: Krishi Risk Early-Warning & Livelihood Vulnerability System

## Model Overview
- **Name:** Krishi Risk Agricultural Shock Predictor & Vulnerability Assessment System
- **Version:** 1.0.0 (Research Release)
- **Primary Institution:** A.P. Shah Institute of Technology, Research Methodology Lab
- **Principal Investigator:** Shivam Mewada (TE Data Science), under Prof. Veena Trivedi
- **Target Task:** 
  1. Measure slow-moving structural rural livelihood vulnerability under IPCC framework.
  2. Predict probability of serious Kharif crop yield shock (residual < -1.0 SD) 3 months in advance (as of July 31).

---

## Intended Use
- **Target Users:** District agricultural contingency planning officers, state disaster management authorities, rural livelihood NGOs, and agricultural insurance planners.
- **Intended Scope:** Strategic early-warning resource allocation, prioritization of contingency crop advisories, targeting of drought-proofing investments, and livelihood safety net mobilization.
- **Out-of-Scope Use:** Should NOT be used for automated claim settlement, individual farm credit denial, or immediate localized disaster compensation without field verification.

---

## Datasets & Provenance
1. **Climate:** India Meteorological Department (IMD) Pune daily gridded rainfall (0.25°, 1990–2020) and temperature (1.0° regridded to 0.25°, 1990–2020).
2. **Agriculture:** ICRISAT Apportioned Database (1966–2017) covering 571 districts across 19 states mapped to consistent 1966 parent baselines.
3. **Demographics:** Office of the Registrar General of India, Census 2011 Primary Census Abstract (PCA).
4. **Independent Validation:** NITI Aayog National Multidimensional Poverty Index (Baseline NFHS-4 & Progress Review NFHS-5).

---

## Architectural Separation & Anti-Circularity Rule
- **Module A (Structural Vulnerability Index):** Descriptive, static IPCC composite index (Exposure + Sensitivity + Lack of Adaptive Capacity). Validated **strictly externally** against NITI Aayog MPI ($r_s = 0.6120, p < 10^{-30}$). MPI is never fed into the index formula.
- **Module B (Shock Early-Warning):** Strictly forward-looking predictive model targeting observed harvest yield shocks using only information available by July 31 (Zero Leakage).

---

## Quantitative Performance
### 1. Temporal Holdout (Trained: 1991–2012, Tested: 2013–2017)
- **Test Base Rate:** 16.65% (Natural shock rate)
- **LightGBM ROC-AUC:** 0.5913
- **LightGBM PR-AUC:** 0.2422 (+45.5% relative gain over climatology baseline of 0.1665)
- **Recall at Top-20% Prioritized Districts:** 30.0%
- **Brier Score (Calibrated):** 0.1371 (vs Raw: 0.2226)

### 2. Spatial Cross-Validation (GroupKFold by Unseen State)
- **Mean ROC-AUC:** 0.6432 (± 0.0203)
- **Mean PR-AUC:** 0.2813 (± 0.0200) (+69.0% gain over base rate)
- **Mean Recall at Top-20%:** 35.3%

---

## Known Limitations
1. **Spatial Aggregation:** District-level averages conceal intra-district, block-, and village-level disparities.
2. **Relative Index:** The IPCC structural vulnerability score is a relative ranking within the study cohort, not an absolute measure of poverty or destitution.
3. **Observational Associations:** SHAP feature attributions represent statistical predictive associations, not verified causal mechanisms.
4. **Historical Coverage:** ICRISAT apportioned agricultural panel spans major agricultural states up to 2017; refreshed operational deployment requires continuous DES data ingestion.
5. **Decision Support:** This tool is designed to complement, not replace, ground assessments and field agronomist evaluations.
"""
    with open(output_path, "w") as f:
        f.write(content)
    print(f"[EXPORT] Model card written to {output_path}")


def main():
    config = load_config()
    processed_dir = config["paths"]["processed_dir"]
    app_dir = config["paths"]["app_dir"]
    os.makedirs(app_dir, exist_ok=True)

    print("[INIT] Verifying and packaging artifacts for Streamlit app...")

    # 1. Build and export Model Card
    build_model_card(os.path.join(app_dir, "model_card.md"))

    # 2. Build GeoJSON with embedded latest predictions (2017)
    geojson_path = os.path.join(processed_dir, "districts.geojson")
    preds_path = os.path.join(processed_dir, "predictions.parquet")
    vuln_path = os.path.join(processed_dir, "vulnerability_index_districts.parquet")

    gdf = gpd.read_file("data/raw/shapefile/2011_Dist.shp")
    gdf_simple = gdf.copy()
    gdf_simple["geometry"] = gdf.geometry.simplify(0.015, preserve_topology=True)

    df_preds = pd.read_parquet(preds_path)
    # Take most recent test year (2017)
    recent_preds = df_preds[df_preds["year"] == df_preds["year"].max()].copy()
    df_vuln = pd.read_parquet(vuln_path)

    # Merge into GeoDataFrame
    gdf_simple = gdf_simple.merge(
        recent_preds[["censuscode", "shock_probability", "priority_score", "top_driver_1", "action_1"]],
        on="censuscode", how="left"
    )
    gdf_simple = gdf_simple.merge(
        df_vuln[["censuscode", "vulnerability_class", "vulnerability_index_equal"]],
        on="censuscode", how="left"
    )

    gdf_simple.to_file(geojson_path, driver="GeoJSON")
    print(f"[EXPORT] Simplified GeoJSON with embedded district risk properties saved to {geojson_path}")

    # 3. Verify Artifact Manifest
    required_files = [
        "models/shock_model.joblib",
        "models/calibrator.joblib",
        "models/imputer.joblib",
        "models/feature_list.json",
        "data/processed/master_panel.parquet",
        "data/processed/vulnerability_index_districts.parquet",
        "data/processed/predictions.parquet",
        "data/processed/shap_values.parquet",
        "data/processed/districts.geojson",
        "reports/metrics.json",
        "reports/figures/roc_curves.png",
        "reports/figures/pr_curves.png",
        "reports/figures/calibration_plot.png",
        "reports/figures/shap_summary.png",
        "reports/figures/vulnerability_distribution.png",
        "app/model_card.md"
    ]

    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"Missing required app artifacts: {missing}")

    print("\n=======================================================")
    print(" [SUCCESS] All 16 App & Modeling Artifacts Verified!")
    for f in required_files:
        print(f"  ✓ {f}")
    print("=======================================================")


if __name__ == "__main__":
    main()

