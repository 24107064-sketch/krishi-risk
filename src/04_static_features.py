"""04_static_features.py

Consolidates district-level static structural, demographic, and socioeconomic indicators:
1. Official Census 2011 Primary Census Abstract (PCA):
   - Female and overall literacy rates
   - Cultivator, agricultural labourer, and total agricultural workforce shares
   - SC/ST social vulnerability share
   - Total population and household counts
2. ICRISAT Operational Holdings:
   - Smallholder and marginal farm holding share (< 2 ha)
3. NITI Aayog Multidimensional Poverty Index (MPI Headcount Ratio %):
   - Strictly reserved for external independent validation of Module A
4. Documented district matching table
5. Exports:
   - data/processed/static_district_features.parquet
   - data/processed/static_district_features.csv
   - data/processed/name_matching_table.csv
"""

import os
import re
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def clean_name(s):
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


NAME_ALIASES = {
    "garhchiroli": "gadchiroli",
    "bauda": "boudh",
    "nagappattinam": "nagapattinam",
    "virudunagar": "virudhunagar",
    "north24parganas": "northtwentyfourparganas",
    "south24parganas": "southtwentyfourparganas",
    "kanpur": "kanpurnagar",
    "kochbihar": "coochbehar",
    "baramula": "baramulla",
    "badgam": "budgam",
    "shupiyan": "shopian",
    "punch": "poonch",
    "chittaurgarh": "chittorgarh",
    "dhaulpur": "dholpur",
    "jalor": "jalore",
    "lahulandspiti": "lahulspiti",
    "sonapur": "subarnapur",
    "baleshwar": "balasore",
    "hugli": "hooghly",
    "haora": "howrah",
    "pashchimichamparan": "westchamparan",
    "purbichamparan": "eastchamparan",
    "ribhoi": "ribhoi",
    "marigaon": "morigaon",
    "sibsagar": "sivasagar",
    "lehladdakh": "leh",
    "ahmadabad": "ahmedabad",
    "ahmadnagar": "ahmednagar",
}


def main():
    config = load_config()
    shapefile_path = "data/raw/shapefile/2011_Dist.shp"
    raw_census_dir = config["paths"]["raw_census"]
    raw_niti_dir = config["paths"]["raw_niti_mpi"]
    raw_icrisat_dir = config["paths"]["raw_icrisat"]
    processed_dir = config["paths"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    print("[INIT] Loading Master District Shapefile...")
    gdf = gpd.read_file(shapefile_path)
    base_df = gdf[["censuscode", "DISTRICT", "ST_NM", "ST_CEN_CD", "DT_CEN_CD"]].copy()

    # 1. Official Census 2011 Primary Census Abstract
    pca_path = os.path.join(raw_census_dir, "pca-total.csv")
    print(f"[CENSUS] Loading official Census 2011 PCA from {pca_path}...")
    df_pca_raw = pd.read_csv(pca_path)
    pca_dist = df_pca_raw[df_pca_raw["Level"] == "DISTRICT"].copy().reset_index(drop=True)
    # The official 640 districts in pca-total map exactly 1:1 to censuscode 1..640
    pca_dist["censuscode"] = pca_dist.index + 1

    # Engineer exact Census 2011 indicators
    pca_dist["overall_literacy_rate"] = (
        (pca_dist["P_LIT"] / pca_dist["TOT_P"]) * 100.0
    ).round(2)

    pca_dist["female_literacy_rate"] = (
        (pca_dist["F_LIT"] / pca_dist["TOT_F"]) * 100.0
    ).round(2)

    pca_dist["sc_st_share_pct"] = (
        ((pca_dist["P_SC"] + pca_dist["P_ST"]) / pca_dist["TOT_P"]) * 100.0
    ).round(2)

    total_cultivators = pca_dist["MAIN_CL_P"] + pca_dist["MARG_CL_P"]
    total_ag_labourers = pca_dist["MAIN_AL_P"] + pca_dist["MARG_AL_P"]
    total_workers = pca_dist["TOT_WORK_P"]

    pca_dist["cultivator_share_pct"] = np.where(
        total_workers > 0,
        (total_cultivators / total_workers) * 100.0,
        np.nan
    ).round(2)

    pca_dist["ag_labourer_share_pct"] = np.where(
        total_workers > 0,
        (total_ag_labourers / total_workers) * 100.0,
        np.nan
    ).round(2)

    pca_dist["ag_workforce_share_pct"] = np.where(
        total_workers > 0,
        ((total_cultivators + total_ag_labourers) / total_workers) * 100.0,
        np.nan
    ).round(2)

    pca_dist["working_population_ratio"] = np.where(
        pca_dist["TOT_P"] > 0,
        (total_workers / pca_dist["TOT_P"]) * 100.0,
        np.nan
    ).round(2)

    pca_features = pca_dist[[
        "censuscode", "overall_literacy_rate", "female_literacy_rate",
        "sc_st_share_pct", "cultivator_share_pct", "ag_labourer_share_pct",
        "ag_workforce_share_pct", "working_population_ratio"
    ]]

    merged = base_df.merge(pca_features, on="censuscode", how="left")
    print(f"[MERGE] Merged Census PCA indicators across {merged['female_literacy_rate'].notna().sum()}/641 districts.")

    # 2. ICRISAT Operational Holdings (Marginal and Smallholder Farm Share)
    hold_path = os.path.join(raw_icrisat_dir, "operational-holdings.parquet")
    if os.path.exists(hold_path):
        print(f"[ICRISAT] Loading Operational Holdings from {hold_path}...")
        df_hold = pd.read_parquet(hold_path)
        df_hold_recent = df_hold[df_hold["Year"] == df_hold["Year"].max()].copy()
        code_map = {800: 80, 770: 77, 740: 74, 860: 86, 840: 84, 660: 66, 820: 82, 700: 70, 990: 99}
        df_hold_recent["censuscode"] = df_hold_recent["Indian census District Code 2011"].replace(code_map).astype(int)

        df_hold_recent["small_marginal_holdings_pct"] = np.where(
            df_hold_recent["TOTAL NUMBER"] > 0,
            ((df_hold_recent["MARGINAL NUMBER"] + df_hold_recent["SMALL NUMBER"]) / df_hold_recent["TOTAL NUMBER"]) * 100.0,
            np.nan
        ).round(2)

        hold_features = df_hold_recent[[
            "censuscode", "small_marginal_holdings_pct"
        ]].drop_duplicates(subset=["censuscode"])
        merged = merged.merge(hold_features, on="censuscode", how="left")

    # 3. NITI Aayog Multidimensional Poverty Index (MPI Headcount Ratio)
    mpi_path = os.path.join(raw_niti_dir, "niti_mpi_district.csv")
    if os.path.exists(mpi_path):
        print(f"[NITI MPI] Loading NITI Aayog MPI validation data from {mpi_path}...")
        df_mpi = pd.read_csv(mpi_path)
        df_mpi["clean_d"] = df_mpi["District"].apply(clean_name).replace(NAME_ALIASES)
        
        def parse_pct(val):
            if pd.isna(val):
                return np.nan
            try:
                return float(str(val).replace("%", "").strip())
            except ValueError:
                return np.nan

        df_mpi["mpi_headcount_ratio"] = df_mpi["MPI HCR"].apply(parse_pct)
        mpi_sub = df_mpi[["clean_d", "mpi_headcount_ratio"]].drop_duplicates(subset=["clean_d"])

        merged["clean_d"] = merged["DISTRICT"].apply(clean_name).replace(NAME_ALIASES)
        merged = merged.merge(mpi_sub, on="clean_d", how="left")
        merged.drop(columns=["clean_d"], inplace=True)
        print(f"[NITI MPI] Mapped MPI Headcount Ratio to {merged['mpi_headcount_ratio'].notna().sum()}/641 districts.")

    # 4. Save name matching table
    matching_table = merged[["censuscode", "DISTRICT", "ST_NM", "mpi_headcount_ratio"]].copy()
    matching_table["matched_mpi"] = matching_table["mpi_headcount_ratio"].notna()
    match_csv = os.path.join(processed_dir, "name_matching_table.csv")
    matching_table.to_csv(match_csv, index=False)

    # 5. Export
    out_parquet = os.path.join(processed_dir, "static_district_features.parquet")
    out_csv = os.path.join(processed_dir, "static_district_features.csv")
    merged.to_parquet(out_parquet, index=False)
    merged.to_csv(out_csv, index=False)

    print("\n=======================================================")
    print(f" [SUCCESS] Static features exported successfully:")
    print(f"  - Parquet: {out_parquet} ({os.path.getsize(out_parquet) / 1024:.1f} KB)")
    print(f"  - CSV:     {out_csv} ({os.path.getsize(out_csv) / 1024:.1f} KB)")
    print(f"  - Districts covered: {len(merged)}")
    print(f"  - Mean Female Literacy: {merged['female_literacy_rate'].mean():.1f}%")
    print(f"  - Mean Cultivator Share: {merged['cultivator_share_pct'].mean():.1f}%")
    print(f"  - Mean Ag Labourer Share: {merged['ag_labourer_share_pct'].mean():.1f}%")
    print(f"  - Mean Ag Workforce Share: {merged['ag_workforce_share_pct'].mean():.1f}%")
    print(f"  - Mean NITI MPI HCR: {merged['mpi_headcount_ratio'].mean():.2f}%")
    print("=======================================================")


if __name__ == "__main__":
    main()

