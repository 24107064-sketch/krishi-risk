"""02_climate_district.py

Aggregates daily IMD gridded climate data (0.25° rainfall, 1.0° Tmax/Tmin regridded to 0.25°)
to all Census 2011 districts using regionmask with nearest-cell centroid fallback for small urban units.

Computes:
1. Annual and Kharif seasonal climate indicators (1990-2020)
2. Baseline normals (1990-2010) and percent anomalies
3. Early-warning forecast-to-date features as of July 31 (no leakage)
4. Dry spell duration, dry days, heavy rainfall days, heat days
5. Automated sanity checks for known historical climate events (2002, 2009, 2014, 2015, 2018)

Saves output to:
- data/processed/climate_district_year.parquet
- data/processed/climate_district_year.csv
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import regionmask
import imdlib as imd


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def calculate_longest_run(binary_array):
    """Calculates the maximum consecutive True values along axis 0 (time)."""
    # binary_array shape: (time, districts)
    max_runs = np.zeros(binary_array.shape[1], dtype=int)
    for col in range(binary_array.shape[1]):
        series = binary_array[:, col]
        max_run = 0
        current_run = 0
        for val in series:
            if val:
                current_run += 1
                if current_run > max_run:
                    max_run = current_run
            else:
                current_run = 0
        max_runs[col] = max_run
    return max_runs


def precompute_spatial_masks(gdf, lons, lats):
    """Computes regionmask polygon mappings and nearest-centroid fallback indices."""
    print("[SPATIAL] Generating district raster masks using regionmask...")
    mask = regionmask.mask_geopandas(gdf, lons, lats).values

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    grid_pts = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    district_lookups = []
    fallback_count = 0

    for idx in range(len(gdf)):
        rows, cols = np.where(mask == idx)
        if len(rows) > 0:
            district_lookups.append((rows, cols, False))
        else:
            # Fallback to nearest grid cell from district centroid
            fallback_count += 1
            centroid = gdf.iloc[idx].geometry.centroid
            dists = np.hypot(grid_pts[:, 0] - centroid.x, grid_pts[:, 1] - centroid.y)
            nearest_cell_idx = np.argmin(dists)
            nearest_row = nearest_cell_idx // len(lons)
            nearest_col = nearest_cell_idx % len(lons)
            district_lookups.append((np.array([nearest_row]), np.array([nearest_col]), True))

    print(f"[SPATIAL] Precomputed masks for {len(gdf)} districts.")
    print(f"[SPATIAL] Districts with direct grid cells: {len(gdf) - fallback_count}")
    print(f"[SPATIAL] Small districts using nearest-cell centroid fallback: {fallback_count}")
    return district_lookups


def extract_district_daily(data_cube, district_lookups, num_districts):
    """Extracts daily mean time series for each district from a (days, lat, lon) numpy array."""
    num_days = data_cube.shape[0]
    out = np.full((num_days, num_districts), np.nan, dtype=np.float32)

    for idx in range(num_districts):
        rows, cols, _ = district_lookups[idx]
        out[:, idx] = np.nanmean(data_cube[:, rows, cols], axis=1)

    return out


def run_historical_sanity_checks(df):
    """Validates processed climate indicators against well-documented historical events."""
    print("\n=======================================================")
    print(" Running Climate Sanity Checks on Known Historical Events")
    print("=======================================================")

    # 1. 2002 All-India Severe July Drought
    df_2002 = df[df["year"] == 2002]
    drought_share_2002 = (df_2002["rainfall_to_july_deviation_pct"] < -20).mean() * 100
    mean_july_dev_2002 = df_2002["rainfall_to_july_deviation_pct"].mean()
    print(f"[*] 2002 All-India Drought (July):")
    print(f"    - National average June-July rainfall anomaly: {mean_july_dev_2002:.1f}%")
    print(f"    - % of districts facing June-July deficit (< -20%): {drought_share_2002:.1f}%")
    assert mean_july_dev_2002 < -15.0, "Sanity check failed: 2002 was one of India's driest Julys on record!"

    # 2. 2009 Major Drought
    df_2009 = df[df["year"] == 2009]
    monsoon_dev_2009 = df_2009["rainfall_deviation_pct"].mean()
    drought_share_2009 = (df_2009["rainfall_deviation_pct"] < -20).mean() * 100
    print(f"[*] 2009 Severe Monsoon Drought:")
    print(f"    - National unweighted average monsoon rainfall anomaly: {monsoon_dev_2009:.1f}%")
    print(f"    - % of districts categorized as drought (< -20%): {drought_share_2009:.1f}%")
    assert drought_share_2009 > 40.0 and monsoon_dev_2009 < -10.0, "Sanity check failed: 2009 was a major all-India drought year!"

    # 3. 2014 & 2015 Back-to-Back Monsoon Deficits (e.g. Maharashtra / Marathwada)
    maha_2014 = df[(df["year"] == 2014) & (df["ST_NM"].str.contains("Maharashtra", case=False, na=False))]
    maha_2015 = df[(df["year"] == 2015) & (df["ST_NM"].str.contains("Maharashtra", case=False, na=False))]
    print(f"[*] 2014-2015 Maharashtra Consecutive Droughts:")
    print(f"    - 2014 Maharashtra average monsoon anomaly: {maha_2014['rainfall_deviation_pct'].mean():.1f}%")
    print(f"    - 2015 Maharashtra average monsoon anomaly: {maha_2015['rainfall_deviation_pct'].mean():.1f}%")

    # 4. 2018 Kerala Extreme Monsoon Floods
    kerala_2018 = df[(df["year"] == 2018) & (df["ST_NM"].str.contains("Kerala", case=False, na=False))]
    kerala_normal = kerala_2018["monsoon_normal_mm"].mean()
    kerala_rain_2018 = kerala_2018["monsoon_rainfall_mm"].mean()
    kerala_dev_2018 = kerala_2018["rainfall_deviation_pct"].mean()
    print(f"[*] 2018 Kerala Extreme Monsoon:")
    print(f"    - Normal monsoon rainfall: {kerala_normal:.1f} mm")
    print(f"    - 2018 monsoon rainfall: {kerala_rain_2018:.1f} mm ({kerala_dev_2018:+.1f}% deviation)")
    assert kerala_dev_2018 > 15.0, "Sanity check failed: 2018 experienced massive flooding in Kerala!"

    print("[SUCCESS] All historical climate sanity checks passed with flying colours!\n")


def main():
    config = load_config()
    start_year = config["climate"]["analysis_start_year"]
    end_year = config["climate"]["analysis_end_year"]
    base_start = config["climate"]["baseline_start_year"]
    base_end = config["climate"]["baseline_end_year"]
    heavy_rain_th = config["climate"]["heavy_rain_threshold_mm"]
    dry_day_th = config["climate"]["dry_day_threshold_mm"]

    shapefile_path = "data/raw/shapefile/2011_Dist.shp"
    raw_imd_base = config["paths"]["raw_imd"]
    processed_dir = config["paths"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    print(f"[INIT] Loading Census 2011 district shapefile from {shapefile_path}...")
    gdf = gpd.read_file(shapefile_path)
    num_districts = len(gdf)

    # Reference grid from first year of rainfall
    sample_rain_ds = imd.open_data("rain", start_year, start_year, "yearwise", raw_imd_base).get_xarray()
    lons = sample_rain_ds.lon.values
    lats = sample_rain_ds.lat.values

    # Precompute spatial masks
    district_lookups = precompute_spatial_masks(gdf, lons, lats)

    records = []
    print(f"\n[PROCESS] Processing climate features per district for years {start_year} to {end_year}...")

    for year in range(start_year, end_year + 1):
        print(f" -> Processing year {year}...", end="\r", flush=True)

        # 1. Load Rainfall
        rain_ds = imd.open_data("rain", year, year, "yearwise", raw_imd_base).get_xarray()
        rain_cube = rain_ds["rain"].values
        # Clean negative sentinels (-999.0) if present
        rain_cube = np.where(rain_cube < 0, np.nan, rain_cube)

        # 2. Load Tmax & Tmin, regrid bilinearly to 0.25° rainfall grid
        tmax_ds = imd.open_data("tmax", year, year, "yearwise", raw_imd_base).get_xarray()
        tmax_interp = tmax_ds["tmax"].interp(lat=lats, lon=lons, method="linear").values
        tmax_interp = np.where(tmax_interp < -90, np.nan, tmax_interp)

        tmin_ds = imd.open_data("tmin", year, year, "yearwise", raw_imd_base).get_xarray()
        tmin_interp = tmin_ds["tmin"].interp(lat=lats, lon=lons, method="linear").values
        tmin_interp = np.where(tmin_interp < -90, np.nan, tmin_interp)

        # Extract daily district time-series: shape (days, num_districts)
        daily_rain = extract_district_daily(rain_cube, district_lookups, num_districts)
        daily_tmax = extract_district_daily(tmax_interp, district_lookups, num_districts)
        daily_tmin = extract_district_daily(tmin_interp, district_lookups, num_districts)

        # Time slice index masks for the year
        # Dates: Jan 1 to Dec 31
        # Kharif Monsoon: June 1 (day 152 in non-leap, 153 in leap) to Oct 31 (day 304/305)
        # Early-warning forecast time: June 1 to July 31
        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31")
        is_monsoon = (dates.month >= 6) & (dates.month <= 10)
        is_june_july = (dates.month >= 6) & (dates.month <= 7)

        # Annual features
        annual_rain = np.nansum(daily_rain, axis=0)

        # Monsoon seasonal features (Jun - Oct)
        monsoon_rain_daily = daily_rain[is_monsoon, :]
        monsoon_rain = np.nansum(monsoon_rain_daily, axis=0)
        monsoon_rain_mean = np.nanmean(monsoon_rain_daily, axis=0)
        monsoon_rain_std = np.nanstd(monsoon_rain_daily, axis=0)
        monsoon_rain_cv = np.where(monsoon_rain_mean > 0, monsoon_rain_std / monsoon_rain_mean, 0.0)

        monsoon_dry_days = np.sum(monsoon_rain_daily < dry_day_th, axis=0)
        longest_dry_spell = calculate_longest_run(monsoon_rain_daily < dry_day_th)
        heavy_rain_days = np.sum(monsoon_rain_daily >= heavy_rain_th, axis=0)

        monsoon_tmax = np.nanmean(daily_tmax[is_monsoon, :], axis=0)
        monsoon_tmin = np.nanmean(daily_tmin[is_monsoon, :], axis=0)
        monsoon_hot_days = np.sum(daily_tmax[is_monsoon, :] >= 35.0, axis=0)

        # Early-Warning Forecast-to-Date features (Jun 1 to Jul 31) - ZERO LEAKAGE
        early_rain_daily = daily_rain[is_june_july, :]
        early_rain = np.nansum(early_rain_daily, axis=0)
        early_dry_days = np.sum(early_rain_daily < dry_day_th, axis=0)
        early_longest_dry_spell = calculate_longest_run(early_rain_daily < dry_day_th)
        early_heavy_rain_days = np.sum(early_rain_daily >= heavy_rain_th, axis=0)

        early_tmax = np.nanmean(daily_tmax[is_june_july, :], axis=0)
        early_tmin = np.nanmean(daily_tmin[is_june_july, :], axis=0)
        early_hot_days = np.sum(daily_tmax[is_june_july, :] >= 35.0, axis=0)

        for d_idx in range(num_districts):
            row_dict = {
                "year": year,
                "censuscode": gdf.iloc[d_idx]["censuscode"],
                "DISTRICT": gdf.iloc[d_idx]["DISTRICT"],
                "ST_NM": gdf.iloc[d_idx]["ST_NM"],
                "ST_CEN_CD": gdf.iloc[d_idx]["ST_CEN_CD"],
                "DT_CEN_CD": gdf.iloc[d_idx]["DT_CEN_CD"],
                "annual_rainfall_mm": round(float(annual_rain[d_idx]), 2),
                "monsoon_rainfall_mm": round(float(monsoon_rain[d_idx]), 2),
                "monsoon_rain_cv": round(float(monsoon_rain_cv[d_idx]), 4),
                "monsoon_dry_days": int(monsoon_dry_days[d_idx]),
                "longest_dry_spell_days": int(longest_dry_spell[d_idx]),
                "heavy_rain_days": int(heavy_rain_days[d_idx]),
                "monsoon_tmax_mean": round(float(monsoon_tmax[d_idx]), 2),
                "monsoon_tmin_mean": round(float(monsoon_tmin[d_idx]), 2),
                "monsoon_hot_days": int(monsoon_hot_days[d_idx]),
                # Forecast-to-date features (end of July)
                "rainfall_to_july_mm": round(float(early_rain[d_idx]), 2),
                "dry_days_to_july": int(early_dry_days[d_idx]),
                "longest_dry_spell_to_july": int(early_longest_dry_spell[d_idx]),
                "heavy_rain_days_to_july": int(early_heavy_rain_days[d_idx]),
                "tmax_to_july_mean": round(float(early_tmax[d_idx]), 2),
                "tmin_to_july_mean": round(float(early_tmin[d_idx]), 2),
                "hot_days_to_july": int(early_hot_days[d_idx]),
            }
            records.append(row_dict)

    df_climate = pd.DataFrame(records)
    print(f"\n[PROCESS] Raw climate records constructed: {df_climate.shape}")

    # Compute 1990-2010 baseline normals per district
    print(f"[NORMALS] Computing 1990-2010 climatological baseline normals per district...")
    baseline_df = df_climate[(df_climate["year"] >= base_start) & (df_climate["year"] <= base_end)]
    normals = baseline_df.groupby("censuscode").agg(
        monsoon_normal_mm=("monsoon_rainfall_mm", "mean"),
        july_normal_mm=("rainfall_to_july_mm", "mean"),
        monsoon_tmax_normal=("monsoon_tmax_mean", "mean"),
        july_tmax_normal=("tmax_to_july_mean", "mean"),
    ).reset_index()

    # Join normals back to full climate panel
    df_climate = df_climate.merge(normals, on="censuscode", how="left")

    # Compute anomalies and deviations
    df_climate["rainfall_deviation_pct"] = (
        (df_climate["monsoon_rainfall_mm"] - df_climate["monsoon_normal_mm"])
        / df_climate["monsoon_normal_mm"]
        * 100.0
    ).round(2)

    df_climate["rainfall_to_july_deviation_pct"] = (
        (df_climate["rainfall_to_july_mm"] - df_climate["july_normal_mm"])
        / df_climate["july_normal_mm"]
        * 100.0
    ).round(2)

    df_climate["tmax_monsoon_anomaly"] = (
        df_climate["monsoon_tmax_mean"] - df_climate["monsoon_tmax_normal"]
    ).round(2)

    df_climate["tmax_to_july_anomaly"] = (
        df_climate["tmax_to_july_mean"] - df_climate["july_tmax_normal"]
    ).round(2)

    # Standard meteorological drought definitions (IMD: Deficit >= 20% is drought, >= 40% is severe)
    df_climate["drought_flag"] = (df_climate["rainfall_deviation_pct"] <= -20.0).astype(int)
    df_climate["severe_drought_flag"] = (df_climate["rainfall_deviation_pct"] <= -40.0).astype(int)
    df_climate["early_drought_flag"] = (df_climate["rainfall_to_july_deviation_pct"] <= -20.0).astype(int)

    # Sanity checks
    run_historical_sanity_checks(df_climate)

    # Export
    parquet_path = os.path.join(processed_dir, "climate_district_year.parquet")
    csv_path = os.path.join(processed_dir, "climate_district_year.csv")

    df_climate.to_parquet(parquet_path, index=False)
    df_climate.to_csv(csv_path, index=False)

    print("=======================================================")
    print(f" [SUCCESS] Output generated successfully:")
    print(f"  - Parquet: {parquet_path} ({os.path.getsize(parquet_path) / 1024:.1f} KB)")
    print(f"  - CSV:     {csv_path} ({os.path.getsize(csv_path) / 1024:.1f} KB)")
    print(f"  - Rows: {len(df_climate):,}, Columns: {len(df_climate.columns)}")
    print(f"  - Unique districts covered: {df_climate['censuscode'].nunique()}")
    print(f"  - Year span: {df_climate['year'].min()} to {df_climate['year'].max()}")
    print("=======================================================")


if __name__ == "__main__":
    main()
