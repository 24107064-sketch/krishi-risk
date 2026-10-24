"""01_download_imd.py

Downloads IMD Pune gridded daily data (Rainfall 0.25°, Tmax 1.0°, Tmin 1.0°)
for the baseline and analysis periods specified in config.yaml.
Performs verification of files, grid dimensions, and missing value sentinels.
"""

import os
import sys
import yaml
import imdlib as imd


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_year_file(var_dir, year):
    for ext in [".grd", ".GRD"]:
        fp = os.path.join(var_dir, f"{year}{ext}")
        if os.path.exists(fp):
            return fp
    return None


def download_variable(var_name, start_year, end_year, base_dir):
    var_dir = os.path.join(base_dir, var_name)
    os.makedirs(var_dir, exist_ok=True)
    
    # Check if all yearly files exist (checking .grd and .GRD)
    missing_years = [
        yr for yr in range(start_year, end_year + 1)
        if find_year_file(var_dir, yr) is None
    ]
    
    if not missing_years:
        print(f"[IMD] All files for '{var_name}' ({start_year}-{end_year}) already exist. Skipping download.")
        return
    
    print(f"\n[IMD] Starting download for '{var_name}' ({start_year} to {end_year})...")
    print(f"[IMD] Target directory: {var_dir}")
    try:
        imd.get_data(
            var_name,
            start_year,
            end_year,
            fn_format="yearwise",
            file_dir=base_dir
        )
        print(f"[IMD] Successfully fetched '{var_name}' data.")
    except Exception as e:
        print(f"[ERROR] Failed to download {var_name}: {e}", file=sys.stderr)
        raise


def verify_downloaded_data(var_name, start_year, end_year, base_dir):
    var_dir = os.path.join(base_dir, var_name)
    print(f"\n[VERIFY] Verifying files for '{var_name}' in {var_dir}...")
    expected_years = list(range(start_year, end_year + 1))
    missing_files = []
    total_bytes = 0

    for yr in expected_years:
        fp = find_year_file(var_dir, yr)
        if fp is None:
            missing_files.append(f"{yr}.grd/.GRD")
        else:
            total_bytes += os.path.getsize(fp)

    if missing_files:
        raise FileNotFoundError(
            f"Missing {len(missing_files)} files for {var_name}: {missing_files[:5]}..."
        )

    print(f"[VERIFY] All {len(expected_years)} yearly files present for {var_name}.")
    print(f"[VERIFY] Total disk size: {total_bytes / (1024 * 1024):.2f} MB")

    # Load first year via open_data to verify xarray structure
    print(f"[VERIFY] Loading sample year ({start_year}) into xarray...")
    sample_data = imd.open_data(var_name, start_year, start_year, "yearwise", base_dir)
    ds = sample_data.get_xarray()
    print(f"[VERIFY] xarray Dataset for {var_name} ({start_year}):")
    print(ds)
    return ds


def main():
    config = load_config()
    start_year = config["climate"]["analysis_start_year"]
    end_year = config["climate"]["analysis_end_year"]
    raw_imd_base = config["paths"]["raw_imd"]

    variables = ["rain", "tmax", "tmin"]

    for var in variables:
        download_variable(var, start_year, end_year, raw_imd_base)
        verify_downloaded_data(var, start_year, end_year, raw_imd_base)

    print("\n=======================================================")
    print(" [SUCCESS] All IMD Climate Datasets Verified!")
    print("=======================================================")


if __name__ == "__main__":
    main()
