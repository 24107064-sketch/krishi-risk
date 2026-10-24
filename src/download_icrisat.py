"""download_icrisat.py

Downloads ICRISAT Apportioned 1966 Database tables from the official DLD API:
- area-production-yield (Crops)
- sourcewise-irrigated-area & cropwise-irrigated-area (Irrigation)
- fertilizer-consumption (Inputs)
- landuse (Biophysical / Land Use)
- districts metadata
Saves both parquet and csv formats into data/raw/icrisat/
"""

import os
import requests
import json
import pandas as pd

OUTPUT_DIR = "data/raw/icrisat"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ENDPOINTS = [
    ("area-production-yield", "apportioned/area-production-yield"),
    ("sourcewise-irrigated-area", "apportioned/sourcewise-irrigated-area"),
    ("cropwise-irrigated-area", "apportioned/cropwise-irrigated-area"),
    ("fertilizer-consumption", "apportioned/fertilizer-consumption"),
    ("landuse", "apportioned/landuse"),
    ("districts", "apportioned/districts"),
]

def download_and_save(name, path_suffix):
    url = f"http://data.icrisat.org/dldAPI/{path_suffix}"
    print(f"\n[ICRISAT] Downloading '{name}' from {url}...")
    
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    payload = response.json()
    
    if name == "districts":
        df = pd.DataFrame(payload)
    else:
        headers = [h["header"] for h in payload["headers"]]
        df = pd.DataFrame(payload["data"], columns=headers)
        
    csv_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    parquet_path = os.path.join(OUTPUT_DIR, f"{name}.parquet")
    
    df.to_csv(csv_path, index=False)
    # Convert object columns that can be numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    df.to_parquet(parquet_path, index=False)
    
    print(f"[ICRISAT] Saved {name}: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"          CSV: {csv_path} ({os.path.getsize(csv_path) / 1024:.1f} KB)")
    print(f"          Parquet: {parquet_path} ({os.path.getsize(parquet_path) / 1024:.1f} KB)")
    return df

def main():
    print("=======================================================")
    print(" Fetching ICRISAT Apportioned Datasets (1966-2017)")
    print("=======================================================")
    summary = {}
    for name, path_suffix in ENDPOINTS:
        try:
            df = download_and_save(name, path_suffix)
            summary[name] = {"rows": len(df), "cols": len(df.columns)}
        except Exception as e:
            print(f"[WARNING] Failed to fetch {name}: {e}")
            
    print("\n[SUCCESS] ICRISAT Download Summary:")
    for k, v in summary.items():
        print(f" - {k}: {v['rows']} rows, {v['cols']} columns")

if __name__ == "__main__":
    main()

