# Data Sources Registry: Krishi Risk Project

This document maintains full provenance, citations, licenses, and access details for all datasets used in the pipeline.

---

## Tier 1: Core Datasets

### 1. IMD Gridded Rainfall & Temperature
- **Source Agency:** India Meteorological Department (IMD), National Climate Centre, Pune.
- **Access Method:** Programmatic download via Python `imdlib` (`imd.get_data` / `imd.load`).
- **Portal URL:** https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html
- **Variables & Spatial Resolution:**
  - Rainfall: 0.25° x 0.25° grid (~25 km), daily binary files (`.grd`).
  - Maximum & Minimum Temperature (`tmax`, `tmin`): 1.0° x 1.0° grid (~100 km), daily binary files (`.grd`).
- **Temporal Range:** 1990–2020 (Rainfall available 1901–present, Temp 1951–present).
- **Spatial Level:** Regular latitude-longitude gridded raster.
- **License / Terms:** Open access for research & academic purposes with attribution to IMD Pune.
- **Storage Path:** `data/raw/imd/rain/`, `data/raw/imd/tmax/`, `data/raw/imd/tmin/`

---

### 2. ICRISAT / TCI District Level Database (DLD)
- **Source Agency:** International Crops Research Institute for the Semi-Arid Tropics (ICRISAT) & Tata-Cornell Institute (TCI).
- **Access Portal:** http://data.icrisat.org/dld/
- **Dataset Version:** **Apportioned 1966 Database** (maps newer split districts back to 1966 parent boundary baselines to maintain consistent long-term panel continuity without spurious structural breaks).
- **Coverage:** 571 districts across 20 major agricultural states.
- **Temporal Range:** 1966–2017 (announcement for 2019–20 extension to be verified).
- **Variables Collected:**
  - Crop area, production, and yield (kharif and rabi seasons for major cereals, pulses, oilseeds).
  - Irrigation by source (canals, tube-wells, tanks).
  - Land use patterns, operational holdings, agricultural workforce, fertilizer consumption.
- **License / Terms:** Open research data; citation required.
- **Storage Path:** `data/raw/icrisat/`

---

### 3. District Boundary Shapefiles (Census 2011)
- **Source:** DataMeet Maps Community Open Repository.
- **Access URL:** https://github.com/datameet/maps/tree/master/Districts
- **Files:** `Census_2011.*` (`.shp`, `.shx`, `.dbf`, `.prj`) or GeoJSON.
- **Spatial Level:** 640 districts (Census 2011 definition).
- **Coordinate Reference System (CRS):** EPSG:4326 (WGS84).
- **License / Terms:** Creative Commons Attribution 2.5 India (CC BY 2.5 IN).
- **Storage Path:** `data/raw/shapefile/`

---

### 4. Census 2011 Primary Census Abstract (PCA)
- **Source:** Office of the Registrar General & Census Commissioner, India / SHRUG (Development Data Lab).
- **Access Portal:** https://www.devdatalab.org/shrug_download/ or https://censusindia.gov.in
- **Variables:** Total population, rural population, cultivator & agricultural labourer counts, total/female literacy rates.
- **Spatial Level:** District level (Census 2011 district codes).
- **Storage Path:** `data/raw/census2011/`

---

## Tier 2: Adaptive Capacity & Validation Datasets

### 5. NFHS-4 (2015-16) & NFHS-5 (2019-21) Factsheets
- **Source:** International Institute for Population Sciences (IIPS), MoHFW.
- **Access Portal:** http://rchiips.org/nfhs/
- **Variables:** Household electricity coverage, improved drinking water, sanitation facilities, clean cooking fuel, health insurance coverage, female literacy.
- **Spatial Level:** District level factsheets.
- **Storage Path:** `data/raw/nfhs/`

---

### 6. NITI Aayog National Multidimensional Poverty Index (MPI)
- **Source:** NITI Aayog, Government of India.
- **Reports:**
  - Baseline Report (2021, based on NFHS-4 2015-16).
  - Progress Review (2023, based on NFHS-5 2019-21).
- **Variables:** Headcount ratio (H %), Intensity (A %), MPI score.
- **Role in Research:** **Strictly external validation** of the Module A Structural Vulnerability Index via Spearman rank correlation ($r_s$). Never used as an input feature into the index to prevent circularity.
- **Storage Path:** `data/raw/niti_mpi/`

