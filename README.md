# Krishi Risk: District-Level Rural Livelihood Vulnerability and Agricultural Shock Early-Warning System (India)

**Author:** Shivam Mewada (TE Data Science, A.P. Shah Institute of Technology)  
**Supervisor / Lab:** Prof. Veena Trivedi (Research Methodology Lab)  
**Research Topic:** *Data-Driven Prediction of Rural Livelihood Vulnerability Under Climate and Economic Risks in India*

---

## 1. Project Overview & Core Architecture
Naïve approaches to vulnerability prediction construct an ad-hoc vulnerability score and subsequently train machine learning models to predict that same score from identical indicators, creating a **circular feedback loop** that yields deceptively high accuracy without real-world utility. Furthermore, static vulnerability scores are purely descriptive and offer no predictive early warning.

**Krishi Risk** resolves this via a two-module decoupled architecture:
1. **Module A (Structural Vulnerability Index):** Descriptive, static IPCC composite index ($\text{Vulnerability} = \text{Exposure} + \text{Sensitivity} - \text{Adaptive Capacity}$). Validated **strictly independently** against external NITI Aayog Multidimensional Poverty Index data ($r_s = 0.612, p = 2.15 \times 10^{-31}$). Zero circularity.
2. **Module B (Agricultural Shock Early-Warning System):** Forward-looking predictive ML system targeting observed, detrended harvest yield failures ($> 1$ SD below district trend). Forecasts risk 3 months in advance (as of July 31) using only observable early-season climate anomalies and structural buffers.
3. **Module C (Decision Support Layer):** Combines Module A vulnerability classes and Module B calibrated shock probabilities into a transparent 5-tier Action Priority Matrix with SHAP risk driver explanations and recommended policy interventions.

---

## 2. Directory Layout
```text
SATY_M/
├── app/
│   ├── main.py                   # Full-featured Streamlit interactive web app
│   └── model_card.md             # Standardized model card (intended use & limitations)
├── config.yaml                   # Central pipeline parameters, seeds, and thresholds
├── data/
│   ├── DATA_DICTIONARY.md        # Comprehensive data dictionary
│   ├── SOURCES.md                # Data provenance registry (URLs, dates, licenses)
│   ├── raw/
│   │   ├── census2011/           # Census 2011 Primary Census Abstract (PCA)
│   │   ├── icrisat/              # ICRISAT/TCI Apportioned 1966 Database (Crops, Irrigation, Land Use)
│   │   ├── imd/                  # IMD Pune daily gridded binary .grd (rain, tmax, tmin)
│   │   ├── niti_mpi/             # NITI Aayog National MPI district tables
│   │   └── shapefile/            # Census 2011 district boundary shapefiles
│   └── processed/
│       ├── agri_district_season_year.parquet  # Detrended crop yields & shock labels
│       ├── climate_district_year.parquet      # Zonal climate anomalies & early features
│       ├── districts.geojson                  # Simplified GeoJSON with embedded risk properties
│       ├── master_panel.parquet               # Master joined panel (8,662 district-years)
│       ├── name_matching_table.csv            # District alias reconciliation log
│       ├── predictions.parquet                # Calibrated test probabilities & SHAP drivers
│       ├── shap_values.parquet                # Full test-set SHAP attribution matrix
│       ├── static_district_features.parquet   # Census & demographic indicators
│       └── vulnerability_index_districts.parquet # Module A IPCC index & tertile classes
├── models/
│   ├── calibrator.joblib         # Isotonic probability calibrator
│   ├── feature_list.json         # Master feature schema
│   ├── imputer.joblib            # Median imputer artifact
│   └── shock_model.joblib        # Trained LightGBM early-warning model
├── reports/
│   ├── figures/                  # Publication-quality ROC, PR, calibration, SHAP curves
│   ├── metrics.json              # Full evaluation benchmark tables
│   └── research_paper_draft.md   # Complete academic paper draft mapped to 5 RQs
├── requirements.txt              # Pinned Python 3.12 dependencies
└── src/
    ├── 01_download_imd.py        # Automated IMD climate data fetcher & verifier
    ├── 02_climate_district.py    # Regionmask spatial aggregation & historical checks
    ├── 03_build_agri.py          # Yield detrending & shock label construction
    ├── 04_static_features.py     # Census PCA & NITI MPI integration
    ├── 05_build_index.py         # Module A IPCC weighting & independent validation
    ├── 06_train_shock_model.py   # Module B baselines vs ML, temporal & spatial CV
    ├── 07_explain.py             # SHAP TreeExplainer & figure generation
    ├── 08_export_artifacts.py    # App packaging & validation
    └── download_icrisat.py       # ICRISAT DLD API bulk ingestion script
```

---

## 3. Pipeline Execution (Reproducible from Zero)
All steps are fully scripted and numbered:

```bash
# 1. Environment Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Climate Ingestion & Aggregation
python src/01_download_imd.py
python src/02_climate_district.py

# 3. Agricultural Detrending & Socioeconomic Panel Construction
python src/download_icrisat.py
python src/03_build_agri.py
python src/04_static_features.py

# 4. Module A: Structural Vulnerability Index & Validation
python src/05_build_index.py

# 5. Module B: Early-Warning Model Training & Evaluation
python src/06_train_shock_model.py

# 6. SHAP Attribution & Publication Figures
python src/07_explain.py

# 7. Packaging & Model Card
python src/08_export_artifacts.py

# 8. Launch Streamlit Web App
streamlit run app/main.py
```

---

## 4. Key Quantitative Results
### Module A: Structural Vulnerability Validation
- **External Spearman Rank Correlation vs NITI Aayog MPI:** $r_s = 0.6120$ ($p = 2.15 \times 10^{-31}$)
- **External Pearson Correlation vs NITI Aayog MPI:** $r = 0.6306$ ($p = 8.85 \times 10^{-34}$)
- **Weighting Sensitivity (Equal Weights vs OECD PCA Weights):** $r_s = 0.9590$ (Mean rank shift: 18.5 ranks out of 296).

### Module B: Shock Early-Warning Prediction (3 Months in Advance)
- **Temporal Hold-Out (Trained: 1991–2012, Tested: 2013–2017):**
  - Natural Base Rate: 16.65%
  - Climatology Baseline PR-AUC: 0.1665
  - Persistence Baseline PR-AUC: 0.1704
  - Logistic Regression PR-AUC: 0.2371 (ROC-AUC: 0.5758)
  - Random Forest PR-AUC: 0.2324 (ROC-AUC: 0.5800)
  - XGBoost PR-AUC: 0.2371 (ROC-AUC: 0.5886)
  - **LightGBM (Winning Model) PR-AUC: 0.2422 (ROC-AUC: 0.5913, +45.5% gain over base rate)**
  - **Recall @ Top-20% Prioritized Districts: 30.0%**
  - **Calibrated Brier Score: 0.1371** (cut down from raw 0.2226 via isotonic calibration).
- **Spatial Cross-Validation (Leave-State-Out GroupKFold):**
  - **LightGBM Mean ROC-AUC: 0.6432 (± 0.0203)**
  - **LightGBM Mean PR-AUC: 0.2813 (± 0.0200, +69.0% gain over base rate)**
  - **Mean Recall @ Top-20%: 35.3%**

---

## 5. Web Application Pages
Run `streamlit run app/main.py` to view:
1. **🏠 Home:** High-level project KPIs, architecture overview, and design rationale.
2. **🗺️ Risk Map:** High-resolution interactive choropleth toggling between Structural Vulnerability, Shock Probability, and Priority Scores.
3. **🔍 District Explorer:** District profile cards, calibrated shock probability gauge, top-3 SHAP risk drivers, suggested actions, and 10-year historical climate/yield graphs.
4. **🧪 What-If Simulator:** Sliders for rainfall deficit %, dry spell length, irrigation coverage %, and temperature anomaly simulating model risk responses.
5. **📊 Model Performance:** Full transparent comparison tables against baselines, interactive ROC/PR curves, calibration plots, and SHAP summary plots.
6. **📋 Priority Action List:** Ranked priority tables for district planners with CSV export.
7. **📚 Data & Methodology:** Complete data dictionary, mathematical equations, limitations, and embedded Model Card.

