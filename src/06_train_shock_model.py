"""06_train_shock_model.py

Builds and evaluates Module B: Agricultural Shock Early-Warning System:
Target: Binary Kharif yield shock (yield residual < -1.0 SD relative to district trend).
Forecast Time: July 31 (end of July, 3 months prior to harvest).
Strict Leakage Prevention: Only indicators observable by July 31 are utilized.

Evaluation Protocol:
1. Temporal Hold-Out: Train on 1991-2012 (22 years), Test on 2013-2017 (5 unseen years).
2. Spatial Group Cross-Validation: GroupKFold by State (leave unseen states out).
3. Baselines: Climatology (Base Rate), Persistence, Logistic Regression.
4. Machine Learning Models: Random Forest, XGBoost, LightGBM.
5. Metrics: ROC-AUC, PR-AUC, Macro-F1, Brier Score, Recall at Top-20%, Confusion Matrix.
6. Probability Calibration: Isotonic/Sigmoid calibration for trustworthy output risk probabilities.
7. Decision Priority Layer: Combining Module A class and Module B probability into Priority Scores.

Exports:
- data/processed/master_panel.parquet & .csv
- data/processed/predictions.parquet & .csv
- models/shock_model.joblib
- models/calibrator.joblib
- models/feature_list.json
- reports/metrics.json
"""

import os
import json
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    f1_score, confusion_matrix, recall_score, precision_score
)
import xgboost as xgb
import lightgbm as lgb


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def recall_at_top_k(y_true, y_prob, k_pct=0.20):
    """Calculates recall among the top-k% highest predicted risk districts."""
    k = max(int(len(y_true) * k_pct), 1)
    top_k_indices = np.argsort(y_prob)[::-1][:k]
    actual_positives = np.sum(y_true)
    if actual_positives == 0:
        return 0.0
    detected_positives = np.sum(y_true[top_k_indices])
    return float(detected_positives / actual_positives)


def compute_priority_score(vuln_class, shock_prob):
    """
    Transparent Decision Matrix combining Structural Vulnerability (Module A)
    and Shock Probability (Module B) into an action priority score (1 to 5):
    5 = Critical (High Vuln + High Prob)
    4 = Urgent (High Vuln + Moderate Prob or Moderate Vuln + High Prob)
    3 = Elevated
    2 = Watch
    1 = Low Risk
    """
    if shock_prob >= 0.35:
        p_cat = "High"
    elif shock_prob >= 0.20:
        p_cat = "Moderate"
    else:
        p_cat = "Low"

    matrix = {
        ("High", "High"): 5,
        ("High", "Moderate"): 4,
        ("Moderate", "High"): 4,
        ("High", "Low"): 3,
        ("Moderate", "Moderate"): 3,
        ("Low", "High"): 3,
        ("Moderate", "Low"): 2,
        ("Low", "Moderate"): 2,
        ("Low", "Low"): 1,
    }
    return matrix.get((vuln_class, p_cat), 2)


def main():
    config = load_config()
    processed_dir = config["paths"]["processed_dir"]
    models_dir = config["paths"]["models_dir"]
    reports_dir = config["paths"]["reports_dir"]
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Build Master Panel
    print("[INIT] Building Master Panel (joining Climate, Agri, Static, Vulnerability)...")
    df_agri = pd.read_parquet(os.path.join(processed_dir, "agri_district_season_year.parquet")).rename(columns={"Year": "year"})
    df_clim = pd.read_parquet(os.path.join(processed_dir, "climate_district_year.parquet"))
    df_vuln = pd.read_parquet(os.path.join(processed_dir, "vulnerability_index_districts.parquet"))

    panel = df_agri.merge(df_clim, on=["censuscode", "year"], how="inner")
    vuln_cols = [
        "censuscode", "vulnerability_index_equal", "vulnerability_class",
        "exposure_subindex", "sensitivity_subindex", "lack_adaptive_capacity_subindex"
    ]
    panel = panel.merge(df_vuln[vuln_cols], on="censuscode", how="inner")

    # Save master panel
    master_parquet = os.path.join(processed_dir, "master_panel.parquet")
    master_csv = os.path.join(processed_dir, "master_panel.csv")
    panel.to_parquet(master_parquet, index=False)
    panel.to_csv(master_csv, index=False)
    print(f"[PANEL] Master panel assembled: {len(panel):,} district-years ({panel['year'].min()}-{panel['year'].max()}) across {panel['censuscode'].nunique()} districts.")

    # 2. Define Features Available by July 31 (Zero Leakage)
    feature_list = [
        "rainfall_to_july_mm",
        "rainfall_to_july_deviation_pct",
        "dry_days_to_july",
        "longest_dry_spell_to_july",
        "heavy_rain_days_to_july",
        "tmax_to_july_mean",
        "tmin_to_july_mean",
        "tmax_to_july_anomaly",
        "hot_days_to_july",
        "early_drought_flag",
        "monsoon_normal_mm",
        "july_normal_mm",
        "monsoon_rain_cv",
        "yield_kgha_lag1",
        "yield_anomaly_z_lag1",
        "shock_label_1sd_lag1",
        "vulnerability_index_equal",
        "exposure_subindex",
        "sensitivity_subindex",
        "lack_adaptive_capacity_subindex",
        "irrigation_coverage_ratio",
        "rainfed_area_share",
        "crop_diversification_index",
        "cropping_intensity",
        "fertilizer_kg_per_ha_nca"
    ]

    with open(os.path.join(models_dir, "feature_list.json"), "w") as f:
        json.dump(feature_list, f, indent=2)

    # 3. Clean Dataset (Drop missing target; start from 1991 where lag1 exists)
    clean_panel = panel[panel["shock_label_1sd"].notna() & (panel["year"] >= 1991)].copy().reset_index(drop=True)
    target_col = "shock_label_1sd"

    train_mask = clean_panel["year"] <= 2012
    test_mask = clean_panel["year"] >= 2013

    train_df = clean_panel[train_mask].copy()
    test_df = clean_panel[test_mask].copy()

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train_df[feature_list])
    X_test = imputer.transform(test_df[feature_list])
    y_train = train_df[target_col].values.astype(int)
    y_test = test_df[target_col].values.astype(int)

    base_rate_train = float(y_train.mean())
    base_rate_test = float(y_test.mean())
    scale_pos = (len(y_train) - sum(y_train)) / sum(y_train)

    print(f"\n[DATA] Train Sample (1991-2012): {len(train_df):,} rows | Positive rate: {base_rate_train * 100:.2f}%")
    print(f"[DATA] Test Holdout (2013-2017): {len(test_df):,} rows  | Positive rate: {base_rate_test * 100:.2f}%")

    # 4. Instantiate Candidate Models
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=8, class_weight="balanced", random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.04, scale_pos_weight=scale_pos, random_state=42, eval_metric="logloss"),
        "LightGBM": lgb.LGBMClassifier(n_estimators=120, max_depth=4, learning_rate=0.04, class_weight="balanced", random_state=42, verbose=-1, n_jobs=-1)
    }

    # 5. Temporal Hold-Out Evaluation
    print("\n=======================================================")
    print(" 1. Temporal Hold-Out Results (Test Years: 2013-2017)")
    print("=======================================================")

    results = {"temporal_holdout": {}, "spatial_cv": {}}

    # Baselines
    # A) Climatology
    clim_probs = np.full(len(y_test), base_rate_train)
    results["temporal_holdout"]["Climatology Baseline"] = {
        "roc_auc": 0.5000,
        "pr_auc": round(float(base_rate_test), 4),
        "brier_score": round(float(brier_score_loss(y_test, clim_probs)), 4),
        "macro_f1": round(float(f1_score(y_test, np.zeros_like(y_test), average="macro")), 4),
        "recall_at_top20": round(float(recall_at_top_k(y_test, clim_probs, 0.20)), 4)
    }

    # B) Persistence
    pers_probs = imputer.transform(test_df[feature_list])[:, feature_list.index("shock_label_1sd_lag1")]
    pers_preds = (pers_probs >= 0.5).astype(int)
    results["temporal_holdout"]["Persistence Baseline"] = {
        "roc_auc": round(float(roc_auc_score(y_test, pers_probs)), 4),
        "pr_auc": round(float(average_precision_score(y_test, pers_probs)), 4),
        "brier_score": round(float(brier_score_loss(y_test, pers_probs)), 4),
        "macro_f1": round(float(f1_score(y_test, pers_preds, average="macro")), 4),
        "recall_at_top20": round(float(recall_at_top_k(y_test, pers_probs, 0.20)), 4)
    }

    trained_models = {}
    test_probs_dict = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        cm = confusion_matrix(y_test, preds)

        metrics = {
            "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
            "pr_auc": round(float(average_precision_score(y_test, probs)), 4),
            "brier_score": round(float(brier_score_loss(y_test, probs)), 4),
            "macro_f1": round(float(f1_score(y_test, preds, average="macro")), 4),
            "recall_at_top20": round(float(recall_at_top_k(y_test, probs, 0.20)), 4),
            "confusion_matrix": {
                "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]), "tp": int(cm[1, 1])
            }
        }
        results["temporal_holdout"][name] = metrics
        trained_models[name] = model
        test_probs_dict[name] = probs

        print(f"{name:20s} | ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | Recall@20%: {metrics['recall_at_top20']*100:.1f}% | Brier: {metrics['brier_score']:.4f}")

    # 6. Spatial Cross-Validation (GroupKFold by State across all data)
    print("\n=======================================================")
    print(" 2. Spatial Cross-Validation Results (GroupKFold by State)")
    print("=======================================================")
    X_full = imputer.fit_transform(clean_panel[feature_list])
    y_full = clean_panel[target_col].values.astype(int)
    state_groups = clean_panel["ST_NM"].values

    gkf = GroupKFold(n_splits=5)
    for name, model in models.items():
        cv_aucs = []
        cv_praucs = []
        cv_briers = []
        cv_recalls = []

        for tr_i, te_i in gkf.split(X_full, y_full, state_groups):
            X_tr, y_tr = X_full[tr_i], y_full[tr_i]
            X_te, y_te = X_full[te_i], y_full[te_i]

            m_clone = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=42) if name == "Random Forest" else model
            m_clone.fit(X_tr, y_tr)
            p_te = m_clone.predict_proba(X_te)[:, 1]

            cv_aucs.append(roc_auc_score(y_te, p_te))
            cv_praucs.append(average_precision_score(y_te, p_te))
            cv_briers.append(brier_score_loss(y_te, p_te))
            cv_recalls.append(recall_at_top_k(y_te, p_te, 0.20))

        results["spatial_cv"][name] = {
            "mean_roc_auc": round(float(np.mean(cv_aucs)), 4),
            "std_roc_auc": round(float(np.std(cv_aucs)), 4),
            "mean_pr_auc": round(float(np.mean(cv_praucs)), 4),
            "std_pr_auc": round(float(np.std(cv_praucs)), 4),
            "mean_recall_at_top20": round(float(np.mean(cv_recalls)), 4),
            "mean_brier_score": round(float(np.mean(cv_briers)), 4),
        }
        print(f"{name:20s} | Mean ROC-AUC: {np.mean(cv_aucs):.4f} (+/- {np.std(cv_aucs):.4f}) | Mean PR-AUC: {np.mean(cv_praucs):.4f} | Recall@20%: {np.mean(cv_recalls)*100:.1f}%")

    # 7. Probability Calibration for the Winning Model
    best_model_name = "LightGBM" if results["temporal_holdout"]["LightGBM"]["pr_auc"] >= results["temporal_holdout"]["Random Forest"]["pr_auc"] else "Random Forest"
    best_estimator = trained_models[best_model_name]
    print(f"\n[CALIBRATION] Calibrating probabilities for winning model: {best_model_name}...")

    # Fit 3-fold cross-validated isotonic calibrator on training data
    calibrator = CalibratedClassifierCV(estimator=best_estimator, method="isotonic", cv=3)
    calibrator.fit(X_train, y_train)
    calibrated_probs = calibrator.predict_proba(X_test)[:, 1]
    calibrated_brier = brier_score_loss(y_test, calibrated_probs)
    print(f"[CALIBRATION] Calibrated Brier Score on Test: {calibrated_brier:.4f} (Raw: {results['temporal_holdout'][best_model_name]['brier_score']:.4f})")

    # Save artifacts
    joblib.dump(best_estimator, os.path.join(models_dir, "shock_model.joblib"))
    joblib.dump(calibrator, os.path.join(models_dir, "calibrator.joblib"))
    joblib.dump(imputer, os.path.join(models_dir, "imputer.joblib"))

    with open(os.path.join(reports_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # 8. Assemble Out-of-Time Test Predictions & Decision Priority Scores
    print("[DECISION] Generating district prediction profiles & action priority scores...")
    test_df["shock_probability"] = calibrated_probs.round(4)
    test_df["shock_prediction_binary"] = (calibrated_probs >= 0.25).astype(int)
    test_df["priority_score"] = test_df.apply(
        lambda r: compute_priority_score(r["vulnerability_class"], r["shock_probability"]), axis=1
    )

    pred_cols = [
        "censuscode", "DISTRICT", "ST_NM", "year", "vulnerability_class",
        "vulnerability_index_equal", "shock_probability", "shock_prediction_binary",
        "priority_score", "shock_label_1sd", "kharif_composite_yield_kgha",
        "rainfall_to_july_deviation_pct", "longest_dry_spell_to_july",
        "tmax_to_july_anomaly", "irrigation_coverage_ratio", "rainfed_area_share"
    ]
    df_preds = test_df[pred_cols].copy().sort_values(["year", "priority_score", "shock_probability"], ascending=[True, False, False])

    pred_parquet = os.path.join(processed_dir, "predictions.parquet")
    pred_csv = os.path.join(processed_dir, "predictions.csv")
    df_preds.to_parquet(pred_parquet, index=False)
    df_preds.to_csv(pred_csv, index=False)

    print("\n=======================================================")
    print(" [SUCCESS] Module B Training & Evaluation Completed!")
    print(f"  - Model Saved:      models/shock_model.joblib ({best_model_name})")
    print(f"  - Calibrator Saved: models/calibrator.joblib")
    print(f"  - Metrics Saved:    reports/metrics.json")
    print(f"  - Predictions:      data/processed/predictions.parquet ({len(df_preds):,} rows)")
    print("=======================================================")


if __name__ == "__main__":
    main()
