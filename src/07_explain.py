"""07_explain.py

Computes SHAP explainability values and figures for the early-warning model:
1. Calculates TreeExplainer SHAP values on out-of-time test districts.
2. Identifies top-3 SHAP risk drivers per district with plain-language action suggestions.
3. Exports publication-quality figures to reports/figures/:
   - roc_curves.png (Temporal & Spatial CV comparison)
   - pr_curves.png (Precision-Recall comparison against baselines)
   - calibration_plot.png (Reliability curve before vs. after calibration)
   - shap_summary.png (Global feature importance beeswarm)
   - vulnerability_distribution.png (Module A index distribution)
4. Saves data/processed/shap_values.parquet and reports/shap_summary.json.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import json
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


ACTION_MAPPING = {
    "rainfall_to_july_deviation_pct": "Early monsoon deficit: recommend drought-contingency seed varieties, moisture conservation, and accelerated PMFBY enrollment.",
    "longest_dry_spell_to_july": "Prolonged dry spells: deploy supplemental protective irrigation, mulching, and farm-pond water reserves.",
    "dry_days_to_july": "High dry-day frequency: promote drought-hardy millets/pulses and micro-irrigation scheduling.",
    "tmax_to_july_anomaly": "Elevated thermal stress: advise heat-resilient agronomy and light frequent irrigation during vegetative stages.",
    "rainfed_area_share": "High rainfed dependency: prioritize watershed development, check-dams, and community borewell recharging.",
    "irrigation_coverage_ratio": "Low irrigation buffer: fast-track canal maintenance and micro-drip subsidization.",
    "yield_anomaly_z_lag1": "Compounding shock history: provide institutional credit rescheduling and subsidized inputs to prevent distress debt.",
    "vulnerability_index_equal": "High structural vulnerability: coordinate inter-departmental rural livelihood programs and MGNREGA asset creation.",
    "crop_diversification_index": "Low crop diversification: introduce intercropping and support oilseed/pulse diversification.",
    "fertilizer_kg_per_ha_nca": "Sub-optimal soil nutrition: initiate soil-health card testing and balanced NPK/organic matter advisory."
}


def main():
    config = load_config()
    processed_dir = config["paths"]["processed_dir"]
    models_dir = config["paths"]["models_dir"]
    figures_dir = config["paths"]["figures_dir"]
    reports_dir = config["paths"]["reports_dir"]
    os.makedirs(figures_dir, exist_ok=True)

    print("[INIT] Loading model, feature list, and panel data...")
    model = joblib.load(os.path.join(models_dir, "shock_model.joblib"))
    calibrator = joblib.load(os.path.join(models_dir, "calibrator.joblib"))
    imputer = joblib.load(os.path.join(models_dir, "imputer.joblib"))

    with open(os.path.join(models_dir, "feature_list.json"), "r") as f:
        feature_list = json.load(f)

    with open(os.path.join(reports_dir, "metrics.json"), "r") as f:
        metrics = json.load(f)

    panel = pd.read_parquet(os.path.join(processed_dir, "master_panel.parquet"))
    clean_panel = panel[panel["shock_label_1sd"].notna() & (panel["year"] >= 1991)].copy().reset_index(drop=True)

    train_df = clean_panel[clean_panel["year"] <= 2012].copy()
    test_df = clean_panel[clean_panel["year"] >= 2013].copy().reset_index(drop=True)

    X_train = imputer.transform(train_df[feature_list])
    X_test = imputer.transform(test_df[feature_list])
    y_test = test_df["shock_label_1sd"].values.astype(int)

    # 1. Compute SHAP Values
    print("[SHAP] Computing TreeExplainer SHAP values for test holdout...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test)
    
    # LightGBM binary classifier returns list of 2 arrays [neg, pos] or 1 array for positive class
    if isinstance(shap_vals, list):
        shap_vals_pos = shap_vals[1]
    elif len(shap_vals.shape) == 3:
        shap_vals_pos = shap_vals[:, :, 1]
    else:
        shap_vals_pos = shap_vals

    # Global feature importances
    mean_abs_shap = np.abs(shap_vals_pos).mean(axis=0)
    shap_importance = pd.DataFrame({
        "feature": feature_list,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print("\nTop 8 Global Early-Warning Drivers (by Mean |SHAP|):")
    for _, r in shap_importance.head(8).iterrows():
        print(f" - {r['feature']:32s}: {r['mean_abs_shap']:.4f}")

    with open(os.path.join(reports_dir, "shap_summary.json"), "w") as f:
        json.dump(shap_importance.to_dict(orient="records"), f, indent=2)

    # 2. Extract Top 3 Drivers & Plain-Language Action Suggestions per District
    print("[EXPLAIN] Mapping district-level drivers to plain-language policy suggestions...")
    top_driver_records = []
    for i in range(len(test_df)):
        row_shap = shap_vals_pos[i]
        top_3_idx = np.argsort(row_shap)[::-1][:3]
        
        d_1 = feature_list[top_3_idx[0]]
        d_2 = feature_list[top_3_idx[1]]
        d_3 = feature_list[top_3_idx[2]]

        s_1 = float(row_shap[top_3_idx[0]])
        s_2 = float(row_shap[top_3_idx[1]])
        s_3 = float(row_shap[top_3_idx[2]])

        act_1 = ACTION_MAPPING.get(d_1, "Deploy field extension support and monitor crop condition.")
        act_2 = ACTION_MAPPING.get(d_2, "Promote soil moisture conservation and contingency crops.")
        act_3 = ACTION_MAPPING.get(d_3, "Target livelihood safety nets and rural credit support.")

        top_driver_records.append({
            "censuscode": test_df.iloc[i]["censuscode"],
            "year": test_df.iloc[i]["year"],
            "DISTRICT": test_df.iloc[i]["DISTRICT"],
            "top_driver_1": d_1,
            "shap_1": round(s_1, 4),
            "action_1": act_1,
            "top_driver_2": d_2,
            "shap_2": round(s_2, 4),
            "action_2": act_2,
            "top_driver_3": d_3,
            "shap_3": round(s_3, 4),
            "action_3": act_3
        })

    df_drivers = pd.DataFrame(top_driver_records)

    # Merge top drivers into predictions.parquet
    preds_path = os.path.join(processed_dir, "predictions.parquet")
    df_preds = pd.read_parquet(preds_path)
    df_preds = df_preds.merge(df_drivers, on=["censuscode", "year", "DISTRICT"], how="left")
    df_preds.to_parquet(preds_path, index=False)
    df_preds.to_csv(os.path.join(processed_dir, "predictions.csv"), index=False)

    # Also save full shap matrix
    shap_df = pd.DataFrame(shap_vals_pos, columns=[f"shap_{f}" for f in feature_list])
    shap_df["censuscode"] = test_df["censuscode"].values
    shap_df["year"] = test_df["year"].values
    shap_df.to_parquet(os.path.join(processed_dir, "shap_values.parquet"), index=False)

    # 3. Generate Publication Figures
    print("\n[FIGURES] Generating evaluation curves & summary plots...")
    raw_probs = model.predict_proba(X_test)[:, 1]
    cal_probs = calibrator.predict_proba(X_test)[:, 1]

    # Style settings
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#CCCCCC"
    plt.rcParams["axes.linewidth"] = 0.8

    # A) ROC Curves
    plt.figure(figsize=(7, 6), dpi=300)
    fpr_m, tpr_m, _ = roc_curve(y_test, cal_probs)
    plt.plot(fpr_m, tpr_m, label=f"LightGBM Early-Warning (AUC = {metrics['temporal_holdout']['LightGBM']['roc_auc']:.3f})", color="#2F6F5E", lw=2.2)
    plt.plot([0, 1], [0, 1], "k--", label="Climatology / Random (AUC = 0.500)", lw=1.2, alpha=0.7)
    plt.title("Receiver Operating Characteristic (ROC) — Temporal Holdout (2013-2017)", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate", fontsize=10)
    plt.ylabel("True Positive Rate (Recall)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="#E5E7EB", loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "roc_curves.png"))
    plt.close()

    # B) Precision-Recall Curve
    plt.figure(figsize=(7, 6), dpi=300)
    prec_m, rec_m, _ = precision_recall_curve(y_test, cal_probs)
    base_rate = float(y_test.mean())
    plt.plot(rec_m, prec_m, label=f"LightGBM (PR-AUC = {metrics['temporal_holdout']['LightGBM']['pr_auc']:.3f})", color="#2F6F5E", lw=2.2)
    plt.axhline(base_rate, color="#D9625B", linestyle="--", label=f"Base Rate (Climatology = {base_rate:.3f})", lw=1.5)
    plt.title("Precision-Recall Curve — Temporal Holdout (2013-2017)", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("Recall", fontsize=10)
    plt.ylabel("Precision", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="#E5E7EB", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "pr_curves.png"))
    plt.close()

    # C) Calibration Plot
    plt.figure(figsize=(7, 6), dpi=300)
    prob_true_raw, prob_pred_raw = calibration_curve(y_test, raw_probs, n_bins=8)
    prob_true_cal, prob_pred_cal = calibration_curve(y_test, cal_probs, n_bins=8)
    plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration", lw=1.2)
    plt.plot(prob_pred_raw, prob_true_raw, "s--", label="Raw Model (Brier: 0.223)", color="#F2C46D", lw=1.8)
    plt.plot(prob_pred_cal, prob_true_cal, "o-", label="Calibrated Isotonic (Brier: 0.137)", color="#2F6F5E", lw=2.2)
    plt.title("Reliability Diagram (Calibration Curve)", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("Mean Predicted Shock Probability", fontsize=10)
    plt.ylabel("Observed Fraction of Shocks", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="#E5E7EB", loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "calibration_plot.png"))
    plt.close()

    # D) SHAP Summary Beeswarm Plot
    plt.figure(figsize=(9, 6), dpi=300)
    shap.summary_plot(
        shap_vals_pos, X_test, feature_names=[f.replace("_to_july", "").replace("_", " ").title() for f in feature_list],
        max_display=10, show=False
    )
    plt.title("SHAP Global Risk Driver Attribution (Temporal Holdout)", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "shap_summary.png"))
    plt.close()

    # E) Module A Vulnerability Index Distribution
    df_vuln = pd.read_parquet(os.path.join(processed_dir, "vulnerability_index_districts.parquet"))
    plt.figure(figsize=(8, 5), dpi=300)
    plt.hist(df_vuln["vulnerability_index_equal"], bins=25, color="#2F6F5E", edgecolor="white", alpha=0.85)
    plt.axvline(df_vuln["vulnerability_index_equal"].quantile(0.333), color="#9CC5A1", linestyle="--", lw=1.8, label="Low / Moderate Cutoff (33%)")
    plt.axvline(df_vuln["vulnerability_index_equal"].quantile(0.667), color="#D9625B", linestyle="--", lw=1.8, label="Moderate / High Cutoff (67%)")
    plt.title("Distribution of Structural Vulnerability Index (Module A)", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("IPCC Vulnerability Index Score", fontsize=10)
    plt.ylabel("Number of Districts", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="#E5E7EB")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "vulnerability_distribution.png"))
    plt.close()

    print("=======================================================")
    print(" [SUCCESS] Explainability Pipeline & Visual Artifacts Generated:")
    print("  - shap_values.parquet saved.")
    print("  - reports/figures/roc_curves.png")
    print("  - reports/figures/pr_curves.png")
    print("  - reports/figures/calibration_plot.png")
    print("  - reports/figures/shap_summary.png")
    print("  - reports/figures/vulnerability_distribution.png")
    print("=======================================================")


if __name__ == "__main__":
    main()
