"""main.py - Krishi Risk Web Application

District-Level Rural Livelihood Vulnerability & Agricultural Shock Early-Warning System (India).
A paired-module decision support dashboard:
- Module A: Structural Vulnerability Index (IPCC Exposure, Sensitivity, Adaptive Capacity)
- Module B: Early-Warning Agricultural Shock Prediction (Calibrated Probability & SHAP Attribution)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & CUSTOM CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Krishi Risk | Decision Support System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        background-color: #FAFAF7;
    }
    
    .stCard {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-bottom: 16px;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #1F2933;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 13px;
        font-weight: 500;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-low {
        background-color: #E8F5E9;
        color: #1B4332;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
    }
    
    .badge-mod {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
    }
    
    .badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
    }

    .info-callout {
        background-color: #F0FDF4;
        border-left: 4px solid #2F6F5E;
        padding: 12px 16px;
        border-radius: 4px;
        color: #166534;
        font-size: 14px;
        margin-bottom: 16px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. DATA LOADERS & CACHING
# -----------------------------------------------------------------------------
@st.cache_data
def load_app_data():
    preds = pd.read_parquet("data/processed/predictions.parquet")
    vuln = pd.read_parquet("data/processed/vulnerability_index_districts.parquet")
    panel = pd.read_parquet("data/processed/master_panel.parquet")
    with open("reports/metrics.json", "r") as f:
        metrics = json.load(f)
    with open("data/processed/districts.geojson", "r") as f:
        geojson = json.load(f)
    with open("models/feature_list.json", "r") as f:
        feature_list = json.load(f)
    return preds, vuln, panel, metrics, geojson, feature_list


@st.cache_resource
def load_models():
    model = joblib.load("models/shock_model.joblib")
    calibrator = joblib.load("models/calibrator.joblib")
    imputer = joblib.load("models/imputer.joblib")
    return model, calibrator, imputer


df_preds, df_vuln, df_panel, metrics_data, geojson_data, feature_names = load_app_data()
model, calibrator, imputer = load_models()


# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌾 Krishi Risk")
    st.markdown(
        "<span style='color: #6B7280; font-size: 13px;'>Early-Warning & Vulnerability System</span>",
        unsafe_allow_html=True
    )
    st.divider()

    selected_page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🗺️ Risk Map",
            "🔍 District Explorer",
            "🧪 What-If Simulator",
            "📊 Model Performance",
            "📋 Priority Action List",
            "📚 Data & Methodology"
        ],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("""
    **Research Project**  
    *Shivam Mewada (TE Data Science)*  
    *A.P. Shah Institute of Technology*  
    *Guide: Prof. Veena Trivedi*
    """)


# -----------------------------------------------------------------------------
# 4. PAGE: HOME
# -----------------------------------------------------------------------------
if selected_page == "🏠 Home":
    st.title("District Rural Livelihood Vulnerability & Crop-Shock Early-Warning")
    st.markdown("""
    **Krishi Risk** is an empirical, decision-support platform that decouples long-term structural vulnerability from short-term agricultural climate shocks. Designed for state disaster management authorities, district planners, and rural livelihood NGOs to prioritize early interventions **before** harvest failure occurs.
    """)

    # Headline Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="stCard">
            <div class="metric-label">Districts Covered</div>
            <div class="metric-value">296</div>
            <div style="font-size: 12px; color: #6B7280; margin-top: 4px;">Across 19 Agricultural States</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="stCard">
            <div class="metric-label">Historical Panel</div>
            <div class="metric-value">1990–2017</div>
            <div style="font-size: 12px; color: #6B7280; margin-top: 4px;">27-Year Clean District Time-Series</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="stCard">
            <div class="metric-label">Spatial ROC-AUC</div>
            <div class="metric-value">0.643</div>
            <div style="font-size: 12px; color: #166534; margin-top: 4px;">GroupKFold on Unseen States</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="stCard">
            <div class="metric-label">MPI Validation</div>
            <div class="metric-value">r = 0.612</div>
            <div style="font-size: 12px; color: #166534; margin-top: 4px;">Spearman vs NITI Aayog (p &lt; 10⁻³⁰)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-callout">
        <b>💡 Why this dual-module design matters:</b><br>
        A single index that labels vulnerability and then predicts that same label from identical indicators creates a <i>circular feedback loop</i>. 
        <b>Krishi Risk</b> separates the problem: 
        <b>Module A</b> measures slow-moving structural sensitivity (validated strictly against external NITI Aayog poverty data), while 
        <b>Module B</b> targets an objective, future, observable outcome—a serious yield shock (&gt;1 SD below district trend) predicted as of July 31 with zero data leakage.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("System Architecture")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        ### Module A: Structural Vulnerability (Descriptive)
        - **Framework:** IPCC 2007/2014 ($V = \text{Exposure} + \text{Sensitivity} - \text{Adaptive Capacity}$)
        - **Exposure:** Rainfall CV, drought frequency, dry spells, heat stress, extreme rain.
        - **Sensitivity:** Rainfed area share, agricultural workforce %, smallholder holding share, crop concentration.
        - **Adaptive Capacity:** Irrigation coverage, female literacy, overall literacy, fertilizer intensity.
        - **Validation:** Independent rank correlation ($r_s = 0.612$) with NITI Aayog Multidimensional Poverty Index.
        """)
    with col_b:
        st.markdown("""
        ### Module B: Crop-Shock Early Warning (Predictive)
        - **Target:** Harvest yield shock ($< -1.0$ SD relative to district linear technology trend).
        - **Forecast Time:** End of July (3 months before harvest).
        - **Observable Inputs:** Cumulative rainfall to July, dry days, dry-spell length, $T_{\\max}$ anomalies, lagged prior shocks, static structural buffers.
        - **Models:** LightGBM, XGBoost, Random Forest compared against Climatology and Persistence baselines.
        - **Evaluation:** Out-of-time temporal holdout (2013–2017) and Spatial Leave-State-Out CV.
        """)


# -----------------------------------------------------------------------------
# 5. PAGE: RISK MAP
# -----------------------------------------------------------------------------
elif selected_page == "🗺️ Risk Map":
    st.title("Interactive Risk & Early-Warning Choropleth")
    st.markdown("Explore structural vulnerability and early-warning yield shock probabilities across Indian districts.")

    # Control Bar
    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
    with ctrl1:
        map_layer = st.radio(
            "Select Map Layer",
            ["Structural Vulnerability (Module A)", "Shock Probability (Module B)", "Action Priority Score (Combined)"],
            horizontal=True
        )
    with ctrl2:
        states = ["All States"] + sorted(df_preds["ST_NM"].unique().tolist())
        selected_state = st.selectbox("Filter State", states)
    with ctrl3:
        years = sorted(df_preds["year"].unique().tolist(), reverse=True)
        selected_year = st.selectbox("Select Year", years)

    # Filter data
    data_slice = df_preds[df_preds["year"] == selected_year].copy()
    if selected_state != "All States":
        data_slice = data_slice[data_slice["ST_NM"] == selected_state]

    # Configure Choropleth based on selected layer
    if map_layer == "Structural Vulnerability (Module A)":
        fig = px.choropleth(
            data_slice,
            geojson=geojson_data,
            locations="censuscode",
            featureidkey="properties.censuscode",
            color="vulnerability_index_equal",
            color_continuous_scale="YlOrRd",
            range_color=[0.2, 0.8],
            hover_name="DISTRICT",
            hover_data={
                "ST_NM": True,
                "vulnerability_class": True,
                "vulnerability_index_equal": ":.3f",
                "shock_probability": ":.1%",
                "priority_score": True,
                "censuscode": False
            },
            labels={"vulnerability_index_equal": "Vulnerability Score"}
        )
    elif map_layer == "Shock Probability (Module B)":
        fig = px.choropleth(
            data_slice,
            geojson=geojson_data,
            locations="censuscode",
            featureidkey="properties.censuscode",
            color="shock_probability",
            color_continuous_scale="Reds",
            range_color=[0.05, 0.50],
            hover_name="DISTRICT",
            hover_data={
                "ST_NM": True,
                "shock_probability": ":.1%",
                "vulnerability_class": True,
                "top_driver_1": True,
                "censuscode": False
            },
            labels={"shock_probability": "Shock Probability"}
        )
    else:
        fig = px.choropleth(
            data_slice,
            geojson=geojson_data,
            locations="censuscode",
            featureidkey="properties.censuscode",
            color="priority_score",
            color_continuous_scale=["#9CC5A1", "#F2C46D", "#E07A5F", "#D9625B", "#8B0000"],
            range_color=[1, 5],
            hover_name="DISTRICT",
            hover_data={
                "ST_NM": True,
                "priority_score": True,
                "vulnerability_class": True,
                "shock_probability": ":.1%",
                "censuscode": False
            },
            labels={"priority_score": "Priority (1-5)"}
        )

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        margin={"r": 0, "t": 20, "l": 0, "b": 0},
        height=620,
        coloraxis_colorbar=dict(
            title="",
            thickness=16,
            len=0.7,
            bgcolor="rgba(255,255,255,0.85)"
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("How to read: Darker hues indicate higher vulnerability, elevated harvest failure probability, or urgent intervention priority.")


# -----------------------------------------------------------------------------
# 6. PAGE: DISTRICT EXPLORER
# -----------------------------------------------------------------------------
elif selected_page == "🔍 District Explorer":
    st.title("District Diagnostic Explorer")
    st.markdown("Detailed risk profile, SHAP early-warning drivers, and multi-year climate/yield trends.")

    col_s, col_d = st.columns(2)
    with col_s:
        dist_states = sorted(df_preds["ST_NM"].unique().tolist())
        chosen_state = st.selectbox("Select State", dist_states)
    with col_d:
        avail_dists = sorted(df_preds[df_preds["ST_NM"] == chosen_state]["DISTRICT"].unique().tolist())
        chosen_dist = st.selectbox("Select District", avail_dists)

    dist_preds = df_preds[(df_preds["ST_NM"] == chosen_state) & (df_preds["DISTRICT"] == chosen_dist)].sort_values("year", ascending=False)
    latest = dist_preds.iloc[0]

    # Profile Header Cards
    st.markdown(f"### {chosen_dist} ({chosen_state}) — Status as of {latest['year']}")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        badge_class = "badge-high" if latest["vulnerability_class"] == "High" else ("badge-mod" if latest["vulnerability_class"] == "Moderate" else "badge-low")
        st.markdown(f"""
        <div class="stCard">
            <div class="metric-label">Structural Vulnerability</div>
            <div class="metric-value"><span class="{badge_class}">{latest['vulnerability_class']}</span></div>
            <div style="font-size: 12px; color: #6B7280; margin-top: 6px;">Index Score: {latest['vulnerability_index_equal']:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="stCard">
            <div class="metric-label">Yield Shock Probability</div>
            <div class="metric-value">{latest['shock_probability'] * 100:.1f}%</div>
            <div style="font-size: 12px; color: #6B7280; margin-top: 6px;">Predicted Harvest Deficit Risk</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        p_score = int(latest["priority_score"])
        p_label = {5: "Critical (Act First)", 4: "Urgent", 3: "Elevated", 2: "Watch", 1: "Low Risk"}.get(p_score, "Standard")
        st.markdown(f"""
        <div class="stCard">
            <div class="metric-label">Action Priority</div>
            <div class="metric-value">Level {p_score}</div>
            <div style="font-size: 12px; color: #D9625B; font-weight: 600; margin-top: 6px;">{p_label}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="stCard">
            <div class="metric-label">Irrigation Buffer</div>
            <div class="metric-value">{latest['irrigation_coverage_ratio'] * 100:.1f}%</div>
            <div style="font-size: 12px; color: #6B7280; margin-top: 6px;">Rainfed Share: {latest['rainfed_area_share'] * 100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Top Drivers & Suggested Interventions
    st.subheader("Top Predictive Drivers (SHAP Attribution) & Suggested Actions")
    st.markdown("""
    <div style="font-size: 13px; color: #6B7280; margin-bottom: 12px;">
        *Note: Attributions reflect statistical early-warning risk associations observable at forecast time (July 31). Actions are decision-support suggestions.*
    </div>
    """, unsafe_allow_html=True)

    for i in range(1, 4):
        d_name = latest.get(f"top_driver_{i}", f"Driver {i}")
        s_val = latest.get(f"shap_{i}", 0.0)
        action = latest.get(f"action_{i}", "Monitor seasonal crop progress.")
        clean_driver = str(d_name).replace("_to_july", "").replace("_", " ").title()
        
        st.markdown(f"""
        <div style="background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 600; color: #1F2933; font-size: 15px;">#{i} {clean_driver}</span>
                <span style="font-size: 12px; font-weight: 600; color: #2F6F5E; background: #E8F5E9; padding: 2px 8px; border-radius: 12px;">SHAP Contribution: +{s_val:.3f}</span>
            </div>
            <div style="font-size: 13px; color: #4B5563; margin-top: 6px;">
                <b>Suggested Policy Action:</b> {action}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 10-Year Historical Trends
    st.subheader("10-Year Climate and Agricultural Trajectory")
    dist_hist = df_panel[(df_panel["ST_NM"] == chosen_state) & (df_panel["DISTRICT"] == chosen_dist)].sort_values("year")
    recent_10 = dist_hist.tail(12)

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=recent_10["year"], y=recent_10["kharif_composite_yield_kgha"],
        mode="lines+markers", name="Kharif Yield (kg/ha)",
        line=dict(color="#2F6F5E", width=2.5), marker=dict(size=6)
    ))
    fig_trend.add_trace(go.Bar(
        x=recent_10["year"], y=recent_10["monsoon_rainfall_mm"],
        name="Monsoon Rainfall (mm)", yaxis="y2",
        marker_color="rgba(242, 196, 109, 0.45)"
    ))

    fig_trend.update_layout(
        yaxis=dict(title="Yield (kg/ha)", color="#2F6F5E"),
        yaxis2=dict(title="Monsoon Rain (mm)", overlaying="y", side="right", color="#D97706"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=30, b=0),
        height=380,
        plot_bgcolor="white"
    )
    st.plotly_chart(fig_trend, use_container_width=True)


# -----------------------------------------------------------------------------
# 7. PAGE: WHAT-IF SIMULATOR
# -----------------------------------------------------------------------------
elif selected_page == "🧪 What-If Simulator":
    st.title("What-If Scenario Simulator")
    st.markdown("""
    Test hypothetical climate or infrastructural shifts to observe how the early-warning model's predicted shock probability changes.  
    **Crucial Note:** *This simulator models statistical association based on historical empirical patterns—it does not claim direct mechanical causality.*
    """)

    c_s, c_d = st.columns(2)
    with c_s:
        sim_state = st.selectbox("Baseline State", sorted(df_preds["ST_NM"].unique().tolist()))
    with c_d:
        sim_dist = st.selectbox("Baseline District", sorted(df_preds[df_preds["ST_NM"] == sim_state]["DISTRICT"].unique().tolist()))

    baseline_row = df_preds[(df_preds["ST_NM"] == sim_state) & (df_preds["DISTRICT"] == sim_dist)].iloc[0]
    base_panel_row = df_panel[(df_panel["ST_NM"] == sim_state) & (df_panel["DISTRICT"] == sim_dist)].sort_values("year").iloc[-1]

    st.subheader("Scenario Parameters")
    s1, s2 = st.columns(2)
    with s1:
        sim_deficit = st.slider(
            "Rainfall to July Anomaly (% Deviation from Normal)",
            min_value=-70.0, max_value=50.0,
            value=float(base_panel_row.get("rainfall_to_july_deviation_pct", 0.0)),
            step=5.0,
            help="Negative values represent monsoon rainfall failure by July 31."
        )
        sim_dry_spell = st.slider(
            "Longest Dry Spell to July (Consecutive Days < 1mm)",
            min_value=0, max_value=45,
            value=int(base_panel_row.get("longest_dry_spell_to_july", 10)),
            step=1
        )
    with s2:
        sim_irrigation = st.slider(
            "District Irrigation Coverage Ratio (%)",
            min_value=0.0, max_value=100.0,
            value=float(base_panel_row.get("irrigation_coverage_ratio", 0.35) * 100.0),
            step=2.0
        )
        sim_tmax_dev = st.slider(
            "July Maximum Temperature Anomaly (°C above normal)",
            min_value=-3.0, max_value=5.0,
            value=float(base_panel_row.get("tmax_to_july_anomaly", 0.0)),
            step=0.5
        )

    # Build input feature vector
    row_dict = base_panel_row.to_dict()
    row_dict["rainfall_to_july_deviation_pct"] = sim_deficit
    row_dict["longest_dry_spell_to_july"] = sim_dry_spell
    row_dict["irrigation_coverage_ratio"] = sim_irrigation / 100.0
    row_dict["rainfed_area_share"] = 1.0 - (sim_irrigation / 100.0)
    row_dict["tmax_to_july_anomaly"] = sim_tmax_dev

    input_df = pd.DataFrame([row_dict])[feature_names]
    imputed_input = imputer.transform(input_df)
    sim_prob = calibrator.predict_proba(imputed_input)[0, 1]
    base_prob = baseline_row["shock_probability"]

    st.divider()
    res1, res2 = st.columns([1, 2])
    with res1:
        st.markdown("### Simulated Risk Output")
        diff = sim_prob - base_prob
        diff_str = f"+{diff*100:.1f}%" if diff >= 0 else f"{diff*100:.1f}%"
        color = "#D9625B" if sim_prob >= 0.30 else ("#F2C46D" if sim_prob >= 0.20 else "#2F6F5E")
        
        st.markdown(f"""
        <div class="stCard">
            <div class="metric-label">Simulated Shock Probability</div>
            <div class="metric-value" style="color: {color};">{sim_prob * 100:.1f}%</div>
            <div style="font-size: 13px; margin-top: 6px;">
                Baseline: {base_prob * 100:.1f}% (<b>{diff_str}</b> change)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with res2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=sim_prob * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Shock Probability Gauge", "font": {"size": 16}},
            delta={"reference": base_prob * 100, "increasing": {"color": "#D9625B"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#2F6F5E"},
                "steps": [
                    {"range": [0, 20], "color": "#E8F5E9"},
                    {"range": [20, 35], "color": "#FEF3C7"},
                    {"range": [35, 100], "color": "#FEE2E2"}
                ],
                "threshold": {
                    "line": {"color": "#D9625B", "width": 4},
                    "thickness": 0.75,
                    "value": 35
                }
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)


# -----------------------------------------------------------------------------
# 8. PAGE: MODEL PERFORMANCE
# -----------------------------------------------------------------------------
elif selected_page == "📊 Model Performance":
    st.title("Model Performance & Rigorous Validation")
    st.markdown("""
    Every metric below is measured honestly against established baselines. We follow a strict temporal hold-out (train early, test on last 5 years) and Spatial Leave-State-Out Cross Validation.
    """)

    st.subheader("1. Temporal Holdout (Trained: 1991–2012, Tested: 2013–2017)")
    temp_res = metrics_data["temporal_holdout"]
    temp_df = pd.DataFrame([
        {"Model": k, "ROC-AUC": v["roc_auc"], "PR-AUC": v["pr_auc"], "Recall @ Top-20%": f"{v['recall_at_top20']*100:.1f}%", "Brier Score": v["brier_score"]}
        for k, v in temp_res.items()
    ])
    st.dataframe(temp_df, hide_index=True, use_container_width=True)

    st.subheader("2. Spatial Cross-Validation (GroupKFold by Unseen State)")
    spat_res = metrics_data["spatial_cv"]
    spat_df = pd.DataFrame([
        {
            "Model": k,
            "Mean ROC-AUC": f"{v['mean_roc_auc']:.4f} (±{v['std_roc_auc']:.3f})",
            "Mean PR-AUC": f"{v['mean_pr_auc']:.4f} (±{v['std_pr_auc']:.3f})",
            "Recall @ Top-20%": f"{v['mean_recall_at_top20']*100:.1f}%",
            "Mean Brier Score": f"{v['mean_brier_score']:.4f}"
        }
        for k, v in spat_res.items()
    ])
    st.dataframe(spat_df, hide_index=True, use_container_width=True)

    st.subheader("Evaluation Visualizations")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["ROC Curve", "Precision-Recall Curve", "Reliability / Calibration", "SHAP Beeswarm", "Module A Distribution"])
    
    with tab1:
        if os.path.exists("reports/figures/roc_curves.png"):
            st.image("reports/figures/roc_curves.png", use_column_width=True)
            st.caption("How to read: Shows trade-off between sensitivity and false positives. LightGBM (0.591) significantly outperforms random climatology (0.500).")
    with tab2:
        if os.path.exists("reports/figures/pr_curves.png"):
            st.image("reports/figures/pr_curves.png", use_column_width=True)
            st.caption("How to read: PR-AUC is the vital metric for rare shock classes (~16% base rate). LightGBM achieves 0.242, an over 45% relative improvement over guessing.")
    with tab3:
        if os.path.exists("reports/figures/calibration_plot.png"):
            st.image("reports/figures/calibration_plot.png", use_column_width=True)
            st.caption("How to read: Reliability diagram. Isotonic calibration reduced Brier loss from 0.223 to 0.137, making output probabilities trustworthy.")
    with tab4:
        if os.path.exists("reports/figures/shap_summary.png"):
            st.image("reports/figures/shap_summary.png", use_column_width=True)
            st.caption("How to read: Beeswarm plot showing direction of feature impact. High early rainfall deficit and prolonged dry spells systematically shift risk to the right.")
    with tab5:
        if os.path.exists("reports/figures/vulnerability_distribution.png"):
            st.image("reports/figures/vulnerability_distribution.png", use_column_width=True)
            st.caption("How to read: Distribution of IPCC Structural Vulnerability across 296 districts, categorized into balanced tertiles (Low, Moderate, High).")


# -----------------------------------------------------------------------------
# 9. PAGE: PRIORITY ACTION LIST
# -----------------------------------------------------------------------------
elif selected_page == "📋 Priority Action List":
    st.title("Intervention Priority Ranking")
    st.markdown("Consolidated ranking combining Structural Vulnerability and Shock Probability to guide where district officials should act first.")

    col1, col2 = st.columns([2, 1])
    with col1:
        p_state = st.selectbox("State Filter", ["All States"] + sorted(df_preds["ST_NM"].unique().tolist()), key="p_state")
    with col2:
        top_n = st.slider("Display Top-N Priority Districts", min_value=5, max_value=50, value=15)

    recent_preds = df_preds[df_preds["year"] == df_preds["year"].max()].copy()
    if p_state != "All States":
        recent_preds = recent_preds[recent_preds["ST_NM"] == p_state]

    ranked = recent_preds.sort_values(["priority_score", "shock_probability"], ascending=[False, False]).head(top_n)

    display_table = ranked[[
        "DISTRICT", "ST_NM", "priority_score", "vulnerability_class",
        "shock_probability", "top_driver_1", "action_1"
    ]].rename(columns={
        "DISTRICT": "District",
        "ST_NM": "State",
        "priority_score": "Priority Level (1-5)",
        "vulnerability_class": "Structural Class",
        "shock_probability": "Shock Probability",
        "top_driver_1": "Primary Risk Driver",
        "action_1": "Suggested Intervention"
    })
    display_table["Shock Probability"] = display_table["Shock Probability"].map(lambda x: f"{x*100:.1f}%")

    st.dataframe(display_table, hide_index=True, use_container_width=True)

    csv_data = display_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Priority List as CSV",
        data=csv_data,
        file_name=f"krishi_risk_priority_{p_state.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )


# -----------------------------------------------------------------------------
# 10. PAGE: DATA & METHODOLOGY
# -----------------------------------------------------------------------------
elif selected_page == "📚 Data & Methodology":
    st.title("Data Sources, Methodology & Model Card")
    st.markdown("Full transparency on data origin, mathematical equations, and operational limitations.")

    tab_sources, tab_method, tab_card = st.tabs(["Data Sources", "Methodology & Formulas", "Official Model Card"])

    with tab_sources:
        st.markdown("""
        ### Tier 1 Core Datasets
        1. **IMD Pune Gridded Meteorological Data (1990–2020)**
           - Daily rainfall: 0.25° grid (~25 km resolution), native binary `.grd`.
           - Daily temperature ($T_{\\max}, T_{\\min}$): 1.0° grid (~100 km), bilinearly regridded to 0.25°.
           - Source: National Climate Centre, IMD Pune.
        2. **ICRISAT/TCI Apportioned District Level Database (1966–2017)**
           - 571 districts mapped to 1966 parent district baselines to ensure longitudinal continuity.
           - Source: ICRISAT Open Data Portal (`data.icrisat.org/dld`).
        3. **Census 2011 Primary Census Abstract (PCA)**
           - District-level workforce, cultivator, agricultural labourer, and literacy parameters.
           - Source: Office of the Registrar General of India / DataMeet.
        4. **NITI Aayog National Multidimensional Poverty Index (MPI)**
           - Baseline (NFHS-4) and Progress Review (NFHS-5) district headcount ratios.
           - Role: Strict independent external validation of Module A ($r_s = 0.612$).
        """)

    with tab_method:
        st.markdown("""
        ### Mathematical Framework
        
        #### 1. Module A: Structural Vulnerability Index
        Based on the IPCC formula:
        $$V_i = \\frac{\\text{Exposure}_i + \\text{Sensitivity}_i + (1 - \\text{Adaptive Capacity}_i)}{3}$$
        Where each dimension is a normalized composite of min-max scaled indicators oriented such that higher values denote greater vulnerability. Validated against external NITI Aayog MPI using Spearman rank correlation ($r_s = 0.612, p = 2.15 \\times 10^{-31}$).
        
        #### 2. Module B: Detrended Agricultural Shock Definition
        To prevent technological growth trends from confounding climate shock detection, district yield series are detrended:
        $$\\hat{Y}_{it} = \\alpha_i + \\beta_i \\cdot t$$
        $$e_{it} = Y_{it} - \\hat{Y}_{it}$$
        $$z_{it} = \\frac{e_{it}}{\\sigma_{i, \\text{residual}}}$$
        A **Serious Yield Shock** is defined as $z_{it} < -1.0$ (yield residual falls more than 1 standard deviation below trend).
        """)

    with tab_card:
        if os.path.exists("app/model_card.md"):
            with open("app/model_card.md", "r") as f:
                st.markdown(f.read())
        else:
            st.info("Model card file not found.")

