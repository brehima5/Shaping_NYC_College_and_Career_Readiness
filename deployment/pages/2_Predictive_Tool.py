"""
Page 2 — Predictive Tool
Interactive sliders → live CCR prediction + contribution breakdown.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from utils.data_loader import (
    fit_beta_model, predict_ccr, FEATURE_DISPLAY, BOROUGHS,
)

st.set_page_config(page_title="Predictive Tool", page_icon="🔮", layout="wide")

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-size: 17px;
    }
    h1 { font-size: 2.2rem !important; }
    h2 { font-size: 1.7rem !important; }
    h3 { font-size: 1.35rem !important; }
    h4 { font-size: 1.15rem !important; }
    .stMetricValue { font-size: 1.9rem !important; }
    .stMetricLabel { font-size: 0.95rem !important; }
    .stTabs [data-baseweb="tab"] { font-size: 1.05rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("CCR Predictive Tool")
st.markdown(
    "Adjust the sliders below to describe a hypothetical school, "
    "then see the model's predicted **4-Year CCR** and which features "
    "contribute most to the outcome."
)

art = fit_beta_model()
ranges = art["feature_ranges"]

# ── sidebar sliders ──────────────────────────────────────────────────
st.sidebar.header(" Main CCR Drivers")
st.sidebar.caption(
    "These three factors have the strongest influence on College & Career Readiness."
)
eni = st.sidebar.slider(
    "Economic Need Index",
    min_value=0.0, max_value=1.0,
    value=0.60,
    step=0.01,
    help="0 = lowest need, 1 = highest need. Strongly linked to CCR.",
)
attendance = st.sidebar.slider(
    "Avg Student Attendance",
    min_value=round(ranges["avg_student_attendance"]["min"], 2),
    max_value=1.0,
    value=0.70,
    step=0.01,
    help="Higher attendance strongly predicts higher CCR.",
)
pct_temp = st.sidebar.slider(
    "% Temporary Housing",
    min_value=0.0, max_value=round(ranges["percent_temp_housing"]["max"], 2),
    value=round(ranges["percent_temp_housing"]["median"], 2),
    step=0.01,
    help="Housing instability is a key barrier to CCR outcomes.",
)

st.sidebar.markdown("---")
with st.sidebar.expander("Additional Controls", expanded=False):
    st.caption("Fine-tune other school characteristics for a more precise prediction.")
    teaching = st.slider(
        "Teaching Environment (% Positive)",
        min_value=round(ranges["teaching_environment_pct_positive"]["min"], 2),
        max_value=1.0,
        value=round(ranges["teaching_environment_pct_positive"]["median"], 2),
        step=0.01,
    )
    support = st.slider(
        "Student Support (% Positive)",
        min_value=round(ranges["student_support_pct"]["min"], 2),
        max_value=1.0,
        value=round(ranges["student_support_pct"]["median"], 2),
        step=0.01,
    )
    borough = st.selectbox("Borough", BOROUGHS, index=0)

# ── prediction ───────────────────────────────────────────────────────
pred_ccr, contribs = predict_ccr(art, eni, pct_temp, teaching, attendance, support, borough)

# clamp display to 0-100
pred_display = max(0.0, min(100.0, pred_ccr))

# ── big metric + gauge ───────────────────────────────────────────────
col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("### Predicted CCR")
    st.metric(
        label="4-Year College & Career Readiness",
        value=f"{pred_display:.1f} %",
    )

    overall_mean = art["model_df"]["metric_value_4yr_ccr_all_students"].mean()
    delta = pred_display - overall_mean
    if delta > 0:
        st.success(f"▲ {delta:+.1f} pts above the citywide average ({overall_mean:.1f} %)")
    else:
        st.error(f"▼ {delta:+.1f} pts below the citywide average ({overall_mean:.1f} %)")

with col_right:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred_display,
        number={"suffix": "%", "font": {"size": 48}},
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=2),
            bar=dict(color="#4682B4"),
            steps=[
                dict(range=[0, 40],  color="#FFCDD2"),
                dict(range=[40, 60], color="#FFF9C4"),
                dict(range=[60, 80], color="#C8E6C9"),
                dict(range=[80, 100], color="#A5D6A7"),
            ],
            threshold=dict(
                line=dict(color="red", width=3),
                thickness=0.8,
                value=overall_mean,
            ),
        ),
        title={"text": "Predicted CCR (red line = city avg)"},
    ))
    fig_gauge.update_layout(height=320, margin=dict(t=60, b=20, l=30, r=30))
    st.plotly_chart(fig_gauge, use_container_width=True)

# ── feature contribution breakdown ───────────────────────────────────
st.markdown("---")
st.markdown("### What's Driving This Prediction?")
st.caption(
    "Each bar shows the feature's contribution to the log-odds (logit) "
    "score. Positive pushes CCR up; negative pushes it down."
)

# sort contributions by absolute value (exclude intercept for the chart)
contrib_items = {
    k: v for k, v in contribs.items() if k != "const"
}
sorted_items = sorted(contrib_items.items(), key=lambda x: x[1])

names  = [FEATURE_DISPLAY.get(k, k) for k, _ in sorted_items]
values = [v for _, v in sorted_items]
colors = ["#4CAF50" if v > 0 else "#EF5350" for v in values]

fig_cb = go.Figure(go.Bar(
    y=names,
    x=values,
    orientation="h",
    marker_color=colors,
    text=[f"{v:+.3f}" for v in values],
    textposition="outside",
))
fig_cb.add_vline(x=0, line_dash="dash", line_color="black")
fig_cb.update_layout(
    title="Feature Contributions (logit scale)",
    xaxis_title="Contribution to log-odds",
    height=420,
    margin=dict(l=20, r=20, t=50, b=30),
    plot_bgcolor="white",
)
st.plotly_chart(fig_cb, use_container_width=True)

# ── intercept context ────────────────────────────────────────────────
intercept_ccr = 1 / (1 + np.exp(-contribs["const"])) * 100
st.info(
    f"**Baseline (intercept):** When all features are at their training-set "
    f"average, the model predicts **{intercept_ccr:.1f} % CCR**. "
    f"The feature contributions above shift the prediction from this baseline "
    f"to the final **{pred_display:.1f} %**."
)

# ── interpretation tips ──────────────────────────────────────────────
with st.expander("💡 How to read this"):
    st.markdown(
        """
        | Component | Meaning |
        |-----------|---------|
        | **Intercept** | Predicted CCR when every feature equals its average |
        | **Green bars** | Features pushing CCR **higher** than the baseline |
        | **Red bars** | Features pulling CCR **lower** than the baseline |
        | **Bar length** | Magnitude of impact on the logit score |
        | **Gauge threshold** | Red line marks the citywide average CCR |

        Because the model uses a **logit link**, contributions are additive
        on the log-odds scale but non-linear on the probability scale.
        The same 0.1 shift in logit has a larger impact near 50 % CCR
        than near 5 % or 95 %.
        """
    )
