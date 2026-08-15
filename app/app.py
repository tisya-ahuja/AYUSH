import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go


# Page configuration
st.set_page_config(
    page_title="AYUSH — RUL Prediction",
    page_icon="⚙️",
    layout="wide"
)


# Custom styling
st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@600;700;800&family=Roboto:wght@400;500;700&display=swap');


    /* Main application background */
    .stApp {
        background-color: rgba(148, 180, 68, 0.2);
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Racing Sans One', sans-serif;
    }

    /* Body text */
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }

    /* Reduce excessive divider contrast */
    hr {
        border: none;
        border-top: 1px solid rgba(0, 0, 0, 0.08);
    }

    /* Metric labels */
    [data-testid="stMetricLabel"] {
        font-family: 'Roboto', sans-serif;
        font-size: 13px;
    }

    /* Metric values */
    [data-testid="stMetricValue"] {
        font-family: 'Roboto', sans-serif;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent

PREDICTIONS_PATH = (
    BASE_DIR
    / "reports"
    / "FD001"
    / "fd001_test_predictions.csv"
)


# Load official FD001 test predictions
test_predictions = pd.read_csv(PREDICTIONS_PATH)


# Maintenance risk classification
def classify_risk(predicted_rul):

    if predicted_rul > 100:
        return "NORMAL"

    elif predicted_rul > 50:
        return "MONITOR"

    else:
        return "HIGH MAINTENANCE PRIORITY"


test_predictions["maintenance_risk"] = (
    test_predictions["predicted_rul"]
    .apply(classify_risk)
)


# Header
st.title("AYUSH")

st.subheader(
    "ML-Based Remaining Useful Life Prediction"
)

st.write(
    "Predictive-maintenance prototype for estimating "
    "Remaining Useful Life (RUL) from equipment data."
)


# Engine selection
st.header("Engine Selection")

engine_ids = (
    test_predictions["unit"]
    .astype(int)
    .tolist()
)

selected_engine = st.selectbox(
    "Select test engine",
    engine_ids
)


engine_row = test_predictions[
    test_predictions["unit"] == selected_engine
].iloc[0]


# Selected engine prediction
predicted_rul = float(
    engine_row["predicted_rul"]
)

risk = engine_row["maintenance_risk"]


# Prediction
st.divider()

st.header("Prediction")

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Current Cycle",
        int(engine_row["cycle"])
    )


with col2:

    st.metric(
        "Predicted RUL",
        f"{predicted_rul:.1f} cycles"
    )


with col3:

    st.metric(
        "Maintenance Risk",
        risk
    )


# Prediction evaluation
st.header("Prediction Evaluation")

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Actual RUL",
        f"{engine_row['actual_rul']:.1f} cycles"
    )


with col2:

    st.metric(
        "Absolute Error",
        f"{engine_row['absolute_error']:.1f}"
    )


with col3:

    st.metric(
        "Prediction Error",
        f"{engine_row['error']:.1f}"
    )

st.divider()
# Fleet maintenance risk overview
st.header("Fleet Maintenance Risk Overview")

risk_counts = (
    test_predictions["maintenance_risk"]
    .value_counts()
    .reindex(
        [
            "NORMAL",
            "MONITOR",
            "HIGH MAINTENANCE PRIORITY"
        ],
        fill_value=0
    )
)


fig = go.Figure(
    go.Bar(
        x=risk_counts.index,
        y=risk_counts.values,
        marker_color="rgba(148, 180, 68, 0.65)",
        marker_line_width=0,
        text=risk_counts.values,
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b>"
            "<br>Engines: %{y}"
            "<extra></extra>"
        )
    )
)


fig.update_layout(
    height=280,

    margin=dict(
        l=10,
        r=10,
        t=20,
        b=20
    ),

    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",

    xaxis=dict(
        title=None,
        showgrid=False,
        showline=False,
        zeroline=False,
        tickfont=dict(
            family="Roboto",
            size=11
        )
    ),

    yaxis=dict(
        title=None,
        showgrid=False,
        showline=False,
        zeroline=False,
        showticklabels=False
    ),

    font=dict(
        family="Roboto",
        size=11
    ),

    showlegend=False
)


st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": False
    }
)


# Prototype disclaimer
st.divider()
st.caption(
    "AYUSH is an academic/research prototype using the "
    "NASA C-MAPSS FD001 simulated turbofan dataset. "
    "Predictions are intended for demonstration and research "
    "purposes and should not be interpreted as operational "
    "defence maintenance recommendations. Maintenance-risk "
    "thresholds are experimental prototype decision rules."
)



with st.bottom:
    st.write("© 2026 Tisya Ahuja · All rights reserved")
