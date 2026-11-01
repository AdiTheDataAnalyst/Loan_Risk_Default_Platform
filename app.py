"""LoanLens — an interactive loan-default risk assessment dashboard."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


st.set_page_config(page_title="LoanLens | Risk Intelligence", page_icon="◈", layout="wide")


st.markdown(
    """
    <style>
      .stApp { background: #f6f8fc; color: #172033; }
      [data-testid="stSidebar"] { background: #101b35; }
      [data-testid="stSidebar"] * { color: #f7f9ff !important; }
      .hero { padding: 1.65rem 1.9rem; border-radius: 22px; color: white;
              background: radial-gradient(circle at 88% 10%, #4f9cff 0, transparent 35%),
                          linear-gradient(120deg, #101c38 0%, #253f72 100%); }
      .hero h1 { margin: 0; font-size: 2.25rem; letter-spacing: -0.06rem; }
      .hero p { margin: .45rem 0 0; opacity: .84; font-size: 1rem; }
      .eyebrow { color: #81b2ff; text-transform: uppercase; letter-spacing: .12rem; font-weight: 700; font-size: .75rem; }
      .metric-card { background: white; padding: 1.1rem 1.2rem; border-radius: 16px;
                     border: 1px solid #e7ebf3; box-shadow: 0 8px 20px rgba(30,54,95,.05); }
      .metric-card h4 { margin: 0 0 .4rem; color: #667085; font-size: .76rem; text-transform: uppercase; letter-spacing: .05rem; }
      .metric-card p { margin: 0; color: #14213d; font-size: 1.5rem; font-weight: 750; }
      div.stButton > button { background: linear-gradient(90deg, #2667ff, #5b8cff); color: white;
          border: 0; border-radius: 10px; font-weight: 700; height: 3rem; width: 100%; }
      .disclaimer { color: #667085; font-size: .8rem; padding-top: .6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_portfolio():
    path = Path("Loan_Dataset.csv")
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def risk_score(v: dict) -> float:
    """Transparent demo score; replace with your fitted model's predict_proba call."""
    logit = -3.2
    logit += max(0, (620 - v["credit_score"]) / 70) * 0.8
    logit += max(0, (v["dti"] - 0.32) / 0.12) * 0.9
    logit += max(0, (v["interest_rate"] - 10) / 8) * 0.35
    logit += max(0, (v["loan_amount"] / max(v["income"], 1) - 0.8)) * 0.55
    logit += 0.34 if v["employment"] in ["Unemployed", "Self-employed"] else 0
    logit += 0.22 if v["has_cosigner"] == "No" else 0
    return float(1 / (1 + np.exp(-logit)))


@st.cache_resource(show_spinner="Training comparison models from the portfolio…")
def train_comparison_models(portfolio: pd.DataFrame) -> dict:
    """Train compact, reproducible comparison models for the dashboard."""
    model_data = portfolio.drop(columns=["LoanID"], errors="ignore").dropna(subset=["Default"])
    if len(model_data) > 30_000:
        model_data = model_data.sample(30_000, random_state=42)
    features = model_data.drop(columns="Default")
    target = model_data["Default"].astype(int)
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=600, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=25, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=120, min_samples_leaf=12, class_weight="balanced", n_jobs=-1, random_state=42),
    }
    return {name: Pipeline([("preprocessor", preprocessor), ("model", estimator)]).fit(features, target)
            for name, estimator in estimators.items()}


def applicant_features(portfolio: pd.DataFrame, values: dict) -> pd.DataFrame:
    """Create a complete model row, using typical portfolio values for omitted fields."""
    features = portfolio.drop(columns=["Default", "LoanID"], errors="ignore").copy()
    row = {}
    for column in features.columns:
        row[column] = features[column].median() if pd.api.types.is_numeric_dtype(features[column]) else features[column].mode().iat[0]
    row.update({
        "Age": values["age"], "Income": values["income"], "LoanAmount": values["loan_amount"],
        "CreditScore": values["credit_score"], "InterestRate": values["interest_rate"],
        "LoanTerm": values["term"], "DTIRatio": values["dti"], "EmploymentType": values["employment"],
        "HasCoSigner": values["has_cosigner"],
    })
    return pd.DataFrame([row], columns=features.columns)


def render_model_comparison(portfolio: pd.DataFrame, values: dict) -> None:
    if portfolio.empty or "Default" not in portfolio:
        st.error("The portfolio dataset with a Default column is required to compare models.")
        return
    metrics_path = Path("outputs") / "model_comparison_metrics.csv"
    if not metrics_path.exists():
        st.error(f"Model evaluation metrics file not found: {metrics_path}")
        return
    metrics = pd.read_csv(metrics_path)
    metric_columns = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]
    required_columns = ["Model", *metric_columns]
    missing_columns = [column for column in required_columns if column not in metrics.columns]
    if missing_columns:
        st.error(f"Model evaluation metrics are missing required columns: {', '.join(missing_columns)}")
        return
    metrics[metric_columns] = metrics[metric_columns].apply(pd.to_numeric, errors="coerce")
    metrics = metrics.dropna(subset=metric_columns)
    if metrics.empty:
        st.error("Model evaluation metrics do not contain any valid rows.")
        return

    models = train_comparison_models(portfolio)
    applicant = applicant_features(portfolio, values)
    comparison = pd.DataFrame([
        {"Model": name, "Default probability": model.predict_proba(applicant)[0, 1]}
        for name, model in models.items()
    ]).sort_values("Default probability", ascending=False)
    comparison = comparison.merge(metrics[["Model"]], on="Model", how="inner")
    if comparison.empty:
        st.error("No prediction models match the model evaluation metrics.")
        return

    best_accuracy = metrics.loc[metrics["Accuracy"].idxmax()]
    best_roc_auc = metrics.loc[metrics["ROC_AUC"].idxmax()]
    kpis = st.columns(3)
    for column, label, value in zip(
        kpis,
        ["Models evaluated", f"Best accuracy · {best_accuracy['Model']}", f"Best ROC-AUC · {best_roc_auc['Model']}"],
        [str(len(metrics)), f"{best_accuracy['Accuracy']:.1%}", f"{best_roc_auc['ROC_AUC']:.1%}"],
    ):
        column.markdown(
            f'<div class="metric-card"><h4>{label}</h4><p>{value}</p></div>',
            unsafe_allow_html=True,
        )

    st.subheader("Default prediction by model")
    prediction_chart = go.Figure(go.Bar(
        x=comparison["Model"],
        y=comparison["Default probability"],
        text=comparison["Default probability"].map(
            lambda value: f"{value:.1%} · {'Default' if value >= 0.50 else 'No default'}"
        ),
        textposition="outside",
        textfont=dict(color="#172033", size=14),
        cliponaxis=False,
        marker_color=[
            "#df4c53" if value >= 0.50 else "#2aaf78"
            for value in comparison["Default probability"]
        ],
    ))
    prediction_chart.update_layout(
        height=360,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_color="#172033",
        showlegend=False,
        yaxis=dict(
            tickformat=".0%",
            range=[0, min(1, max(0.6, comparison["Default probability"].max() + 0.18))],
            title="Probability of loan default",
        ),
        margin=dict(l=30, r=30, t=45, b=30),
    )
    st.plotly_chart(prediction_chart, use_container_width=True, config={"displayModeBar": False})

    st.subheader("Model performance")
    performance_chart = go.Figure()
    for metric in metric_columns:
        performance_chart.add_bar(
            name=metric.replace("_", " "),
            x=metrics["Model"],
            y=metrics[metric],
            text=metrics[metric].map(lambda value: f"{value:.1%}"),
            textposition="outside",
            textfont=dict(color="#172033", size=11),
            cliponaxis=False,
        )
    performance_chart.update_layout(
        barmode="group",
        height=430,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_color="#172033",
        yaxis=dict(tickformat=".0%", range=[0, 1.18], title="Score"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=30, r=30, t=70, b=30),
    )
    st.plotly_chart(performance_chart, use_container_width=True, config={"displayModeBar": False})

    st.subheader("Evaluation metrics")
    st.dataframe(
        metrics[required_columns].style.format({column: "{:.1%}" for column in metric_columns}),
        hide_index=True,
        use_container_width=True,
    )


data = load_portfolio()

with st.sidebar:
    st.markdown("## ◈ LoanLens")
    st.caption("RISK INTELLIGENCE PLATFORM")
    st.divider()
    page = st.radio("Page", ["Risk assessment", "Model comparison"])
    st.divider()
    st.markdown("### Applicant profile")
    age = st.slider("Age", 18, 85, 38)
    income = st.number_input("Annual income ($)", 10_000, 500_000, 72_000, step=1_000)
    employment = st.selectbox("Employment type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
    st.markdown("### Credit & loan")
    credit_score = st.slider("Credit score", 300, 850, 680)
    loan_amount = st.number_input("Loan amount ($)", 1_000, 500_000, 45_000, step=1_000)
    interest_rate = st.slider("Interest rate (%)", 1.0, 30.0, 9.5, 0.1)
    term = st.select_slider("Loan term (months)", options=[12, 24, 36, 48, 60, 72], value=36)
    dti = st.slider("Debt-to-income ratio", 0.01, 0.95, 0.31, 0.01)
    has_cosigner = st.radio("Co-signer", ["No", "Yes"], horizontal=True)
    calculate = st.button("Assess application")

values = {"age": age, "income": income, "employment": employment, "credit_score": credit_score,
          "loan_amount": loan_amount, "interest_rate": interest_rate, "term": term, "dti": dti,
          "has_cosigner": has_cosigner}
probability = risk_score(values)
if calculate:
    st.session_state["probability"] = probability
probability = st.session_state.get("probability", probability)

if page == "Model comparison":
    render_model_comparison(data, values)
    st.stop()

st.markdown("""<div class="hero"><div class="eyebrow">Underwriting workspace</div>
<h1>Decide with clarity, lend with confidence.</h1>
<p>A concise risk view for every loan application — designed for faster, more consistent decisions.</p></div>""", unsafe_allow_html=True)
st.write("")

default_rate = data["Default"].mean() if "Default" in data else 0.12
portfolio_count = len(data) if not data.empty else 10_000
cols = st.columns(4)
for col, label, value in zip(cols, ["Default probability", "Risk band", "Portfolio baseline", "Application amount"],
                             [f"{probability:.1%}", "Low" if probability < .12 else "Moderate" if probability < .28 else "High", f"{default_rate:.1%}", f"${loan_amount:,.0f}"]):
    col.markdown(f'<div class="metric-card"><h4>{label}</h4><p>{value}</p></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([1.08, .92], gap="large")
with left:
    st.subheader("Risk assessment")
    band_color = "#22a06b" if probability < .12 else "#eaa10d" if probability < .28 else "#df4c53"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=probability * 100, number={"suffix": "%", "font": {"size": 45}},
        title={"text": "Estimated default probability", "font": {"size": 16}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": band_color, "thickness": .3},
               "steps": [{"range": [0, 12], "color": "#e8f6ef"}, {"range": [12, 28], "color": "#fff5d9"}, {"range": [28, 100], "color": "#fdebed"}]}
    ))
    fig.update_layout(height=300, margin=dict(l=25, r=25, t=55, b=10), paper_bgcolor="white", font_color="#172033")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with right:
    st.subheader("What is influencing this score")
    drivers = pd.DataFrame({"Driver": ["Credit score", "Debt-to-income", "Loan-to-income", "Interest rate", "Co-signer"],
                            "Impact": [-(credit_score - 650) / 90, (dti - .30) * 3.2, (loan_amount / income - .6) * 1.5, (interest_rate - 9) / 7, -.4 if has_cosigner == "Yes" else .25]})
    drivers = drivers.sort_values("Impact")
    fig2 = go.Figure(go.Bar(x=drivers["Impact"], y=drivers["Driver"], orientation="h",
                             marker_color=["#e05a64" if x > 0 else "#2aaf78" for x in drivers["Impact"]]))
    fig2.add_vline(x=0, line_color="#98a2b3")
    fig2.update_layout(height=300, margin=dict(l=150, r=25, t=15, b=10), paper_bgcolor="white", plot_bgcolor="white",
                       xaxis=dict(visible=False), yaxis=dict(automargin=True, tickfont=dict(size=13, color="#172033")))
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

st.divider()
st.subheader("Decision brief")
if probability < .12:
    message = "This application sits below the portfolio default baseline. Standard approval terms appear supportable."
elif probability < .28:
    message = "This application needs a closer review. Consider a lower amount, co-signer, or strengthened documentation."
else:
    message = "The estimated risk is elevated. Escalate for manual underwriting before a lending decision."
st.info(message)
st.caption(f"Portfolio reference: {portfolio_count:,} historical applications. This interface currently uses a transparent demonstration score — connect it to your trained model before production use.")
