import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="UCLA Admission Predictor", layout="wide")


@st.cache_resource
def load_model():
    with open("models/NNmodel.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    return model


@st.cache_resource
def load_scaler():
    with open("models/scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    return scaler


@st.cache_data
def load_data():
    df = pd.read_csv("data/raw/Admission.csv")

    if "Serial_No" in df.columns:
        df = df.drop("Serial_No", axis=1)

    return df


model = load_model()
scaler = load_scaler()
df = load_data()


# Sidebar
st.sidebar.markdown("## 📋 Enter Your Academic Profile")

GRE_Score = st.sidebar.number_input(
    "GRE Score", min_value=260, max_value=340, value=300
)
TOEFL_Score = st.sidebar.number_input(
    "TOEFL Score", min_value=0, max_value=120, value=100
)
University_Rating = st.sidebar.selectbox(
    "University Rating", options=[1, 2, 3, 4, 5], index=0
)
SOP = st.sidebar.number_input(
    "SOP Strength", min_value=1.0, max_value=5.0, value=3.0, step=0.5
)
LOR = st.sidebar.number_input(
    "LOR Strength", min_value=1.0, max_value=5.0, value=3.0, step=0.5
)
CGPA = st.sidebar.number_input(
    "CGPA (out of 10)", min_value=0.0, max_value=10.0, value=8.0, step=0.1
)
Research = st.sidebar.selectbox(
    "Research Experience", options=["No", "Yes"], index=1
)

predict_button = st.sidebar.button("🚀 Predict My Admission Chance")


# Header
st.markdown("# 🎓 UCLA Admission Predictor")
st.write("Predict your admission chance using a trained Neural Network regression model.")


# Prediction
if predict_button:
    research_value = 1 if Research == "Yes" else 0

    input_df = pd.DataFrame([{
        "GRE_Score": GRE_Score,
        "TOEFL_Score": TOEFL_Score,
        "University_Rating": University_Rating,
        "SOP": SOP,
        "LOR": LOR,
        "CGPA": CGPA,
        "Research": research_value
    }])

    input_scaled = scaler.transform(input_df)
    prediction = float(model.predict(input_scaled)[0])
    prediction = max(0.0, min(1.0, prediction))

    pred_percent = prediction * 100

    if pred_percent < 40:
        label = "Low"
    elif pred_percent < 75:
        label = "Moderate"
    else:
        label = "High"

    st.markdown("## 🎯 Prediction Result")
    st.success(f"Predicted Admission Chance: {pred_percent:.2f}%")
    st.info(f"Admission likelihood: {label}")

    if hasattr(model, "loss_curve_"):
        with st.expander("📉 Show Model Loss Curve"):
            fig_loss, ax_loss = plt.subplots(figsize=(8, 4))
            ax_loss.plot(model.loss_curve_, linewidth=2, label="Loss")
            ax_loss.set_title("Neural Network Loss Curve")
            ax_loss.set_xlabel("Iterations")
            ax_loss.set_ylabel("Loss")
            ax_loss.legend()
            st.pyplot(fig_loss)


# Visuals
st.markdown("## 📊 Admission Data Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### GRE vs TOEFL Scores by Admission Probability")
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    scatter1 = ax1.scatter(
        df["GRE_Score"],
        df["TOEFL_Score"],
        c=df["Admit_Chance"],
        cmap="viridis",
        alpha=0.8
    )
    ax1.set_title("GRE vs TOEFL Colored by Admission Probability")
    ax1.set_xlabel("GRE Score")
    ax1.set_ylabel("TOEFL Score")
    fig1.colorbar(scatter1, ax=ax1, label="Admission Probability")
    st.pyplot(fig1)

with col2:
    st.markdown("### CGPA Distribution")
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    sns.histplot(data=df, x="CGPA", kde=True, bins=15, ax=ax2)
    ax2.set_title("CGPA Distribution")
    ax2.set_xlabel("CGPA")
    ax2.set_ylabel("Count")
    st.pyplot(fig2)

st.markdown("## Relationships Between GRE, TOEFL, and CGPA")
st.markdown("### Pairplot Colored by Continuous Admission Probability")

pairplot_df = df[["GRE_Score", "TOEFL_Score", "CGPA", "Admit_Chance"]].copy()

pairplot_fig = sns.pairplot(
    pairplot_df,
    vars=["GRE_Score", "TOEFL_Score", "CGPA"],
    hue="Admit_Chance",
    palette="viridis",
    diag_kind="hist",
    plot_kws={"s": 50, "alpha": 0.8}
)

st.pyplot(pairplot_fig.fig)

st.markdown("### Actual vs Predicted Admission Probability")

full_X = df.drop("Admit_Chance", axis=1)
full_y = df["Admit_Chance"]
full_X_scaled = scaler.transform(full_X)
full_pred = model.predict(full_X_scaled)

fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=full_y, y=full_pred, ax=ax3)
min_val = min(np.min(full_y), np.min(full_pred))
max_val = max(np.max(full_y), np.max(full_pred))
ax3.plot([min_val, max_val], [min_val, max_val], linestyle="--")
ax3.set_title("Actual vs Predicted Admission Probability")
ax3.set_xlabel("Actual")
ax3.set_ylabel("Predicted")
st.pyplot(fig3)
