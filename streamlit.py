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

    # Create a binary class only for visualization
    df["Admit_Class"] = (df["Admit_Chance"] >= 0.75).astype(int)

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


# Main header
st.markdown("# 🎓 UCLA Admission Predictor")
st.write("Predict your admission chance using a trained Neural Network model.")


# Prediction section
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

    # Keep prediction inside valid probability range
    prediction = max(0.0, min(1.0, prediction))

    st.markdown("## 🎯 Prediction Result")

    if prediction < 0.40:
        st.error(f"❌ Low chance of admission. (Probability: {prediction:.2f})")
    elif prediction < 0.75:
        st.warning(f"⚠️ Moderate chance of admission. (Probability: {prediction:.2f})")
    else:
        st.success(f"✅ High chance of admission! (Probability: {prediction:.2f})")

    st.markdown(
        f"""
        <div style="
            background-color:#d4edda;
            padding:16px;
            border-radius:10px;
            font-size:20px;
            margin-bottom:20px;
        ">
            Estimated Admission Chance: <b>{prediction * 100:.2f}%</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Show training loss curve if available
    if hasattr(model, "loss_curve_"):
        with st.expander("📉 Show Model Loss Curve"):
            fig_loss, ax_loss = plt.subplots(figsize=(8, 4))
            ax_loss.plot(model.loss_curve_, linewidth=2, label="Loss")
            ax_loss.set_title("Neural Network Loss Curve")
            ax_loss.set_xlabel("Iterations")
            ax_loss.set_ylabel("Loss")
            ax_loss.legend()
            st.pyplot(fig_loss)


# Graphs
st.markdown("## 📊 Admission Data Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### GRE vs TOEFL Scores by Admission Chance")
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=df,
        x="GRE_Score",
        y="TOEFL_Score",
        hue="Admit_Class",
        palette="Set1",
        ax=ax1
    )
    ax1.set_title("GRE vs TOEFL Score by Admission Chance")
    st.pyplot(fig1)

with col2:
    st.markdown("### CGPA Distribution by Admission Chance")
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    sns.histplot(
        data=df,
        x="CGPA",
        hue="Admit_Class",
        kde=True,
        bins=15,
        ax=ax2
    )
    ax2.set_title("CGPA Distribution by Admission Chance")
    st.pyplot(fig2)

st.markdown("## Relationships Between GRE, TOEFL, and CGPA")

pairplot_df = df[["GRE_Score", "TOEFL_Score", "CGPA", "Admit_Class"]].copy()

pairplot_fig = sns.pairplot(
    pairplot_df,
    vars=["GRE_Score", "TOEFL_Score", "CGPA"],
    hue="Admit_Class",
    diag_kind="kde",
    plot_kws={"s": 50, "alpha": 0.8}
)

st.pyplot(pairplot_fig.fig)

