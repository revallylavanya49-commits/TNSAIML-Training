import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #151515;
    color: #F5F5F5;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #202020;
    border-right: 1px solid #B99746;
}

[data-testid="stSidebar"] * {
    color: #F5F5F5;
}

/* Main headings */
h1, h2, h3 {
    color: #D9B45B !important;
}

/* Hero banner */
.hero {
    background: linear-gradient(120deg, #080808, #292929);
    border: 1px solid #B99746;
    padding: 35px;
    border-radius: 18px;
    margin-bottom: 25px;
    box-shadow: 0 5px 20px rgba(185,151,70,0.12);
}

.hero h1 {
    color: #E3BE65 !important;
    font-size: 36px;
}

.hero p {
    color: #E5E5E5;
    font-size: 16px;
}

/* Section titles */
.section-title {
    color: #E3BE65;
    font-size: 25px;
    font-weight: bold;
    margin-top: 25px;
    margin-bottom: 15px;
}

/* Dashboard cards */
.card {
    background: #242424;
    border: 1px solid #A8873E;
    padding: 22px;
    border-radius: 15px;
    min-height: 145px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
}

.card h3 {
    color: #E3BE65;
}

.card p {
    color: #E0E0E0;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: #242424;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #A8873E;
}

div[data-testid="stMetricLabel"] {
    color: #E3BE65;
}

div[data-testid="stMetricValue"] {
    color: #FFFFFF;
}

/* Forms */
div[data-testid="stForm"] {
    background: #222222;
    padding: 25px;
    border-radius: 16px;
    border: 1px solid #B99746;
}

/* Input fields */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background-color: #303030;
    color: white;
    border-radius: 8px;
}

/* Labels */
.stSelectbox label,
.stNumberInput label,
.stTextInput label {
    color: #E3BE65 !important;
}

/* Buttons */
.stButton button,
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, #A8873E, #E3BE65);
    color: #111111 !important;
    border: none;
    border-radius: 10px;
    min-height: 48px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    background: #FFFFFF;
    color: #111111 !important;
    border: 1px solid #D9B45B;
}

/* Radio buttons */
.stRadio label {
    color: #F5F5F5;
}

/* Information boxes */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Footer */
.footer {
    background: #242424;
    border: 1px solid #A8873E;
    padding: 20px;
    text-align: center;
    border-radius: 12px;
    margin-top: 30px;
    color: #E3BE65;
}

hr {
    border-color: #A8873E;
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():

    artifact = joblib.load(MODEL_PATH)

    return (
        artifact["model"],
        artifact["features"],
        artifact["input_schema"]
    )


if not MODEL_PATH.exists():

    st.error(
        "model.joblib not found. "
        "Keep it in the same folder as app.py."
    )

    st.stop()


try:

    model, features, schema = load_model()

except Exception as e:

    st.error(f"Model loading error: {e}")
    st.stop()

with st.sidebar:

    st.markdown("## 🩺 DIABETES CARE")

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Risk Assessment"
        ]
    )

    st.markdown("---")

    st.markdown("### MODEL INFORMATION")

    st.metric("Input Features", len(features))

    st.write("**Algorithm:** Logistic Regression")
    st.write("**Framework:** Streamlit")
    st.write("**Language:** Python")

    st.markdown("---")

    st.info(
        "Enter patient information carefully "
        "for model-based assessment."
    )


if page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>🩺 Diabetes Risk Prediction</h1>
        <p>PREMIUM HEALTHCARE MONITORING DASHBOARD</p>
        <p>Machine Learning Based Early Stage Diabetes Assessment</p>
    </div>
    """, unsafe_allow_html=True)


    st.markdown(
        '<div class="section-title">Dashboard Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Input Features",
            len(features)
        )

    with col2:

        st.metric(
            "ML Algorithm",
            "Logistic Regression"
        )

    with col3:

        st.metric(
            "Prediction Type",
            "Classification"
        )


    st.markdown(
        '<div class="section-title">Welcome to Diabetes Care</div>',
        unsafe_allow_html=True
    )

    a, b = st.columns(2)

    with a:

        st.markdown("""
        <div class="card">
            <h3>👤 Patient Assessment</h3>
            <p>Enter patient details and clinical symptoms
            to generate a machine learning prediction.</p>
        </div>
        """, unsafe_allow_html=True)

    with b:

        st.markdown("""
        <div class="card">
            <h3>📊 Risk Analysis</h3>
            <p>View the predicted class and probability
            distribution from the trained model.</p>
        </div>
        """, unsafe_allow_html=True)


    st.markdown(
        '<div class="section-title">How It Works</div>',
        unsafe_allow_html=True
    )

    x, y, z = st.columns(3)

    with x:

        st.markdown("""
        <div class="card">
            <h3>01</h3>
            <p>Enter patient information.</p>
        </div>
        """, unsafe_allow_html=True)

    with y:

        st.markdown("""
        <div class="card">
            <h3>02</h3>
            <p>Submit clinical symptoms.</p>
        </div>
        """, unsafe_allow_html=True)

    with z:

        st.markdown("""
        <div class="card">
            <h3>03</h3>
            <p>View prediction results.</p>
        </div>
        """, unsafe_allow_html=True)


    st.markdown("---")

    st.info(
        "Select Risk Assessment from the sidebar to begin."
    )


# =====================================
# PAGE 2: RISK ASSESSMENT
# =====================================

elif page == "Risk Assessment":

    st.markdown("""
    <div class="hero">
        <h1>👤 Patient Risk Assessment</h1>
        <p>Enter patient information and clinical symptoms</p>
    </div>
    """, unsafe_allow_html=True)


    st.markdown(
        '<div class="section-title">Patient Information & Symptoms</div>',
        unsafe_allow_html=True
    )

    patient_data = {}

    with st.form("patient_form"):

        cols = st.columns(2, gap="large")

        for i, feature in enumerate(features):

            field = schema[feature]

            label = feature.replace("_", " ").title()

            with cols[i % 2]:

                if field["kind"] == "number":

                    default = float(field["default"])

                    if field["integer"]:

                        patient_data[feature] = st.number_input(
                            label,
                            value=int(default),
                            min_value=1,
                            max_value=120,
                            step=1
                        )

                    else:

                        patient_data[feature] = st.number_input(
                            label,
                            value=default,
                            step=0.1,
                            format="%.2f"
                        )

                else:

                    patient_data[feature] = st.selectbox(
                        label,
                        field["categories"]
                    )


        st.markdown("---")

        submitted = st.form_submit_button(
            "🔍 PREDICT DIABETES RISK",
            use_container_width=True
        )


    if submitted:

        patient_df = pd.DataFrame(
            [patient_data],
            columns=features
        )

        try:

            prediction = model.predict(patient_df)[0]

            st.markdown("---")

            st.markdown(
                '<div class="section-title">📊 Prediction Results</div>',
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)

            positive = (
                str(prediction).strip().lower() == "positive"
            )


            with col1:

                if positive:

                    st.error("### Positive Prediction")

                    st.write(
                        "The model classified the entered "
                        "patient information as Positive."
                    )

                else:

                    st.success("### Negative Prediction")

                    st.write(
                        "The model classified the entered "
                        "patient information as Negative."
                    )


                st.metric(
                    "Predicted Class",
                    str(prediction)
                )


            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(patient_df)[0]

                classes = list(model.classes_)

                probability_df = pd.DataFrame({

                    "Class": [str(c) for c in classes],

                    "Probability (%)": [
                        float(p) * 100
                        for p in probabilities
                    ]

                }).set_index("Class")


                positive_index = next(
                    (
                        i for i, label in enumerate(classes)
                        if str(label).strip().lower() == "positive"
                    ),
                    None
                )


                with col2:

                    if positive_index is not None:

                        st.metric(
                            "Positive Class Probability",
                            f"{probabilities[positive_index] * 100:.2f}%"
                        )

                    st.markdown("### Probability Distribution")


                st.bar_chart(
                    probability_df,
                    y="Probability (%)"
                )


            st.warning(
                "This application is for educational purposes only. "
                "It is not a medical diagnosis. Consult a qualified "
                "healthcare professional for medical evaluation."
            )


        except Exception as e:

            st.error(f"Prediction error: {e}")



st.markdown("""
<div class="footer">
    <h3>🩺 DIABETES RISK PREDICTION</h3>
    Premium Healthcare Monitoring Dashboard<br>
    <small>Machine Learning Academic Project | Educational Purpose Only</small>
</div>
""", unsafe_allow_html=True)