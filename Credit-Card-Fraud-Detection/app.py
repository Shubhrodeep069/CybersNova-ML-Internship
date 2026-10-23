import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("models/random_forest_fraud_model.pkl")
scaler = joblib.load("models/transaction_scaler.pkl")

FINAL_THRESHOLD = 0.4

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection")
st.write(
    "Enter transaction details to determine whether the transaction "
    "is likely to be fraudulent."
)

with st.sidebar:
    st.header("📊 Model Information")

    st.write("**Model:** Random Forest")
    st.write("**Threshold:** 0.40")
    st.write("**Precision:** 88.04%")
    st.write("**Recall:** 82.65%")
    st.write("**F1-Score:** 85.26%")

    st.divider()

    st.caption(
        "This application uses a Random Forest model "
        "trained on the Credit Card Fraud Detection dataset."
    )

st.divider()

# Input fields
st.subheader("Transaction Details")

if "sample_transaction" not in st.session_state:
    st.session_state.sample_transaction = None

col_sample1, col_sample2 = st.columns(2)

if "sample_transaction" not in st.session_state:
    st.session_state.sample_transaction = None

with col_sample1:
    if st.button("🚨 Load Fraud Sample"):
        sample = pd.read_csv("data/creditcard.csv")
        fraud_sample = sample[sample["Class"] == 1].iloc[0]
        st.session_state.sample_transaction = fraud_sample

with col_sample2:
    if st.button("✅ Load Normal Sample"):
        sample = pd.read_csv("data/creditcard.csv")
        normal_sample = sample[sample["Class"] == 0].iloc[0]
        st.session_state.sample_transaction = normal_sample


sample = st.session_state.sample_transaction

col1, col2 = st.columns(2)

with col1:
    time = st.number_input(
        "Transaction Time",
        value=float(sample["Time"]) if sample is not None else 0.0
    )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=float(sample["Amount"]) if sample is not None else 0.0
    )

with col2:
    st.info(
        "V1–V28 are PCA-transformed features from the original dataset."
    )

st.subheader("PCA Features")

features = {}

cols = st.columns(4)

for i in range(1, 29):
    with cols[(i - 1) % 4]:
        features[f"V{i}"] = st.number_input(
            f"V{i}",
            value=float(sample[f"V{i}"]) if sample is not None else 0.0,
            format="%.6f"
        )

if st.button("🔍 Detect Fraud", type="primary"):

    # Create input dataframe
    transaction = pd.DataFrame([{
        "Time": time,
        **features,
        "Amount": amount
    }])

    # Scale Time and Amount
    transaction[["Time", "Amount"]] = scaler.transform(
        transaction[["Time", "Amount"]]
    )

    # Predict probability
    fraud_probability = model.predict_proba(transaction)[0][1]

    # Apply threshold
    prediction = fraud_probability >= FINAL_THRESHOLD

    st.divider()

    if prediction:
        st.error("🚨 Potential Fraud Detected")
    else:
        st.success("✅ Transaction Appears Normal")

    st.metric(
        "Fraud Probability",
        f"{fraud_probability * 100:.2f}%"
    )

    st.caption(
        f"Decision threshold: {FINAL_THRESHOLD:.2f}"
    )

    st.progress(
    min(int(fraud_probability * 100), 100),
    text="Fraud Probability"
)