# filename: app.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model & scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🚦 Network Traffic Classifier")

uploaded = st.file_uploader("Upload CSV with network traffic:", type="csv")

if uploaded:
    data = pd.read_csv(uploaded)

    # Drop label column if present
    if "Label" in data.columns:
        data = data.drop(columns=["Label"])

    # Scale and predict
    scaled_data = scaler.transform(data)
    preds = model.predict(scaled_data)

    # Show raw predictions
    st.subheader("🧾 Predictions:")
    st.write(preds)

    # Count predictions
    pred_df = pd.DataFrame(preds, columns=["Predicted Label"])
    pred_counts = pred_df["Predicted Label"].value_counts().sort_index()

    # Plot
    st.subheader("📊 Prediction Summary")
    fig, ax = plt.subplots()
    sns.barplot(x=pred_counts.index, y=pred_counts.values, palette="Blues_d", ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Predictions")
    st.pyplot(fig)
else:
    st.info("📁 Please upload a CSV file to begin.")
