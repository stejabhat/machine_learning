import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load model and label encoder
model = joblib.load("emotion_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Streamlit UI
st.title("🎭 Emotion Detection from Text")
user_input = st.text_area("📝 Enter your text:", "")

if st.button("🔍 Predict Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Predict probabilities
        probabilities = model.predict_proba([user_input])[0]
        predicted_class = np.argmax(probabilities)
        emotion = label_encoder.inverse_transform([predicted_class])[0]

        st.success(f"🎯 Predicted Emotion: **{emotion}**")

        # Prepare data for plotting
        emotion_names = label_encoder.classes_
        prob_df = pd.DataFrame({
            'Emotion': emotion_names,
            'Probability': probabilities
        }).sort_values(by='Probability', ascending=True)

        # Plot horizontal bar chart
        st.subheader("📊 Prediction Confidence")
        fig, ax = plt.subplots(figsize=(8, len(emotion_names) * 0.4))
        ax.barh(prob_df['Emotion'], prob_df['Probability'], color='skyblue')
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_title("Model Confidence per Emotion")
        st.pyplot(fig)
