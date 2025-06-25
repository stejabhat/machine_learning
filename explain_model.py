# filename: explain_model.py

import shap
import joblib
import pandas as pd

# Load model and data
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("train_dataset.csv")
X = df.drop(columns=["Label"])
X_scaled = scaler.transform(X)

# SHAP Explainer
explainer = shap.Explainer(model, X_scaled)
shap_values = explainer(X_scaled)

# Plot feature importance
shap.summary_plot(shap_values, X, plot_type="bar")
