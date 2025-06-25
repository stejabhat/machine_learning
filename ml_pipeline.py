# filename: ml_pipeline.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Step 1: Load data
df = pd.read_csv("train_dataset.csv")

# Step 2: Split features and target
X = df.drop(columns=["Label"])
y = df["Label"]

# Step 3: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Step 5: Train Random Forest
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Step 6: Predict and evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
conf = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
# ... same as above

# Plot confusion matrix
sns.heatmap(conf, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

# ✅ Save AND Show
plt.savefig("confusion_matrix.png")  # Saves image
plt.show()  # ✅ Shows image window (if GUI supported)

# DO NOT call plt.close() before plt.show()



# Step 7: Save model and scaler
joblib.dump(model, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
