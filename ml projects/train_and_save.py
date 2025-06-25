# train_and_save.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("sentimentdataset.csv")
df = df[['Text', 'Sentiment']].dropna()
df['Sentiment'] = df['Sentiment'].str.strip()

min_samples = 5
valid_labels = df['Sentiment'].value_counts()[lambda x: x >= min_samples].index
df = df[df['Sentiment'].isin(valid_labels)].copy()

le = LabelEncoder()
df['Label'] = le.fit_transform(df['Sentiment'])

X_train, _, y_train, _ = train_test_split(df['Text'], df['Label'], test_size=0.2, random_state=42, stratify=df['Label'])

model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])
model.fit(X_train, y_train)

joblib.dump(model, "emotion_model.pkl")
joblib.dump(le, "label_encoder.pkl")
