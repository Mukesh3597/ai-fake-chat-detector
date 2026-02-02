# train.py
# ✅ Final Training Script (Better accuracy + better confidence probabilities)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# 1) Load dataset
data = pd.read_csv("data.csv")

# 2) Basic cleaning + validation
data["text"] = data["text"].astype(str).fillna("").str.strip()
data["label"] = data["label"].astype(str).fillna("").str.strip().str.lower()

# Keep only valid labels
valid_labels = {"real", "fake", "suspicious"}
data = data[data["label"].isin(valid_labels)]

# Drop empty text rows
data = data[data["text"].str.len() > 0]

if len(data) < 30:
    print("⚠️ Warning: Dataset is too small. Add more rows in data.csv for better accuracy.")
    print("Current rows:", len(data))

# 3) Features and Labels
X = data["text"]
y = data["label"]

# 4) Train / Test split (stratify keeps class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y if y.nunique() > 1 else None
)

# 5) Better TF-IDF + Balanced classifier (improves confidence outputs)
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),      # unigrams + bigrams
        max_features=30000,      # limit vocab
        min_df=1                 # keep all terms (small data friendly)
    )),
    ("clf", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"  # helps when classes uneven
    ))
])

# 6) Train
model.fit(X_train, y_train)

# 7) Evaluate
accuracy = model.score(X_test, y_test)
print("✅ Model Accuracy:", round(accuracy, 4))

# 8) Save model
joblib.dump(model, "chat_detector_model.pkl")
print("✅ Model saved as chat_detector_model.pkl")
