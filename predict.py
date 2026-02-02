import joblib
import numpy as np

model = joblib.load("chat_detector_model.pkl")

def predict_with_details(text: str):
    proba = model.predict_proba([text])[0]
    classes = list(model.classes_)  # e.g. ['fake','real','suspicious'] (order can vary)

    prob_map = {classes[i]: round(float(proba[i]) * 100, 2) for i in range(len(classes))}
    best_idx = int(np.argmax(proba))
    best_label = classes[best_idx]
    best_conf = prob_map[best_label]

    # Emoji label for UI
    if best_label == "real":
        result = "😇 REAL"
        explain = "Looks like normal human conversation."
    elif best_label == "fake":
        result = "🤥 FAKE"
        explain = "Scam/Threat pattern detected."
    else:
        result = "⚠️ SUSPICIOUS"
        explain = "Mixed/Unusual patterns detected."

    # keywords optional (keep empty if you want)
    keywords = []
    return result, best_conf, keywords, prob_map, explain
