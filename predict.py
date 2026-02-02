import joblib

model = joblib.load("chat_detector_model.pkl")

def predict_chat(chat_text: str):
    pred = model.predict([chat_text])[0]

    if pred == "fake":
        return "🤥 FAKE CHAT", "Patterns look copied / scammy language detected."
    elif pred == "real":
        return "😇 REAL CHAT", "Natural conversation flow detected."
    else:
        return "⚠️ SUSPICIOUS", "Mixed patterns or unusual wording detected."
