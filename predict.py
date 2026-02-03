# predict.py
import re
import joblib
import numpy as np
from markupsafe import escape

model = joblib.load("chat_detector_model.pkl")

# Words lists (simple + effective)
FAKE_WORDS = {
    "otp","kyc","blocked","block","verify","verification","account","refund","win","prize",
    "click","link","urgent","fee","pay","payment","bank","upi","password","limited","offer"
}

SUSPICIOUS_WORDS = {
    "forward","share","breaking","viral","everyone","group","must","immediately","dont ignore","don’t ignore",
    "send to","10 people","5 friends","asap"
}

REAL_HINT_WORDS = {
    "please","thanks","thank","sorry","call","meet","tomorrow","today","homework","class",
    "okay","ok","fine","where","when","time","bro","sis","mom","dad","love","miss"
}

URL_RE = re.compile(r"(https?://[^\s]+|www\.[^\s]+)", re.IGNORECASE)

def extract_urls(text: str):
    return URL_RE.findall(text or "")

def find_hits(text: str, word_set):
    t = (text or "").lower()
    hits = []
    for w in word_set:
        if w in t:
            hits.append(w)
    return hits[:8]

def highlight_text(text: str, fake_hits, real_hits, suspicious_hits):
    """
    Returns safe HTML with highlighted spans.
    Priority: fake > suspicious > real
    """
    if not text:
        return ""

    # Escape first to avoid XSS
    safe = escape(text)

    def repl(words, color):
        nonlocal safe
        for w in sorted(words, key=len, reverse=True):
            if len(w) < 2:
                continue
            # regex on escaped text (case-insensitive)
            pattern = re.compile(re.escape(w), re.IGNORECASE)
            safe = pattern.sub(f'<span style="padding:2px 6px;border-radius:8px;background:{color};font-weight:700;">\\g<0></span>', safe)
        return safe

    # Order matters
    safe = repl(fake_hits, "#ffe1e1")          # light red
    safe = repl(suspicious_hits, "#fff0c9")    # light yellow
    safe = repl(real_hits, "#dff7e6")          # light green
    return safe

def build_explanation(best_label, fake_hits, suspicious_hits, urls):
    if best_label == "fake":
        reason = "यह मैसेज FAKE लग रहा है क्योंकि इसमें scam/pressure वाले शब्द मिले: " + ", ".join(fake_hits[:6] or ["(keywords नहीं मिले)"])
        if urls:
            reason += " और इसमें link/source भी है।"
        return reason
    if best_label == "suspicious":
        reason = "यह मैसेज SUSPICIOUS लग रहा है क्योंकि इसमें forward/viral टाइप संकेत मिले: " + ", ".join(suspicious_hits[:6] or ["(keywords नहीं मिले)"])
        if urls:
            reason += " और इसमें link भी है, ध्यान रखें।"
        return reason
    return "यह मैसेज REAL लग रहा है क्योंकि भाषा सामान्य बातचीत जैसी है (कोई scam/forward pattern strong नहीं मिला)।"

def predict_with_details(text: str):
    proba = model.predict_proba([text])[0]
    classes = list(model.classes_)  # ['fake','real','suspicious'] order can vary

    prob_map = {classes[i]: round(float(proba[i]) * 100, 2) for i in range(len(classes))}
    best_idx = int(np.argmax(proba))
    best_label = classes[best_idx]
    best_conf = prob_map[best_label]

    # hits
    fake_hits = find_hits(text, FAKE_WORDS)
    suspicious_hits = find_hits(text, SUSPICIOUS_WORDS)
    real_hits = find_hits(text, REAL_HINT_WORDS)
    urls = extract_urls(text)

    # If confidence too low → suspicious
    if best_conf < 45:
        best_label = "suspicious"

    # Result label
    if best_label == "real":
        result = "😇 REAL"
    elif best_label == "fake":
        result = "🤥 FAKE"
    else:
        result = "⚠️ SUSPICIOUS"

    explain = build_explanation(best_label, fake_hits, suspicious_hits, urls)

    # Highlighted HTML
    highlighted_html = highlight_text(text, fake_hits, real_hits, suspicious_hits)

    return result, best_conf, prob_map, explain, highlighted_html, urls, fake_hits, suspicious_hits, real_hits
