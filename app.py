from flask import Flask, render_template, request
from predict import predict_with_details

app = Flask(__name__)


def is_meaningful_short_text(text: str) -> bool:
    """
    छोटे लेकिन meaningful messages allow करने के लिए:
    - कम से कम 3 words हों
    - और words में कोई meaningful word हो
    """
    words = text.lower().split()
    meaningful_words = {
        "love", "miss", "sorry", "thanks", "thank", "please",
        "busy", "call", "meet", "meeting", "tomorrow", "today",
        "yes", "no", "okay", "ok", "fine", "hello", "hi"
    }
    return len(words) >= 3 and any(w in meaningful_words for w in words)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    keywords = []
    prob_map = {}
    explain = None
    chat_text = ""

    if request.method == "POST":
        chat_text = request.form.get("chat", "").strip()

        words_count = len(chat_text.split())
        chars_count = len(chat_text)

        # ✅ Rule:
        # 1) अगर बहुत छोटा है AND meaningful भी नहीं है -> MORE TEXT NEEDED
        # 2) अगर meaningful short है (जैसे "I love you") -> prediction चलाओ
        if (chars_count < 20 or words_count < 4) and (not is_meaningful_short_text(chat_text)):
            result = "📝 MORE TEXT NEEDED"
            explain = "Please paste a longer chat (min 4–5 words / 20+ chars) for accurate detection."
            confidence = 10.0  # orange
            prob_map = {}      # scores hide
            keywords = []      # keywords hide
        else:
            # ✅ NORMAL prediction (including meaningful short texts)
            result, confidence, keywords, prob_map, explain = predict_with_details(chat_text)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        keywords=keywords,
        prob_map=prob_map,
        explain=explain,
        chat=chat_text
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
