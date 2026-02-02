from flask import Flask, render_template, request
from predict import predict_chat

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    reason = None
    chat_text = ""

    if request.method == "POST":
        chat_text = request.form.get("chat", "")
        result, reason = predict_chat(chat_text)

    return render_template("index.html", result=result, reason=reason, chat_text=chat_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
