# app.py
from flask import Flask, render_template, request
from datetime import date
import sqlite3
import hashlib

from predict import predict_with_details

app = Flask(__name__)

DB_PATH = "trend.db"

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS trending (
      day TEXT,
      msg_hash TEXT,
      sample TEXT,
      count INTEGER,
      PRIMARY KEY (day, msg_hash)
    )
    """)
    con.commit()
    con.close()

def add_trending(msg: str):
    d = date.today().isoformat()
    h = hashlib.sha1((msg or "").strip().lower().encode("utf-8")).hexdigest()
    sample = (msg or "").strip().replace("\n", " ")[:120]

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT count FROM trending WHERE day=? AND msg_hash=?", (d, h))
    row = cur.fetchone()
    if row:
        cur.execute("UPDATE trending SET count=count+1 WHERE day=? AND msg_hash=?", (d, h))
    else:
        cur.execute("INSERT INTO trending(day, msg_hash, sample, count) VALUES(?,?,?,1)", (d, h, sample))
    con.commit()
    con.close()

def get_trending(limit=5):
    d = date.today().isoformat()
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT sample, count FROM trending WHERE day=? ORDER BY count DESC LIMIT ?", (d, limit))
    rows = cur.fetchall()
    con.close()
    return [{"sample": r[0], "count": r[1]} for r in rows]

init_db()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    prob_map = {}
    explain = None
    chat_text = ""
    highlighted_html = ""
    urls = []
    trending = get_trending()

    if request.method == "POST":
        chat_text = request.form.get("chat", "").strip()

        # Trending count always (even short)
        if chat_text:
            add_trending(chat_text)
            trending = get_trending()

        # Meaningful short messages allow
        words = chat_text.split()
        if len(chat_text) < 6 and len(words) <= 1:
            result = "📝 MORE TEXT NEEDED"
            explain = "कृपया थोड़ा बड़ा मैसेज डालें (कम से कम 3–4 शब्द)।"
            confidence = 10.0
            prob_map = {}
            highlighted_html = ""
            urls = []
        else:
            (result, confidence, prob_map, explain,
             highlighted_html, urls, fake_hits, suspicious_hits, real_hits) = predict_with_details(chat_text)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        prob_map=prob_map,
        explain=explain,
        chat=chat_text,
        highlighted_html=highlighted_html,
        urls=urls,
        trending=trending
    )

@app.route("/card", methods=["POST"])
def card():
    # Shareable card page (render only)
    result = request.form.get("result", "")
    confidence = request.form.get("confidence", "")
    real_p = request.form.get("real_p", "0")
    fake_p = request.form.get("fake_p", "0")
    sus_p = request.form.get("sus_p", "0")
    return render_template("card.html", result=result, confidence=confidence, real_p=real_p, fake_p=fake_p, sus_p=sus_p)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
