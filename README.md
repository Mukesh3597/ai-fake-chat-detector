🤖 AI Fake WhatsApp Chat Detector

An AI-powered web app that detects whether a WhatsApp chat/message is REAL, FAKE, or SUSPICIOUS, along with confidence scores.

⚠️ Disclaimer: This tool is for awareness & entertainment only.
It helps identify scam/forward patterns but does not guarantee 100% accuracy.

🚀 Live Demo

👉 (Add your Render URL here)
https://ai-fake-chat-detector.onrender.com

✨ Features

🔍 Detects Fake / Real / Suspicious WhatsApp messages

📊 Shows confidence percentages (REAL / FAKE / SUSPICIOUS)

🎨 Color-coded confidence (🟢🟡🟠)

🧠 Machine Learning (TF-IDF + Logistic Regression)

🌐 Simple Flask web interface

📱 User-friendly & screenshot-ready UI

🧠 How It Works

User pastes a WhatsApp chat/message

Text is processed using TF-IDF

ML model predicts probabilities for:

Real

Fake

Suspicious

App displays result + confidence scores

🛠️ Tech Stack

Python

Flask

scikit-learn

pandas

Joblib

HTML / CSS

Gunicorn (for deployment)

📂 Project Structure
AI_project/
│
├── app.py
├── predict.py
├── train.py
├── data.csv
├── chat_detector_model.pkl
├── requirements.txt
├── Procfile
├── runtime.txt
├── README.md
│
└── templates/
    └── index.html

⚙️ Installation & Run (Local)
pip install -r requirements.txt
python app.py


Open in browser:

http://127.0.0.1:5000

🧪 Model Training

To retrain the model with new data:

python train.py


The trained model is saved as:

chat_detector_model.pkl

🧾 Dataset

The dataset (data.csv) contains three labels:

real

fake

suspicious

Example:

text,label
"I love you",real
"Your account is blocked, verify KYC",fake
"Forward this to 10 people",suspicious

📌 Example Output
This message is: REAL
Confidence:
REAL: 70%
FAKE: 10%
SUSPICIOUS: 20%

🔮 Future Improvements

📈 Larger & multilingual dataset

🌍 Hindi/English toggle

📡 Public REST API

📊 Accuracy improvements

📱 WhatsApp-style UI

👨‍💻 Author

Mukesh
GitHub: https://github.com/Mukesh3597

⭐ Support

If you like this project:

⭐ Star the repository

🧠 Share for awareness

🤝 Contributions welcome