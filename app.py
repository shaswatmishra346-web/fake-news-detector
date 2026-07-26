from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# ----------------------------
# Load model and vectorizer once at startup
# ----------------------------
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


def clean_text(text):
    text = str(text)
    if "(Reuters)" in text:
        text = text.split("(Reuters)", 1)[-1]
    return text.strip()


def predict_news(text):
    text_clean = clean_text(text)
    vec = vectorizer.transform([text_clean])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label = "REAL" if pred == 1 else "FAKE"
    confidence = float(prob[pred])
    return label, confidence


def extract_article_from_url(url):
    """Try newspaper3k first, fall back to raw requests + BeautifulSoup."""
    try:
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        if article.text and len(article.text.strip().split()) >= 15:
            return {"success": True, "title": article.title, "text": article.text}
    except Exception:
        pass

    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {"User-Agent": "Mozilla/5.0 (compatible; Verascope/1.0)"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")

        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs).strip()
        title = soup.title.string.strip() if soup.title and soup.title.string else ""

        if text and len(text.split()) >= 15:
            return {"success": True, "title": title, "text": text}
        return {"success": False, "error": "Not enough readable article text found on this page."}
    except Exception as e:
        return {"success": False, "error": f"Could not fetch this URL ({e})."}


# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/predict-text", methods=["POST"])
def api_predict_text():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "")

    if not text or len(text.strip().split()) < 15:
        return jsonify({
            "success": False,
            "error": "Please provide at least a few sentences of article text."
        }), 400

    label, confidence = predict_news(text)
    return jsonify({"success": True, "label": label, "confidence": confidence})


@app.route("/api/predict-url", methods=["POST"])
def api_predict_url():
    data = request.get_json(force=True, silent=True) or {}
    url = data.get("url", "").strip()

    if not url.lower().startswith(("http://", "https://")):
        return jsonify({"success": False, "error": "Please provide a valid URL."}), 400

    result = extract_article_from_url(url)
    if not result["success"]:
        return jsonify({
            "success": False,
            "error": result.get("error", "Could not extract article text from this URL.")
        }), 422

    label, confidence = predict_news(result["text"])
    return jsonify({
        "success": True,
        "label": label,
        "confidence": confidence,
        "title": result.get("title", ""),
        "extracted_text": result["text"][:3000],
    })


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
