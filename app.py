import os
import json
import requests
from flask import Flask, render_template, request, jsonify

# --- ADDED FOR CHATBOT: import the new blueprint (isolated in chatbot.py) ---
from chatbot import chatbot_bp

app = Flask(__name__)
# --- ADDED FOR CHATBOT: register the /api/chat route ---
app.register_blueprint(chatbot_bp)

# ----------------------------
# Groq API setup (matches chatbot.py's setup — same env var, same endpoint)
# ----------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Using the larger model here (vs chatbot.py's 8b-instant) since this is the
# core verdict logic and benefits more from stronger reasoning than raw speed.
GROQ_MODEL = "llama-3.3-70b-versatile"


def predict_news_llm(text):
    """
    Ask an LLM to judge whether a piece of text reads as REAL or FAKE news,
    and return a label, confidence score, and short reasoning.
    """
    prompt = f"""You are a fact-checking assistant. Analyze the following text and judge whether it reads as REAL (legitimate, credible news) or FAKE (misinformation, fabricated, or highly misleading news).

Base your judgment on:
- Internal factual consistency and plausibility
- Presence of verifiable, specific details (names, dates, institutions) vs vague claims
- Sensationalist, emotionally manipulative, or clickbait-style language
- Whether claims match things you know to be true, or contradict well-established facts
- Tone and structure typical of credible journalism vs propaganda/satire/fabrication

Text to analyze:
\"\"\"
{text[:4000]}
\"\"\"

Respond with ONLY a JSON object, no other text, in this exact format:
{{"label": "REAL" or "FAKE", "confidence": <float between 0 and 1>, "reasoning": "<1-2 sentence explanation>"}}"""

    if not GROQ_API_KEY:
        return "REAL", 0.5, "Chatbot/model service is not configured (missing GROQ_API_KEY)."

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 300,
            },
            timeout=20,
        )
        resp.raise_for_status()
        payload = resp.json()
        raw = payload["choices"][0]["message"]["content"].strip()
    except (requests.exceptions.RequestException, KeyError, IndexError) as e:
        return "REAL", 0.5, f"Could not reach the verification service ({e})."

    # Strip markdown code fences if the model added them despite instructions
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        result = json.loads(raw)
        label = result.get("label", "REAL").upper()
        if label not in ("REAL", "FAKE"):
            label = "REAL"
        confidence = float(result.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        reasoning = result.get("reasoning", "")
        return label, confidence, reasoning
    except (json.JSONDecodeError, ValueError, KeyError):
        # Fallback if the model didn't return valid JSON
        return "REAL", 0.5, "Could not parse model response confidently."


def extract_article_from_url(url):
    """Try newspaper3k first, fall back to raw requests + BeautifulSoup."""
    try:
        from newspaper import Article

        article = Article(url)
        article.download()
        article.parse()
        if article.text and article.text.strip():
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
        if text:
            return {"success": True, "title": title, "text": text}
        return {"success": False, "error": "No readable text found on this page."}
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
    text = data.get("text", "").strip()
    if not text:
        return jsonify({
            "success": False,
            "error": "Please enter or paste some text first."
        }), 400

    label, confidence, reasoning = predict_news_llm(text)
    return jsonify({
        "success": True,
        "label": label,
        "confidence": confidence,
        "reasoning": reasoning,
    })


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

    label, confidence, reasoning = predict_news_llm(result["text"])
    return jsonify({
        "success": True,
        "label": label,
        "confidence": confidence,
        "reasoning": reasoning,
        "title": result.get("title", ""),
        "extracted_text": result["text"][:3000],
    })


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
