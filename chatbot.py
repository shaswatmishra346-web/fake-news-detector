"""
chatbot.py — AI chatbot module for the Verascope Fake News Detector.

WHAT THIS FILE DOES
--------------------
Adds a conversational assistant that explains an *already-made* ML prediction.
It never predicts, re-classifies, or overrides anything the model in app.py
has decided. It only receives the model's existing verdict + confidence and
talks about it.

WHY A SEPARATE BLUEPRINT
------------------------
Keeping this in its own file means app.py (your model-loading + prediction
logic) needs only two extra lines total: one import, one
`app.register_blueprint(...)` call. Nothing in your existing routes,
`predict_news()`, or `clean_text()` is touched.

REQUIRES
--------
- GROQ_API_KEY environment variable (free tier: https://console.groq.com)
- `requests` (already in your requirements.txt)
"""

import os
import requests
from flask import Blueprint, request, jsonify

chatbot_bp = Blueprint("chatbot", __name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"  # fast + generous free tier

SYSTEM_PROMPT = """You are an AI assistant for a Fake News Detection System called Verascope.

You do NOT predict whether news is fake or real. A separate Machine Learning
model has already made that prediction before you were ever called. Your job is to:
- Explain the ML model's existing prediction and confidence score
- Summarize the article when asked
- Educate the user about general signs of misinformation
- Suggest ways to independently verify the news (cross-checking sources,
  fact-checking organizations, etc.)
- Answer natural follow-up questions about the analyzed article

Hard rules — never break these:
- NEVER contradict, override, or silently re-run the ML model's verdict.
- NEVER invent a new or different confidence score. Only reference the exact
  value provided to you.
- If the prediction is FAKE, explain *possible* reasons a model might flag
  text like this (e.g. sensational phrasing, lack of attributed sources,
  patterns common in misinformation) — frame these as possibilities, not as
  proven facts about this specific article.
- If you are unsure about something, say so plainly instead of guessing.
- If the user asks something unrelated to this article or to fake news
  detection in general, politely explain that you specialize in fake news
  analysis and steer the conversation back.
"""


def _build_messages(user_message, news_text, prediction, confidence, history=None):
    """Assemble the message list sent to the Groq chat completions endpoint."""
    context = (
        f"ARTICLE TEXT (may be truncated):\n{news_text[:3000]}\n\n"
        f"ML MODEL PREDICTION: {prediction}\n"
        f"ML MODEL CONFIDENCE: {confidence:.2%}\n\n"
        "Treat the prediction and confidence above as fixed ground truth. "
        "Do not recompute, question, or restate them differently."
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + context}]

    # Optional prior turns from the frontend, capped so payloads stay small
    if history:
        for turn in history[-10:]:
            role = "user" if turn.get("role") == "user" else "assistant"
            content = turn.get("content", "")
            if content:
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})
    return messages


@chatbot_bp.route("/api/chat", methods=["POST"])
def api_chat():
    """
    POST /api/chat

    Request JSON:
        {
            "user_message": str,               # required
            "news_text": str,                  # required — the analyzed article
            "prediction": "REAL" | "FAKE",      # required — from the ML model
            "confidence": float,                # required — from the ML model
            "history": [                        # optional, for follow-ups
                {"role": "user" | "assistant", "content": str}, ...
            ]
        }

    Response JSON (success):
        { "success": true, "chatbot_reply": str }

    Response JSON (failure):
        { "success": false, "error": str }
    """
    if not GROQ_API_KEY:
        return jsonify({
            "success": False,
            "error": "Chatbot is not configured. Set the GROQ_API_KEY environment variable."
        }), 503

    data = request.get_json(force=True, silent=True) or {}

    user_message = (data.get("user_message") or "").strip()
    news_text = data.get("news_text") or ""
    prediction = data.get("prediction") or ""
    history = data.get("history") or []

    if not user_message:
        return jsonify({"success": False, "error": "Message cannot be empty."}), 400

    if not news_text or prediction not in ("REAL", "FAKE"):
        return jsonify({
            "success": False,
            "error": "Missing article context. Please run a prediction first, then chat about it."
        }), 400

    try:
        confidence = float(data.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0

    messages = _build_messages(user_message, news_text, prediction, confidence, history)

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "temperature": 0.4,
                "max_tokens": 500,
            },
            timeout=20,
        )
        resp.raise_for_status()
        payload = resp.json()
        reply = payload["choices"][0]["message"]["content"].strip()
        return jsonify({"success": True, "chatbot_reply": reply})

    except requests.exceptions.Timeout:
        return jsonify({
            "success": False,
            "error": "The chatbot took too long to respond. Please try again."
        }), 504
    except requests.exceptions.RequestException as e:
        return jsonify({"success": False, "error": f"Chatbot service error: {e}"}), 502
    except (KeyError, IndexError):
        return jsonify({"success": False, "error": "Unexpected response from the chatbot service."}), 502
