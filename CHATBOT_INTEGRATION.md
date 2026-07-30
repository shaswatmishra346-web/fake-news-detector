# Verascope Chatbot Integration

## What changed, file by file

### 1. `chatbot.py` — **new file**
Everything chatbot-related lives here: the Groq API call, the system prompt
that constrains the LLM's behavior, and the `/api/chat` route. Your model,
your vectorizer, and your prediction logic are never imported or touched by
this file.

### 2. `app.py` — **2 lines added, nothing else changed**
```python
from chatbot import chatbot_bp        # added
...
app.register_blueprint(chatbot_bp)    # added
```
Every other line — `clean_text()`, `predict_news()`, `extract_article_from_url()`,
`/api/predict-text`, `/api/predict-url` — is byte-for-byte identical to your
original file. I've included the full updated `app.py` so you can diff it
against your original and confirm this yourself.

### 3. `chat_widget_snippet.html` — **new file, to be pasted into `templates/index.html`**
I wasn't able to fetch your actual `templates/index.html` from GitHub (the
raw/blob path kept getting blocked), so rather than guess at your layout and
risk breaking it, I built the chat panel as a **fully self-contained block**
— its own `<style>`, HTML, and `<script>`, all scoped under `vs-chat-*`
IDs/classes so nothing in your existing CSS or JS can collide with it.

**To install it:** open your `templates/index.html`, paste the entire
contents of `chat_widget_snippet.html` right before the closing `</body>`
tag. That's it for the HTML side.

**One JS hook you need to add** (this is the only place I need help from your
existing code, since I can't see it): wherever your current frontend
receives a successful response from `/api/predict-text` or `/api/predict-url`
and displays the REAL/FAKE badge, add one line:

```javascript
window.verascopeSetChatContext(articleText, data.label, data.confidence);
```

- `articleText` — the text you sent to `/api/predict-text`, or
  `data.extracted_text` if you're in URL mode
- `data.label` and `data.confidence` — already exist in your API response

This tells the chat panel what article and verdict to talk about. Without
it, the widget will politely tell the user to "run a check above first."

### 4. `requirements.txt` — **1 line added**
Added `python-dotenv` (optional, for loading `.env` locally — see below).
`requests` was already there, so no other new dependency is needed for the
Groq integration itself.

### 5. `.env.example` — **new file**
Template for your API key.

---

## Setup instructions

1. **Get a free Groq API key** at https://console.groq.com/keys

2. **Set the environment variable.**

   Locally:
   ```bash
   cp .env.example .env
   # edit .env and paste your real key
   ```
   Then either:
   - Load it manually before running: `export GROQ_API_KEY=your_key_here` (Mac/Linux) or
     `set GROQ_API_KEY=your_key_here` (Windows), **or**
   - Add these two lines to the very top of `app.py` if you want `.env` loaded
     automatically:
     ```python
     from dotenv import load_dotenv
     load_dotenv()
     ```

   On a host like Render: set `GROQ_API_KEY` in the dashboard's environment
   variables section — never commit your real `.env` file to GitHub.

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Paste the widget** into `templates/index.html` (see step 3 above), and
   add the one-line JS hook to your existing predict handler.

5. **Run it:**
   ```bash
   python app.py
   ```
   Visit `http://localhost:5000`, run a prediction, then use the new chat
   panel below the result.

---

## How the guardrails work

The chatbot can't override your model because it's architecturally
incapable of it:
- It never receives the raw article without also receiving the model's
  `prediction` and `confidence` — and the system prompt instructs it to treat
  those as fixed ground truth.
- There's no code path where the chatbot's reply feeds back into `predict_news()`
  or changes what's displayed as the verdict — the two systems are only
  connected one-way (verdict → chatbot context), never the other way.
- The `/api/chat` endpoint requires an existing `prediction` before it will
  respond at all — if `news_text` or `prediction` is missing, it returns an
  error rather than guessing.

## Error handling included

- Missing `GROQ_API_KEY` → clean 503 response, not a crash
- Empty user message → 400 with a clear error
- No prediction context yet → 400, frontend shows a friendly nudge
- Groq API timeout → 504 with a "took too long" message
- Any other Groq API error → 502 with the underlying error message surfaced
- Malformed Groq response → 502 with a generic fallback message

## A note on what I couldn't verify

I don't have your actual `templates/index.html`, so I can't guarantee the
widget's dark "case file" theme will look seamless next to your existing
design — I matched the color palette I could see from the live site
(dark background, gold/amber accent, monospace font), but you may want to
tweak the CSS variables at the top of `chat_widget_snippet.html` to match
exactly. If you paste me your actual `index.html` content, I can fine-tune
this further or integrate it inline instead of as a bolt-on block.
