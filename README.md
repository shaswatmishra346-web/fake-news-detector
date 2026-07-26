# 🔍 VERASCOPE — News Verification Desk

A machine learning powered fake news detector with a full standalone website. Paste article text or a URL, and VERASCOPE analyzes it and returns a verdict — REAL or FAKE — along with a confidence score.

**Live site:** https://verascope-two.vercel.app/

---

## Features

- **Paste text or paste a URL** — the app extracts article content directly from a link using `newspaper3k`, with a BeautifulSoup fallback for stubborn pages
- **Confidence score** shown alongside every verdict
- **Multi-domain training data** — combines the classic ISOT political/world news dataset with the BharatFakeNewsKosh dataset (India-focused, multilingual fact-checks translated to English), extending coverage beyond US politics
- **Clean Flask backend** serving both the API and the frontend from one app — no separate hosting needed
- **Custom-designed UI** — a forensic "case file" themed interface, no generic template look

---

## Tech Stack

| Layer | Tool |
|---|---|
| Backend | Flask + gunicorn |
| Model | scikit-learn (TF-IDF + Logistic Regression) |
| URL scraping | newspaper3k, BeautifulSoup, requests |
| Frontend | HTML / CSS / vanilla JS (single template, no framework) |
| Hosting | Render (or any Python-friendly host) |

---

## Project Structure

```
├── app.py                  # Flask backend — routes, model loading, prediction logic
├── templates/
│   └── index.html          # Frontend website
├── fake_news_model.pkl     # Trained classifier
├── tfidf_vectorizer.pkl    # Fitted TF-IDF vectorizer
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Running Locally

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000` in your browser.

---

## API Endpoints

### `POST /api/predict-text`
Check raw article text.

**Request body:**
```json
{ "text": "Full article text here..." }
```

**Response:**
```json
{ "success": true, "label": "REAL", "confidence": 0.87 }
```

### `POST /api/predict-url`
Fetch and check an article by URL.

**Request body:**
```json
{ "url": "https://example.com/news/some-article" }
```

**Response:**
```json
{
  "success": true,
  "label": "FAKE",
  "confidence": 0.79,
  "title": "Article title",
  "extracted_text": "First 3000 characters of the extracted article..."
}
```

---

## Model & Training Data

The current model is trained on a combined dataset of:
- **ISOT Fake News Dataset** — ~44,900 English-language political/world news articles (2016–2017, US-focused, sourced via Reuters and flagged unreliable sites)
- **BharatFakeNewsKosh** — ~26,200 fact-checked claims from Indian IFCN-verified fact-checkers (Alt News and others), covering politics, society, health, and more, across multiple Indian languages (translated to English for training)

To retrain the model with new or additional data, see `merge_and_retrain.py` (not included in production deployment — used only to regenerate the `.pkl` files).

---

## ⚠️ Disclaimer

This is a machine learning demo, **not a fact-checking authority**. Its accuracy depends heavily on how similar new input is to its training data. It may not generalize well to:
- Very recent events
- Topics or regions underrepresented in training data
- Deliberately adversarial or sarcastic writing

Always cross-check anything important with a trusted news source or a professional fact-checking organization.

---

## Roadmap / Ideas for Improvement

- [ ] Add more sports and finance-specific training data
- [ ] Expand crisis/conflict coverage
- [ ] Explore a transformer-based model for improved accuracy
- [ ] Add a browser extension for one-click checking
- [ ] Support additional languages natively (without relying on pre-translated text)

---

## License

_Add your preferred license here (MIT, Apache 2.0, etc.) or state "All rights reserved" if unsure._
