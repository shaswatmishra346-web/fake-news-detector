# 🔍 Fake News Detector

A machine learning web app that predicts whether a news article is **REAL** or **FAKE** based on its text content.

**🌐 Live Demo:** https://fakenewsdetector-4u.streamlit.app

---

## What This Project Does

Paste any news article's text into the app, and it will classify it as REAL or FAKE along with a confidence score, using a machine learning model trained on labeled news data.

This was built as a first hands-on machine learning project, covering the full pipeline: data collection, preprocessing, model training, evaluation, and deployment as a live web app.

---

## How It Works

1. **Data:** Trained on the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (~45,000 labeled news articles).
2. **Preprocessing:** Text is cleaned to remove source-specific artifacts (e.g., wire service tags) before training.
3. **Feature Extraction:** Article text is converted into numerical features using **TF-IDF** (Term Frequency–Inverse Document Frequency).
4. **Model:** A **Logistic Regression** classifier is trained on these features to distinguish fake from real articles.
5. **Interface:** Built with **Streamlit** and deployed on **Streamlit Community Cloud**.

---

## Results

- **Test Accuracy:** ~98% on the held-out test set from the training dataset.
- Precision, recall, and F1-score are balanced across both classes (~0.97–0.98).

---

## ⚠️ Known Limitations

This is an honest section, and an important one — no fake news detector is perfect, and understanding *why* a model fails is as valuable as its accuracy score.

- **Narrow training domain:** The training data is almost entirely U.S. political news from 2016–2017, sourced from Reuters (real) and flagged hoax/clickbait sites (fake). As a result, the model performs very well on that specific domain but **does not generalize well** to other topics (e.g., sports, technology, entertainment) or other countries/languages.
- **Style detection, not fact-checking:** The model is a **text-style classifier**, not a fact-checker. It learned to recognize *patterns of phrasing* common in its training data (e.g., wire-service style: `"CITY (Reuters) - ..."`) rather than verifying the truth of any claim. A true statement written in an unfamiliar style can be misclassified as fake, and a false statement written in a familiar style can be misclassified as real.
- **No real-time knowledge:** Since the model isn't connected to the internet or any live fact database, it has no knowledge of current events beyond its training data's time period (2016–2017).
- **Not a substitute for fact-checking:** This tool should be treated as a demonstration of NLP/ML techniques, not a reliable arbiter of truth.

---

## Tech Stack

- Python
- scikit-learn (TF-IDF, Logistic Regression)
- pandas
- Streamlit (web interface + deployment)

---

## Future Improvements

- Expand training data to include diverse topics (sports, tech, entertainment) and multiple languages/regions.
- Add more recent news data to reduce the "time gap" limitation.
- Experiment with transformer-based models (e.g., DistilBERT) for better generalization beyond surface-level style patterns.
- Extend the project to include a fake/phishing website detector and a fake social media account detector.

---

## Run It Locally

```bash
git clone https://github.com/YOUR-USERNAME/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
streamlit run app.py
```

---

## Disclaimer

This project is for educational purposes. It is not intended to be used as a definitive source of truth about any news article's authenticity.
