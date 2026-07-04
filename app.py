import streamlit as st
import joblib
import re

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ----------------------------
# Custom styling
# ----------------------------
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .title-text {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .subtitle-text {
        font-size: 1rem;
        text-align: center;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .result-real {
        background-color: #d4edda;
        border-left: 6px solid #28a745;
        padding: 1.2rem;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: 600;
        color: #155724;
        text-align: center;
        margin-top: 1rem;
    }
    .result-fake {
        background-color: #f8d7da;
        border-left: 6px solid #dc3545;
        padding: 1.2rem;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: 600;
        color: #721c24;
        text-align: center;
        margin-top: 1rem;
    }
    .confidence-text {
        font-size: 0.95rem;
        font-weight: 400;
        margin-top: 0.4rem;
    }
    .footer-note {
        text-align: center;
        color: #999;
        font-size: 0.8rem;
        margin-top: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Load model and vectorizer (cached so it only loads once)
# ----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ----------------------------
# Text cleaning (must match training preprocessing)
# ----------------------------
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
    confidence = prob[pred]
    return label, confidence

# ----------------------------
# Header
# ----------------------------
st.markdown('<p class="title-text">🔍 Fake News Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Paste a news article below to check if it looks REAL or FAKE</p>', unsafe_allow_html=True)

# ----------------------------
# Input area
# ----------------------------
user_input = st.text_area(
    "Article text",
    height=200,
    placeholder="Paste the full article text here (a few paragraphs works best)...",
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    check_button = st.button("Check Article", use_container_width=True, type="primary")

# ----------------------------
# Prediction and result display
# ----------------------------
if check_button:
    if not user_input.strip():
        st.warning("Please paste some article text first.")
    elif len(user_input.strip().split()) < 15:
        st.warning("This looks quite short — for best results, paste at least a few sentences of the article body.")
    else:
        with st.spinner("Analyzing..."):
            label, confidence = predict_news(user_input)

        if label == "REAL":
            st.markdown(f"""
                <div class="result-real">
                    ✅ This looks REAL
                    <div class="confidence-text">Confidence: {confidence:.1%}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-fake">
                    ⚠️ This looks FAKE
                    <div class="confidence-text">Confidence: {confidence:.1%}</div>
                </div>
            """, unsafe_allow_html=True)

# ----------------------------
# Footer / disclaimer
# ----------------------------
st.markdown("""
    <p class="footer-note">
    ⚠️ This tool is a machine learning demo trained on a limited dataset (primarily 2016–2017 US political news).<br>
    It may not generalize well to other topics, regions, or recent events. Not a substitute for fact-checking.
    </p>
""", unsafe_allow_html=True)
