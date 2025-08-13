# app.py
# =========================
# Fake News Detector using Streamlit
# =========================

import streamlit as st
import joblib
import re

# -------------------------
# Page configuration (must be first Streamlit command)
# -------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# -------------------------
# Load model and vectorizer
# -------------------------
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# -------------------------
# Utility function to check if text is meaningful
# -------------------------
def is_meaningful(text):
    """
    Returns True if the text contains enough real words.
    """
    words = text.split()
    # Count words that are at least 2 letters long (letters only)
    real_words = [w for w in words if re.match("^[a-zA-Z]{2,}$", w)]
    return len(real_words) >= max(3, len(words)//2)  # at least half the words or 3

# -------------------------
# Streamlit UI
# -------------------------
st.title("📰 Fake News Detector")
st.write("Enter a news article or statement below and we'll predict whether it's **Real** or **Fake**.")

# Text input
news_text = st.text_area("📝 Paste your news text here:", height=200)

# Prediction button
if st.button("🔍 Check News"):
    if news_text.strip():
        # Minimum length check
        if len(news_text.strip()) < 20:
            st.warning("⚠️ Please enter a longer news article for a reliable prediction.")
        # Gibberish/meaningless text check
        elif not is_meaningful(news_text):
            st.error("🚨 Fake News (text seems meaningless)")
        else:
            # Vectorize and predict
            text_vectorized = vectorizer.transform([news_text])
            prediction = bool(model.predict(text_vectorized)[0])

            # Only two outputs
            if prediction:
                st.error("✅ Real News")
            else:
                st.success("🚨 Fake News") 
    else:
        st.warning("⚠️ Please enter some text before checking.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and scikit-learn")