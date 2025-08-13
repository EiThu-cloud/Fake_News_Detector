import streamlit as st
import pickle

# =========================
# Load model and vectorizer
# =========================
@st.cache_resource
def load_model_and_vectorizer():
    with open("fake_news_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.title("📰 Fake News Detector")
st.write("Enter a news article or statement below and we'll predict whether it's **Real** or **Fake**.")

# Text input
news_text = st.text_area("📝 Paste your news text here:", height=200)

# Prediction button
if st.button("🔍 Check News"):
    if news_text.strip():
        # Vectorize input
        text_vectorized = vectorizer.transform([news_text])

        # Predict
        prediction = model.predict(text_vectorized)[0]

        # Display result
        if prediction == 1:
            st.error("🚨 This news might be **Fake**.")
        else:
            st.success("✅ This news seems **Real**.")
    else:
        st.warning("⚠️ Please enter some text before checking.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and scikit-learn")
