import streamlit as st
import requests

# Set page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
)

from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load your trained model and vectorizer files
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400

    # Vectorize input text
    X = vectorizer.transform([text])

    # Predict: assuming model outputs 0 for Fake, 1 for True
    pred = model.predict(X)[0]

    # Convert numeric prediction to label
    label = "Fake" if pred == 0 else "True"

    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(debug=True, port=5000)



# App Title
st.title("📰 Fake News Detector")
st.markdown("Check whether a news article is **Fake** or **True** in seconds.")

# User input
news_text = st.text_area("✏️ Enter the news text here:")

# Check button
if st.button("🔍 Check News"):
    if not news_text.strip():
        st.warning("⚠️ Please enter some text to check.")
    else:
        try:
            # Send request to backend API
            response = requests.post(
                "https://your-app.up.railway.app/predict", 
                json={"text": news_text}
        )


            if response.status_code == 200:
                result = response.json().get("prediction", "Unknown")
                if result.lower() == "fake":
                    st.error("🚨 This news appears to be **FAKE**.")
                elif result.lower() == "true":
                    st.success("✅ This news appears to be **TRUE**.")
                else:
                    st.info(f"ℹ️ Prediction: {result}")
            else:
                st.error("❌ API error. Please check the backend server.")

        except Exception as e:
            st.error(f"⚠️ Could not connect to backend: {e}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")