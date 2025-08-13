import streamlit as st
import requests

# Set page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
)

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