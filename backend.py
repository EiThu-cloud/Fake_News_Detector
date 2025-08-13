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