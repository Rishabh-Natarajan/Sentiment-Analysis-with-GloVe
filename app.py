from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import pickle
import re
import string
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load pre-trained model
model = tf.keras.models.load_model("model/glove_sentiment_model.h5")

# Load tokenizer
with open("model/tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Define max sequence length
MAX_LEN = 100

# Preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

# Predict sentiment
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    review = data.get("review", "")
    cleaned_review = clean_text(review)
    
    sequence = tokenizer.texts_to_sequences([cleaned_review])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN)
    
    prediction = model.predict(padded_sequence)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    
    return jsonify({"sentiment": sentiment, "confidence": float(prediction)})

# Render homepage
@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
