This is a Flask-based web application for sentiment analysis using an LSTM model trained on the IMDB movie reviews dataset. The model utilizes GloVe word embeddings to enhance the understanding of textual sentiment.

Features
✅ Web interface for inputting text and getting sentiment predictions.
✅ Uses LSTM (Long Short-Term Memory) neural network for classification.
✅ Pre-trained GloVe embeddings for improved word representation.
✅ Built with Flask for the backend and Bootstrap for the frontend.

Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/Rishabh-Natarajan/Sentiment-Analysis-with-GloVe.git
cd Sentiment-Analysis-with-GloVe

2️⃣ Install Dependencies
Ensure you have Python installed. Then, install the required packages:
pip install -r requirements.txt

3️⃣ Run the Flask Application
python app.py
Then, open http://127.0.0.1:5000/ in your browser.

Training the Model (Optional)
To retrain the model, run:
python train_model.py

Dataset & Model
Dataset: IMDB Movie Reviews
Model: LSTM
Embeddings: GloVe (100d)

Tech Stack
Frontend: HTML, CSS, Bootstrap
Backend: Flask
Machine Learning: TensorFlow, Keras, NumPy, Pandas
Data Processing: Tokenization, Padding, Embedding
