import nltk
from flask import Flask, request, jsonify, send_from_directory
import joblib
import os
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Check and download NLTK punkt data if not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = Flask(
    __name__,
    static_folder='frontend',  # Ensure the 'frontend' folder contains your static files (like index.html)
    static_url_path='/static'   # This means static files are accessed via /static/
)

# Load model and vectorizer
try:
    model = joblib.load(os.path.join(os.path.dirname(__file__), 'model.pkl'))
    vectorizer = joblib.load(os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl'))
except Exception as e:
    print(f"Error loading model/vectorizer: {e}")
    exit()

def preprocessed_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in filtered]
    return " ".join(lemmatized)

@app.route('/')
def index():
    # Serve 'index.html' from the frontend folder
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    # Serve static files (CSS, JS, images, etc.) from the frontend folder
    return send_from_directory(app.static_folder, path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'review' not in data:
        return jsonify({'error': 'No review text provided'}), 400
    try:
        processed = preprocessed_text(data['review'])
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]

        if prediction == 'pos':
            return jsonify({'sentiment': 'Positive 😊'})
        elif prediction == 'neg':
            return jsonify({'sentiment': 'Negative 😞'})
        else:
            return jsonify({'sentiment': f'Sentiment: {prediction}'})
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Error predicting sentiment.'}), 500

if __name__ == '__main__':
    # Binding the app to listen on port 10000 (for Render) and making it accessible externally
    app.run(debug=True, host='0.0.0.0', port=10000)
