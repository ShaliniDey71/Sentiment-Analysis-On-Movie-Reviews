from flask import Flask, request, jsonify, send_from_directory
import joblib
import os
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(
    __name__,
    static_folder='../frontend',
    static_url_path=''
)

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
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
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
    app.run(debug=True, port=5500)
