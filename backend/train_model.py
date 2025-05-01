import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer  

def load_data(file_path):
    return pd.read_csv(file_path)

def train_and_evaluate_model(train_features, train_labels, val_features, val_labels):
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(train_features, train_labels)
    val_predictions = model.predict(val_features)
    accuracy = accuracy_score(val_labels, val_predictions)
    report = classification_report(val_labels, val_predictions)
    cm = confusion_matrix(val_labels, val_predictions)
    return model, accuracy, report, cm

if __name__ == "__main__":
    train_df = load_data(r'D:\Sentiment-Analysis-On-Movie-Reviews\backend\data\train_processed.csv')
    test_df = load_data(r'D:\Sentiment-Analysis-On-Movie-Reviews\backend\data\test_processed.csv')

    review_column = 'processed_review'
    sentiment_column = 'sentiment'

    train_corpus = train_df[review_column]
    train_labels = train_df[sentiment_column]
    test_corpus = test_df[review_column]
    test_labels = test_df[sentiment_column]

    tfidf_vectorizer = joblib.load(r'D:\Sentiment-Analysis-On-Movie-Reviews\backend\tfidf_vectorizer.pkl')

    train_features = tfidf_vectorizer.transform(train_corpus)
    test_features = tfidf_vectorizer.transform(test_corpus)

    X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

    model, accuracy, report, cm = train_and_evaluate_model(X_train, y_train, X_val, y_val)

    print("Validation Accuracy:", accuracy)
    print("Validation Classification Report:\n", report)
    print("Validation Confusion Matrix:\n", cm)

    test_predictions = model.predict(test_features)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    test_report = classification_report(test_labels, test_predictions)
    test_cm = confusion_matrix(test_labels, test_predictions)

    print("\nTest Accuracy:", test_accuracy)
    print("Test Classification Report:\n", test_report)
    print("Test Confusion Matrix:\n", test_cm)

    joblib.dump(model, r'D:\Sentiment-Analysis-On-Movie-Reviews\backend\model.pkl')
    print("\nTrained Logistic Regression model saved to backend/model.pkl")