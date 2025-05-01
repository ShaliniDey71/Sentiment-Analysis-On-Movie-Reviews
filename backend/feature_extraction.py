import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def load_processed_data(file_path):
    return pd.read_csv(file_path)

def extract_tfidf_features(train_corpus, test_corpus,max_features=5000):
    vectorizer=TfidfVectorizer(max_features=max_features)
    train_features=vectorizer.fit_transform(train_corpus)
    test_features=vectorizer.transform(test_corpus)
    return train_features, test_features, vectorizer

if __name__ == "__main__":
    train_df=load_processed_data(r"D:\Sentiment-Analysis-On-Movie-Reviews\backend\data\train_processed.csv")
    test_df=load_processed_data(r"D:\Sentiment-Analysis-On-Movie-Reviews\backend\data\test_processed.csv")
    review_column='processed_review'
    sentiment_column='sentiment'
    train_corpus=train_df[review_column]
    train_labels=train_df[sentiment_column]
    test_corpus=test_df[review_column]
    test_labels=test_df[sentiment_column]

    train_features, test_features, tfidf_vectorizer = extract_tfidf_features(train_corpus, test_corpus)

    print("Shape of training features:", train_features.shape)
    print("Shape of testing features:", test_features.shape)
    joblib.dump(tfidf_vectorizer, r'D:\Sentiment-Analysis-On-Movie-Reviews\backend\tfidf_vectorizer.pkl')
    print("\nTF-IDF feature extraction complete. Fitted vectorizer saved to backend/tfidf_vectorizer.pkl")