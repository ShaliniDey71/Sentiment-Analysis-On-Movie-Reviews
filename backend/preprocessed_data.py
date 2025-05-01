import pandas as pd
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')

print(stopwords)
def load_data(filepath):
    return pd.read_csv(filepath)

def preprocessed_text(text):
    text=text.lower()
    text=text.translate(str.maketrans('','',string.punctuation))
    tokens=word_tokenize(text)
    stop_words=set(stopwords.words('english'))
    tokens=[word for word in tokens if word.lower() not in stop_words]
    lemmatizer=WordNetLemmatizer()
    tokens=[lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# print(preprocessed_text("HELLO, EVERYONE! WISHING YOU ALL THE VERY BEST. ITS 5:30AM IN THE MORNING."))

if __name__ == "__main__":
    train_df=load_data(r"D:\Sentiment-Analysis-On-Movie-Reviews\backend\data\imdb_train.csv")
    test_df=load_data(r"D:\Sentiment-Analysis-On-Movie-Reviews\backend\data\imdb_test.csv")
    review_column='review'
    train_df['processed_review']=train_df[review_column].apply(preprocessed_text)
    test_df['processed_review']=test_df[review_column].apply(preprocessed_text)

    train_df.to_csv(r"D:\Sentiment-Analysis-On-Movie-Reviews\backend\data\train_processed.csv", index=False)
    test_df.to_csv(r"D:\Sentiment-Analysis-On-Movie-Reviews\backend\data\test_processed.csv", index=False)

    print("Preprocessing complete")