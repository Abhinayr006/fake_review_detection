# data_preprocessing.py

import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(text):
    """
    Cleans and tokenizes text data.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def create_tfidf_features(X_train, X_test):
    """
    Creates TF-IDF features for the training and testing data.
    """
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer

def load_and_preprocess_data(file_path):
    """
    Loads the IMDb dataset, preprocesses the text, and splits the data.
    """
    df = pd.read_csv(file_path)
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    df['deceptive'] = df['deceptive'].map({'deceptive': 1, 'truthful': 0})
    X = df['cleaned_text']
    y = df['deceptive']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test