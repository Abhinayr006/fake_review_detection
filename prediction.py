# prediction.py

import pickle
import numpy as np
from scipy.sparse import hstack
from data_preprocessing import preprocess_text

def load_assets(model_path, vectorizer_path):
    """
    Loads the saved model and TF-IDF vectorizer.
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def predict_fake_review(review_text, model, vectorizer, stars=3, useful=0, funny=0, cool=0,
                       user_review_count=5, user_average_stars=3.5, is_elite=0, review_length=None):
    """
    Makes a prediction on a single review text using hybrid features.
    """
    # Preprocess text
    cleaned_text = preprocess_text(review_text)

    # Create TF-IDF features
    vectorized_text = vectorizer.transform([cleaned_text])

    # Calculate review length if not provided
    if review_length is None:
        review_length = len(review_text.split())

    # Create behavioral features array
    behavioral_features = np.array([[
        stars, useful, funny, cool,
        user_review_count, user_average_stars, is_elite, review_length
    ]])

    # Combine features
    hybrid_features = hstack([vectorized_text, behavioral_features])

    # Make prediction
    prediction = model.predict(hybrid_features)

    if prediction[0] == 1:
        return "This is likely a FAKE review. ❌"
    else:
        return "This seems to be a GENUINE review. ✅"
