# prediction.py

import pickle
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

def predict_fake_review(review_text, model, vectorizer):
    """
    Makes a prediction on a single review text.
    """
    cleaned_text = preprocess_text(review_text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    
    if prediction[0] == 1:
        return "This is likely a FAKE review. ❌"
    else:
        return "This seems to be a GENUINE review. ✅"