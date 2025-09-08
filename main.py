# main.py

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from data_preprocessing import load_and_preprocess_data, create_tfidf_features
from model_training import train_model, evaluate_model, save_model
from prediction import load_assets, predict_fake_review

MODEL_PATH = 'logistic_regression_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'
DATA_PATH = 'data/deceptive-opinion.csv'

def main():
    """
    The main function to run the entire fake review detection pipeline.
    """
    print("Starting the fake review detection pipeline...")

    # Step 1: Load and Preprocess Data
    print("1. Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)
    
    # Step 2: Create Features
    print("2. Creating TF-IDF features...")
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = create_tfidf_features(X_train, X_test)
    
    # Step 3: Train and Save Model
    print("3. Training the model...")
    model = train_model(X_train_tfidf, y_train)
    save_model(model, MODEL_PATH)
    save_model(tfidf_vectorizer, VECTORIZER_PATH)
    
    # Step 4: Evaluate Model
    print("4. Evaluating the model...")
    evaluate_model(model, X_test_tfidf, y_test)
    
    # Step 5: Demonstrate Prediction on a new review
    print("\n5. Demonstrating a real-time prediction...")
    loaded_model, loaded_vectorizer = load_assets(MODEL_PATH, VECTORIZER_PATH)
    
    new_review_1 = "The hotel was absolutely amazing, a perfect stay. I couldn't have asked for anything more."
    new_review_2 = "Worst experience ever! The staff was rude and the room was dirty."
    
    print(f"\nReview: '{new_review_1}'")
    print(predict_fake_review(new_review_1, loaded_model, loaded_vectorizer))
    
    print(f"\nReview: '{new_review_2}'")
    print(predict_fake_review(new_review_2, loaded_model, loaded_vectorizer))
    
    print("\nPipeline complete.")

if __name__ == "__main__":
    main()