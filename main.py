# main.py

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, classification_report, make_scorer
import pickle

# Import our custom modules
from data_preprocessing import preprocess_text, create_tfidf_features
from data_preprocessing_yelp import load_yelp_data
from model_training import save_model

# --- Configuration Variables ---
# Set the model to use. 'hybrid' is our new ensemble approach.
MODEL_TO_USE = 'hybrid'
# Set the dataset to use. 'yelp' will use our new preprocessing script.
DATASET_TO_USE = 'yelp'

# --- Paths ---
# Use the correct, full absolute paths you provided
DATA_PATH_DECEPTIVE = 'data/deceptive-opinion.csv'
DATA_PATH_YELP_REVIEW = '/home/tondamanati-abhinay/Sem 7/SLP/Project/fake_review_detection/data/yelp_json/Yelp-JSON/Yelp JSON/yelp_dataset/yelp_academic_dataset_review.json'
DATA_PATH_YELP_USER = '/home/tondamanati-abhinay/Sem 7/SLP/Project/fake_review_detection/data/yelp_json/Yelp-JSON/Yelp JSON/yelp_dataset/yelp_academic_dataset_user.json'

HYBRID_MODEL_PATH = 'hybrid_model.pkl'
TFIDF_VECTORIZER_PATH = 'tfidf_vectorizer.pkl'

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and prints the performance metrics.
    """
    y_pred = model.predict(X_test)
    print("\n--- Model Performance on Test Set ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


def main():
    """
    The main function to run the entire fake review detection pipeline.
    """
    print(f"Starting the fake review detection pipeline with '{DATASET_TO_USE}' dataset and '{MODEL_TO_USE}' model...")

    # Step 1: Load and Preprocess Data based on selection
    if DATASET_TO_USE == 'deceptive_opinion':
        print("1. Loading and preprocessing data from the IMDb dataset...")
        df = pd.read_csv(DATA_PATH_DECEPTIVE)
        df['cleaned_text'] = df['text'].apply(preprocess_text)
        df['deceptive'] = df['deceptive'].map({'deceptive': 1, 'truthful': 0})
        X = df.drop(columns=['deceptive', 'text', 'hotel', 'polarity', 'source'])
        y = df['deceptive']
    
    elif DATASET_TO_USE == 'yelp':
        print("1. Loading and preprocessing data from the Yelp dataset...")
        df = load_yelp_data(DATA_PATH_YELP_REVIEW, DATA_PATH_YELP_USER)
        df['cleaned_text'] = df['text'].apply(preprocess_text)
        X = df.drop(columns=['deceptive', 'text'])
        y = df['deceptive']
    
    else:
        raise ValueError("Invalid dataset selected. Choose 'deceptive_opinion' or 'yelp'.")

    # Step 2: Split the data into training and testing sets
    print("2. Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Handle Hybrid Model Training
    if MODEL_TO_USE == 'hybrid':
        print("3. Training the hybrid model...")
        
        # Textual Feature Extraction (using TF-IDF)
        print("Creating TF-IDF features...")
        X_train_text, X_test_text, tfidf_vectorizer = create_tfidf_features(X_train['cleaned_text'], X_test['cleaned_text'])
        
        # Behavioral Features
        behavioral_features = [
            'stars', 'useful', 'funny', 'cool',
            'user_review_count', 'user_average_stars', 'is_elite', 'review_length'
        ]
        X_train_behavioral = X_train[behavioral_features]
        X_test_behavioral = X_test[behavioral_features]
        
        # Concatenate Features
        X_train_hybrid = hstack([X_train_text, X_train_behavioral.values])
        X_test_hybrid = hstack([X_test_text, X_test_behavioral.values])

        # Overfitting Mitigation: Cross-Validation
        print("Performing 5-fold cross-validation to assess model stability...")
        hybrid_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        scorer = make_scorer(accuracy_score)
        cv_scores = cross_val_score(hybrid_model, X_train_hybrid, y_train, cv=5, scoring=scorer, n_jobs=-1)
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Average CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train the final model on the entire training set
        print("Training final model on the full training set...")
        hybrid_model.fit(X_train_hybrid, y_train)

        # Step 4: Evaluate the model
        evaluate_model(hybrid_model, X_test_hybrid, y_test)
        
        # Step 5: Save the trained model and vectorizer
        print("5. Saving the model and vectorizer...")
        save_model(hybrid_model, HYBRID_MODEL_PATH)
        save_model(tfidf_vectorizer, TFIDF_VECTORIZER_PATH)
        print(f"Model and vectorizer saved to {HYBRID_MODEL_PATH} and {TFIDF_VECTORIZER_PATH}")

    else:
        print("Model not supported. Please select 'hybrid'.")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()