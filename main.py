# main.py

import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import json

# Import our custom modules
from data_preprocessing_yelp import load_yelp_data
from distilbert_model_training import train_distilbert_model

# --- Configuration Variables ---
MODEL_TO_USE = 'distilbert'
DATASET_TO_USE = 'yelp'

# --- Paths ---
DATA_PATH_YELP_REVIEW = '/home/tondamanati-abhinay/Sem 7/SLP/Project/fake_review_detection/data/yelp_json/Yelp-JSON/Yelp JSON/yelp_dataset/yelp_academic_dataset_review.json'
DATA_PATH_YELP_USER = '/home/tondamanati-abhinay/Sem 7/SLP/Project/fake_review_detection/data/yelp_json/Yelp-JSON/Yelp JSON/yelp_dataset/yelp_academic_dataset_user.json'

DISTILBERT_MODEL_PATH = 'distilbert_model.pth'
DISTILBERT_TOKENIZER_PATH = 'distilbert_tokenizer.pkl'
METRICS_PATH = 'model_metrics.json'


def main():
    """
    The main function to run the DistilBERT fake review detection pipeline.
    """
    print(f"Starting the pipeline with '{DATASET_TO_USE}' dataset and '{MODEL_TO_USE}' model...")

    # Step 1: Load and Preprocess Data
    print("1. Loading and preprocessing data from the Yelp dataset...")
    df = load_yelp_data(DATA_PATH_YELP_REVIEW, DATA_PATH_YELP_USER)
    
    # Step 2: Split the data into training and testing sets
    print("2. Splitting data into training and testing sets...")
    X = df['text']
    y = df['deceptive']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Step 3: Train the DistilBERT model
    if MODEL_TO_USE == 'distilbert':
        print("3. Training the DistilBERT model...")
        train_distilbert_model(X_train, y_train, X_test, y_test, DISTILBERT_MODEL_PATH, DISTILBERT_TOKENIZER_PATH, METRICS_PATH)
    else:
        print("Model not supported. Please select 'distilbert'.")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()