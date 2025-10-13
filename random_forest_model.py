import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import json

def load_and_preprocess_data(csv_path, use_text=True):
    """
    Load the provided dataset and preprocess for Random Forest.
    """
    df = pd.read_csv(csv_path)

    if use_text:
        # Text features: TF-IDF vectorization
        tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        text_features = tfidf.fit_transform(df['text']).toarray()
    else:
        tfidf = None
        text_features = np.empty((len(df), 0))

    # Numerical features
    numerical_features = df[['stars', 'useful', 'funny', 'cool', 'user_review_count',
                           'user_average_stars', 'is_elite', 'review_length', 'user_friends',
                           'user_fans', 'user_compliment_count', 'sentiment', 'num_sentences', 'avg_word_length']].values

    # Combine features
    X = np.hstack([text_features, numerical_features])
    y = df['deceptive'].values

    return X, y, tfidf

def train_random_forest(X, y):
    """
    Train Random Forest classifier using cross-validation.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    return rf

def evaluate_with_cross_validation(model, X, y):
    """
    Evaluate using 5-fold cross-validation.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accuracies = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    roc_aucs = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

    return {
        'mean_accuracy': accuracies.mean(),
        'std_accuracy': accuracies.std(),
        'mean_roc_auc': roc_aucs.mean(),
        'std_roc_auc': roc_aucs.std(),
        'accuracies': accuracies.tolist(),
        'roc_aucs': roc_aucs.tolist()
    }

def evaluate_on_test(model, X_test, y_test):
    """
    Evaluate the model on the test set.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }

def save_model_and_assets(model, tfidf, model_path='random_forest_model.pkl', tfidf_path='tfidf_vectorizer.pkl'):
    """
    Save the trained model and TF-IDF vectorizer.
    """
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    if tfidf:
        with open(tfidf_path, 'wb') as f:
            pickle.dump(tfidf, f)

def main():
    """
    Main function to train and evaluate Random Forest on the provided dataset using cross-validation.
    """
    csv_path = '/home/tondamanati-abhinay/Sem 7/SLP/Project/fake_review_detection/data/yelp_reviews.csv'
    use_text = False  # Set to False to use only numerical features

    print("Loading and preprocessing data...")
    X, y, tfidf = load_and_preprocess_data(csv_path, use_text=use_text)

    print(f"Dataset size: {len(X)} samples, {X.shape[1]} features")
    print(f"Using text features: {use_text}")

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    print("Training Random Forest model...")
    model = train_random_forest(X_train, y_train)
    model.fit(X_train, y_train)

    print("Evaluating with cross-validation on train set...")
    cv_metrics = evaluate_with_cross_validation(model, X_train, y_train)

    print("Cross-validation Results on Train:")
    print(f"Mean Accuracy: {cv_metrics['mean_accuracy']:.4f} ± {cv_metrics['std_accuracy']:.4f}")
    print(f"Mean ROC-AUC: {cv_metrics['mean_roc_auc']:.4f} ± {cv_metrics['std_roc_auc']:.4f}")

    print("Evaluating on test set...")
    test_metrics = evaluate_on_test(model, X_test, y_test)

    print("Test Results:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print("Classification Report:")
    print(json.dumps(test_metrics['classification_report'], indent=2))

    # Save model and assets
    save_model_and_assets(model, tfidf)
    print("Model saved.")

    # Save metrics
    metrics = {'cv_metrics': cv_metrics, 'test_metrics': test_metrics}
    with open('random_forest_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved to random_forest_metrics.json")

if __name__ == "__main__":
    main()
