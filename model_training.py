# model_training.py

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_model(X_train, y_train):
    """
    Trains a Logistic Regression model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and prints the performance metrics.
    """
    y_pred = model.predict(X_test)
    print("\n--- Model Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def save_model(model, file_path):
    """
    Saves the trained model to a file.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {file_path}")

if __name__ == '__main__':
    # This block is for testing this module independently.
    # It will be called from main.py in the final setup.
    print("This module is meant to be imported and used by main.py.")