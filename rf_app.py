import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# Load Random Forest model and metrics
@st.cache_resource
def load_rf_model():
    with open('random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_rf_metrics():
    with open('random_forest_metrics.json', 'r') as f:
        metrics = json.load(f)
    return metrics

def preprocess_input(stars, useful, funny, cool, user_review_count, user_average_stars,
                    is_elite, review_length, user_friends, user_fans, user_compliment_count,
                    sentiment, num_sentences, avg_word_length):
    """
    Preprocess input features for Random Forest prediction.
    """
    # Convert is_elite to int
    is_elite_int = 1 if is_elite else 0

    # Create feature array in the same order as training (14 features)
    features = np.array([[
        stars, useful, funny, cool, user_review_count, user_average_stars,
        is_elite_int, review_length, user_friends, user_fans, user_compliment_count,
        sentiment, num_sentences, avg_word_length
    ]])

    return features

def main():
    st.title("🕵️ Fake Review Detection - Random Forest Model")
    st.markdown("---")

    # Load model and metrics
    model = load_rf_model()
    metrics = load_rf_metrics()

    # Model Performance Section
    st.header("📊 Model Performance")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Test Metrics")
        st.metric("Accuracy", f"{metrics['test_metrics']['accuracy']:.1%}")
        st.metric("ROC-AUC", f"{metrics['test_metrics']['roc_auc']:.1%}")

    with col2:
        st.subheader("Cross-Validation")
        st.metric("CV Accuracy", f"{metrics['cv_metrics']['mean_accuracy']:.1%} ± {metrics['cv_metrics']['std_accuracy']:.1%}")
        st.metric("CV ROC-AUC", f"{metrics['cv_metrics']['mean_roc_auc']:.1%} ± {metrics['cv_metrics']['std_roc_auc']:.1%}")

    st.markdown("---")

    # Input Section
    st.header("🔍 Review Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Review Metadata")
        stars = st.slider("Star Rating", 1, 5, 4)
        useful = st.number_input("Useful Votes", 0, 1000, 5)
        funny = st.number_input("Funny Votes", 0, 1000, 2)
        cool = st.number_input("Cool Votes", 0, 1000, 3)
        review_length = st.number_input("Review Length (characters)", 10, 5000, 150)

    with col2:
        st.subheader("User Features")
        user_review_count = st.number_input("User Review Count", 0, 10000, 25)
        user_average_stars = st.slider("User Average Stars", 1.0, 5.0, 3.8, 0.1)
        is_elite = st.checkbox("Elite User")
        user_friends = st.number_input("User Friends Count", 0, 10000, 15)
        user_fans = st.number_input("User Fans Count", 0, 10000, 8)
        user_compliment_count = st.number_input("User Compliment Count", 0, 10000, 12)

    # Hidden derived features (used internally for prediction)
    sentiment = stars / 5.0  # Simple sentiment based on stars
    num_sentences = 3  # Default value
    avg_word_length = 4.8  # Default value

    # Prediction
    if st.button("🔮 Analyze Review", type="primary"):
        try:
            # Preprocess input
            features = preprocess_input(
                stars, useful, funny, cool, user_review_count, user_average_stars,
                is_elite, review_length, user_friends, user_fans, user_compliment_count,
                sentiment, num_sentences, avg_word_length
            )

            # Make prediction
            prediction_proba = model.predict_proba(features)[0]
            prediction = model.predict(features)[0]

            # Display results
            st.markdown("---")
            st.header("🎯 Prediction Results")

            if prediction == 1:
                st.error("🚨 **FAKE REVIEW DETECTED**")
                st.metric("Confidence", f"{prediction_proba[1]:.1%}")
            else:
                st.success("✅ **GENUINE REVIEW**")
                st.metric("Confidence", f"{prediction_proba[0]:.1%}")

            # Detailed probabilities
            st.subheader("Prediction Probabilities")
            prob_col1, prob_col2 = st.columns(2)
            with prob_col1:
                st.metric("Genuine Probability", f"{prediction_proba[0]:.1%}")
            with prob_col2:
                st.metric("Fake Probability", f"{prediction_proba[1]:.1%}")

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("*Random Forest model trained on 5,638 samples with 14 behavioral features*")

if __name__ == "__main__":
    main()
