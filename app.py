# app.py
import streamlit as st
import torch
from transformers import DistilBertTokenizer
import pickle
import pandas as pd
import json
import numpy as np
from multi_modal_model import MultiModalFakeReviewDetector

# Set the device for the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Paths ---
MODEL_PATH = 'multi_modal_model.pth'
TOKENIZER_PATH = 'multi_modal_tokenizer.pkl'
METRICS_PATH = 'multi_modal_metrics.json'

# --- Load the saved model, tokenizer, and metrics ---
@st.cache_resource
def load_assets():
    """Loads the multi-modal model, tokenizer, and metrics."""
    try:
        # Load tokenizer
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)

        # Load model
        num_behavioral_features = 7  # Based on training
        model = MultiModalFakeReviewDetector(num_behavioral_features=num_behavioral_features)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()

        # Load metrics
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)

        return model, tokenizer, metrics
    except FileNotFoundError:
        st.error("Model, tokenizer, or metrics files not found. Please run train_multi_modal.py first to train the model.")
        return None, None, None

model, tokenizer, metrics = load_assets()

# --- Prediction Functions ---
def predict_review(review_text, behavioral_features, model, tokenizer):
    """Predicts a single review using the multi-modal model."""
    # Encode text
    encoded_review = tokenizer.encode_plus(
        review_text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)

    # Prepare behavioral features
    behavioral_tensor = torch.tensor(behavioral_features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs, _ = model(input_ids, attention_mask=attention_mask, behavioral_features=behavioral_tensor)

    logits = outputs
    prediction = torch.argmax(logits, dim=1).item()
    confidence = torch.softmax(logits, dim=1)[0][prediction].item()
    return 'Deceptive ❌' if prediction == 1 else 'Genuine ✅', confidence

# --- Main App ---
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    font-size: 4em;
    font-weight: bold;
    color: #4CAF50;
    text-align: center;
    margin-bottom: 0.5em;
    text-shadow: 2px 2px 4px #000000;
}
.sub-header {
    font-size: 2em;
    font-weight: bold;
    color: #1a1a1a;
    text-align: center;
    margin-bottom: 1em;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2em;
    justify-content: center;
}
.stTabs [data-baseweb="tab-list"] button {
    background-color: #f0f2f6;
    border-radius: 8px 8px 0px 0px;
    font-size: 1.2em;
    font-weight: bold;
}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    background-color: #4CAF50;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Fake Review Detector 🕵️‍♀️</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by Multi-Modal Deep Learning Model</div>', unsafe_allow_html=True)

# Single page with all inputs and metrics
st.markdown("### Enter Review Details for Prediction")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Review Text")
    review_text = st.text_area(
        "Review Content:",
        height=150,
        help="Enter the review text to analyze",
        placeholder="This restaurant was amazing! The food was delicious and the service was excellent..."
    )

    st.markdown("#### Review Metadata")
    stars = st.slider("Star Rating", 1, 5, 4, help="Number of stars given in the review")
    useful = st.slider("Useful Votes", 0, 100, 5, help="Number of users who found this review useful")
    funny = st.slider("Funny Votes", 0, 50, 1, help="Number of users who found this review funny")
    cool = st.slider("Cool Votes", 0, 50, 2, help="Number of users who found this review cool")

with col2:
    st.markdown("#### User Profile Features")
    user_review_count = st.slider("User Total Reviews", 1, 1000, 25, help="Total number of reviews written by this user")
    user_average_stars = st.slider("User Average Stars", 1.0, 5.0, 3.8, step=0.1, help="Average star rating given by this user")
    is_elite = st.selectbox("Elite User?", [0, 1], index=0, help="Whether the user is an elite member (1) or not (0)")
    user_friends = st.slider("User Friends Count", 0, 500, 15, help="Number of friends this user has")
    user_fans = st.slider("User Fans Count", 0, 200, 3, help="Number of fans this user has")
    user_compliment_count = st.slider("User Compliments", 0, 100, 8, help="Total compliments received by this user")

# Prepare behavioral features in the correct order (only impactful features)
review_length = len(review_text.split()) if review_text else 0
behavioral_features = [
    user_review_count,
    user_average_stars,
    is_elite,
    review_length,
    user_friends,
    user_fans,
    user_compliment_count
]

# Prediction button
if st.button("🔍 Analyze Review", type="primary", use_container_width=True):
    if model and tokenizer and review_text.strip():
        with st.spinner("Analyzing review with multi-modal AI..."):
            prediction, confidence = predict_review(review_text, behavioral_features, model, tokenizer)

            # Display result
            st.markdown("---")
            col_result1, col_result2 = st.columns([2, 1])

            with col_result1:
                if 'Deceptive' in prediction:
                    st.error(f"### {prediction}")
                    st.markdown("**⚠️ This review appears to be potentially fake or manipulated.**")
                else:
                    st.success(f"### {prediction}")
                    st.markdown("**✅ This review appears to be genuine.**")

            with col_result2:
                st.metric("Confidence", f"{confidence:.1%}")

            # Show input summary
            with st.expander("📊 Review Analysis Summary"):
                st.markdown(f"**Review Length:** {review_length} words")
                st.markdown(f"**Star Rating:** {stars} ⭐")
                st.markdown(f"**User Review Count:** {user_review_count}")
                st.markdown(f"**User Average Rating:** {user_average_stars}")
                st.markdown(f"**Elite Status:** {'Yes' if is_elite else 'No'}")

    elif not review_text.strip():
        st.warning("⚠️ Please enter review text to analyze.")
    else:
        st.error("❌ Model not loaded. Please ensure the model files are available.")

# Model Metrics Section
st.markdown("---")
st.markdown("### 📈 Model Performance Metrics")

if metrics:
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    with col2:
        st.metric("ROC AUC", f"{metrics['roc_auc']:.1%}")
    with col3:
        precision = metrics['classification_report']['macro avg']['precision']
        st.metric("Precision", f"{precision:.1%}")
    with col4:
        recall = metrics['classification_report']['macro avg']['recall']
        st.metric("Recall", f"{recall:.1%}")

    # Detailed metrics in expander
    with st.expander("🔍 Detailed Metrics"):
        st.markdown("#### Classification Report")
        report_df = pd.DataFrame(metrics['classification_report']).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0))

        st.markdown("#### Confusion Matrix")
        cm_df = pd.DataFrame(
            metrics['confusion_matrix'],
            index=['Actual Genuine', 'Actual Deceptive'],
            columns=['Predicted Genuine', 'Predicted Deceptive']
        )
        st.dataframe(cm_df)

        st.markdown("#### Training History")
        history_df = pd.DataFrame(metrics['training_history'])
        st.line_chart(history_df[['val_accuracy', 'val_roc_auc']], use_container_width=True)

else:
    st.warning("Model metrics not found. Please run train_multi_modal.py first to train the model.")
