import streamlit as st
import torch
from transformers import RobertaTokenizer
import pickle
import pandas as pd
import json
import numpy as np
from roberta_multi_modal_model import RobertaMultiModalFakeReviewDetector

# Set the device for the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Paths ---
MODEL_PATH = 'roberta_multi_modal_model.pth'
TOKENIZER_PATH = 'roberta_tokenizer.pkl'  # We'll create this
METRICS_PATH = 'roberta_multi_modal_metrics.json'

# --- Load the saved model, tokenizer, and metrics ---
@st.cache_resource
def load_roberta_assets():
    """Loads the RoBERTa multi-modal model, tokenizer, and metrics."""
    try:
        # Load tokenizer
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        # Load model
        num_behavioral_features = 7  # Based on training
        model = RobertaMultiModalFakeReviewDetector(num_behavioral_features=num_behavioral_features)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()

        # Load metrics
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)

        return model, tokenizer, metrics
    except FileNotFoundError:
        st.error("RoBERTa model, tokenizer, or metrics files not found. Please run train_roberta_multi_modal.py first to train the model.")
        return None, None, None

model, tokenizer, metrics = load_roberta_assets()

# --- Prediction Functions ---
def predict_review_roberta(review_text, behavioral_features, model, tokenizer):
    """Predicts a single review using the RoBERTa multi-modal model."""
    # Encode text
    encoded_review = tokenizer(
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
        outputs, attention_weights = model(input_ids, attention_mask=attention_mask, behavioral_features=behavioral_tensor)

    logits = outputs
    prediction = torch.argmax(logits, dim=1).item()
    confidence = torch.softmax(logits, dim=1)[0][prediction].item()

    # Get attention weights for interpretability
    text_attention = attention_weights[0][0].item()
    behavioral_attention = attention_weights[0][1].item()

    return 'Deceptive ❌' if prediction == 1 else 'Genuine ✅', confidence, text_attention, behavioral_attention

# --- Main App ---
st.set_page_config(
    page_title="RoBERTa Fake Review Detector",
    page_icon="🔍"
)

st.title("🔍 RoBERTa Multi-Modal Fake Review Detection")
st.markdown("---")

# Sidebar with model info
st.sidebar.header("🤖 Model Information")
st.sidebar.markdown("**Architecture:** RoBERTa + Behavioral Features + Attention Fusion")
st.sidebar.markdown("**Training Data:** 54,690 balanced samples from 100k reviews")
st.sidebar.markdown("**Features:** Text analysis + 7 behavioral metrics")

if metrics:
    st.sidebar.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    st.sidebar.metric("ROC-AUC", f"{metrics['roc_auc']:.1%}")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Review Analysis")

    # Review text input
    review_text = st.text_area(
        "Enter the review text:",
        height=150,
        placeholder="Paste your review text here..."
    )

    # Behavioral features
    st.subheader("👤 User Behavioral Features")

    col_a, col_b = st.columns(2)

    with col_a:
        stars = st.slider("Star Rating", 1, 5, 4)
        useful = st.number_input("Useful Votes", 0, 1000, 5, key="useful")
        funny = st.number_input("Funny Votes", 0, 1000, 2, key="funny")
        cool = st.number_input("Cool Votes", 0, 1000, 3, key="cool")

    with col_b:
        user_review_count = st.number_input("User Review Count", 0, 10000, 25)
        user_average_stars = st.slider("User Average Stars", 1.0, 5.0, 3.8, 0.1)
        is_elite = st.checkbox("Elite User")
        review_length = st.number_input("Review Length (words)", 1, 1000, 150)

    # Hidden behavioral features (calculated)
    user_friends = st.number_input("User Friends Count", 0, 10000, 15, key="friends")
    user_fans = st.number_input("User Fans Count", 0, 10000, 8, key="fans")
    user_compliment_count = st.number_input("User Compliment Count", 0, 10000, 12, key="compliments")

    # Prediction button
    if st.button("🔮 Analyze Review", type="primary", use_container_width=True):
        if review_text.strip() and model:
            # Prepare behavioral features
            behavioral_features = [
                user_review_count, user_average_stars, 1 if is_elite else 0,
                review_length, user_friends, user_fans, user_compliment_count
            ]

            # Make prediction
            prediction, confidence, text_attn, behavioral_attn = predict_review_roberta(
                review_text, behavioral_features, model, tokenizer
            )

            # Display results
            st.success("✅ Analysis Complete!")

            # Main result
            if prediction == 'Deceptive ❌':
                st.error(f"### {prediction}")
                st.markdown("**⚠️ This review appears to be potentially fake or manipulated.**")
            else:
                st.success(f"### {prediction}")
                st.markdown("**✅ This review appears to be genuine.**")

            # Confidence and attention
            col_result1, col_result2 = st.columns(2)
            with col_result1:
                st.metric("Confidence", f"{confidence:.1%}")
            with col_result2:
                st.metric("Text Attention Weight", f"{text_attn:.3f}")

            # Attention breakdown
            st.subheader("🎯 Attention Analysis")
            attn_col1, attn_col2 = st.columns(2)
            with attn_col1:
                st.metric("Text Focus", f"{text_attn:.1%}")
            with attn_col2:
                st.metric("Behavioral Focus", f"{behavioral_attn:.1%}")

            # Show input summary
            with st.expander("📊 Review Analysis Summary"):
                st.markdown(f"**Review Length:** {review_length} words")
                st.markdown(f"**Star Rating:** {stars} ⭐")
                st.markdown(f"**User Review Count:** {user_review_count}")
                st.markdown(f"**User Average Rating:** {user_average_stars}")
                st.markdown(f"**Elite Status:** {'Yes' if is_elite else 'No'}")
                st.markdown(f"**Attention Distribution:** {text_attn:.1%} text, {behavioral_attn:.1%} behavioral")

        elif not review_text.strip():
            st.warning("⚠️ Please enter review text to analyze.")
        else:
            st.error("❌ RoBERTa model not loaded. Please ensure the model files are available.")

with col2:
    st.header("📈 Model Performance")

    if metrics:
        # Key metrics
        st.metric("Test Accuracy", f"{metrics['accuracy']:.1%}")
        st.metric("ROC-AUC Score", f"{metrics['roc_auc']:.1%}")

        # Classification report
        st.subheader("Classification Report")
        report_df = pd.DataFrame(metrics['classification_report']).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)

        # Confusion matrix
        st.subheader("Confusion Matrix")
        cm_df = pd.DataFrame(
            metrics['confusion_matrix'],
            index=['Actual Genuine', 'Actual Deceptive'],
            columns=['Predicted Genuine', 'Predicted Deceptive']
        )
        st.dataframe(cm_df, use_container_width=True)

        # Training history
        if 'training_history' in metrics:
            st.subheader("Training History")
            history_df = pd.DataFrame(metrics['training_history'])
            st.line_chart(history_df[['val_accuracy', 'val_roc_auc']], use_container_width=True)

    else:
        st.warning("RoBERTa model metrics not found. Please run train_roberta_multi_modal.py first.")

# Footer
st.markdown("---")
st.markdown("*RoBERTa Multi-Modal model trained on 54,690 samples with text analysis and 7 behavioral features*")
st.markdown("*Compared to DistilBERT: RoBERTa uses larger base model (125M vs 66M parameters) for potentially better performance*")
