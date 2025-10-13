# prediction.py

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

def load_assets(model_path):
    """
    Loads the saved fine-tuned DistilBERT model and tokenizer.
    """
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer

def predict_fake_review(review_text, model, tokenizer, max_length=512):
    """
    Makes a prediction on a single review text using fine-tuned DistilBERT.

    Parameters:
    - review_text: The review text to analyze
    - model: The fine-tuned DistilBERT model
    - tokenizer: The DistilBERT tokenizer
    - max_length: Maximum sequence length

    Returns:
    - str: Prediction result with confidence
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Tokenize the input text
    inputs = tokenizer(
        review_text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=max_length
    ).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][prediction].item()

    if prediction == 1:
        return f"This is likely a FAKE review. ❌ (Confidence: {confidence:.2f})"
    else:
        return f"This seems to be a GENUINE review. ✅ (Confidence: {confidence:.2f})"

def predict_batch(reviews, model, tokenizer, max_length=512, batch_size=16):
    """
    Makes predictions on a batch of reviews.

    Parameters:
    - reviews: List of review texts
    - model: The fine-tuned DistilBERT model
    - tokenizer: The DistilBERT tokenizer
    - max_length: Maximum sequence length
    - batch_size: Batch size for processing

    Returns:
    - list: List of prediction results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    results = []
    for i in range(0, len(reviews), batch_size):
        batch_reviews = reviews[i:i+batch_size]

        inputs = tokenizer(
            batch_reviews,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=max_length
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()

        for j, pred in enumerate(predictions):
            confidence = probabilities[j][pred]
            if pred == 1:
                result = f"FAKE (Confidence: {confidence:.2f})"
            else:
                result = f"GENUINE (Confidence: {confidence:.2f})"
            results.append(result)

    return results
