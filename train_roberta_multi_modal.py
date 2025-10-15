import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import json
import time
from roberta_multi_modal_model import RobertaMultiModalFakeReviewDetector
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ReviewDataset(Dataset):
    """Custom dataset for multi-modal review classification"""
    def __init__(self, texts, behavioral_features, labels, tokenizer, max_length=128):
        self.texts = texts
        self.behavioral_features = behavioral_features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Tokenize text using RoBERTa tokenizer
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # Get behavioral features
        behavioral = torch.tensor(self.behavioral_features[idx], dtype=torch.float32)

        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'behavioral_features': behavioral,
            'labels': label
        }

def train_roberta_model(num_reviews=100000, num_epochs=30, batch_size=16,
                       learning_rate=2e-5, patience=5, save_path='roberta_multi_modal_model.pth'):
    """
    Train RoBERTa-based multi-modal fake review detector
    """
    print("🚀 Starting RoBERTa Multi-Modal Training")
    print("=" * 50)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data using the same preprocessing as DistilBERT model
    print("Loading Yelp data...")
    from data_preprocessing_yelp import load_yelp_data

    # Use the same paths as the original training
    DATA_PATH_YELP_REVIEW = '/home/tondamanati-abhinay/Sem 7/SLP/Project/fake_review_detection/data/yelp_json/Yelp-JSON/Yelp JSON/yelp_dataset/yelp_academic_dataset_review.json'
    DATA_PATH_YELP_USER = '/home/tondamanati-abhinay/Sem 7/SLP/Project/fake_review_detection/data/yelp_json/Yelp-JSON/Yelp JSON/yelp_dataset/yelp_academic_dataset_user.json'

    df = load_yelp_data(DATA_PATH_YELP_REVIEW, DATA_PATH_YELP_USER, num_reviews=num_reviews)
    print(f"Data loaded: {len(df)} samples")
    print(f"Deceptive reviews: {df['deceptive'].sum()}")
    print(f"Genuine reviews: {len(df) - df['deceptive'].sum()}")

    # Prepare features
    text_data = df['text'].values
    behavioral_features = df[['user_review_count', 'user_average_stars', 'is_elite',
                             'review_length', 'user_friends', 'user_fans',
                             'user_compliment_count']].values
    labels = df['deceptive'].values

    # Split data
    train_texts, val_texts, train_behavioral, val_behavioral, train_labels, val_labels = train_test_split(
        text_data, behavioral_features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")

    # Initialize model
    num_behavioral_features = behavioral_features.shape[1]
    model = RobertaMultiModalFakeReviewDetector(num_behavioral_features=num_behavioral_features)
    model.to(device)

    # Initialize tokenizer
    tokenizer = model.tokenizer

    # Create datasets
    train_dataset = ReviewDataset(train_texts, train_behavioral, train_labels, tokenizer)
    val_dataset = ReviewDataset(val_texts, val_behavioral, val_labels, tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=3
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training tracking
    best_val_accuracy = 0.0
    patience_counter = 0
    training_history = []

    print("🏃 Starting training...")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels_epoch = []

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            behavioral = batch['behavioral_features'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs, _ = model(input_ids, attention_mask, behavioral)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_labels_epoch.extend(labels.cpu().numpy())

        train_accuracy = accuracy_score(train_labels_epoch, train_preds)
        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels_epoch = []
        val_probs = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                behavioral = batch['behavioral_features'].to(device)
                labels = batch['labels'].to(device)

                outputs, _ = model(input_ids, attention_mask, behavioral)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                probs = torch.softmax(outputs, dim=1)[:, 1]

                val_preds.extend(preds.cpu().numpy())
                val_labels_epoch.extend(labels.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())

        val_accuracy = accuracy_score(val_labels_epoch, val_preds)
        val_roc_auc = roc_auc_score(val_labels_epoch, val_probs)
        avg_val_loss = val_loss / len(val_loader)

        # Update learning rate
        scheduler.step(val_accuracy)

        # Track history
        epoch_info = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy,
            'val_roc_auc': val_roc_auc
        }
        training_history.append(epoch_info)

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{num_epochs} - {epoch_time:.1f}s")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val ROC-AUC: {val_roc_auc:.4f}")

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"💾 New best model saved! (Accuracy: {best_val_accuracy:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⏳ Patience: {patience_counter}/{patience}")

        # Early stopping
        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered after {epoch+1} epochs")
            break

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.1f} seconds")
    # Load best model for final evaluation
    print("🔄 Loading best model for final evaluation...")
    model.load_state_dict(torch.load(save_path))
    model.eval()

    # Final evaluation
    final_preds = []
    final_labels = []
    final_probs = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            behavioral = batch['behavioral_features'].to(device)
            labels = batch['labels'].to(device)

            outputs, _ = model(input_ids, attention_mask, behavioral)
            preds = torch.argmax(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            final_preds.extend(preds.cpu().numpy())
            final_labels.extend(labels.cpu().numpy())
            final_probs.extend(probs.cpu().numpy())

    # Calculate final metrics
    final_accuracy = accuracy_score(final_labels, final_preds)
    final_roc_auc = roc_auc_score(final_labels, final_probs)
    class_report = classification_report(final_labels, final_preds, output_dict=True)
    conf_matrix = confusion_matrix(final_labels, final_preds).tolist()

    # Save metrics
    metrics = {
        'accuracy': final_accuracy,
        'roc_auc': final_roc_auc,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'training_history': training_history,
        'best_epoch': max(training_history, key=lambda x: x['val_accuracy'])['epoch'],
        'total_training_time': total_time,
        'model_type': 'RoBERTa Multi-Modal',
        'dataset_size': len(df),
        'hyperparameters': {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'patience': patience,
            'num_epochs': num_epochs
        }
    }

    with open('roberta_multi_modal_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("📊 Final Results:")
    print(f"Accuracy: {final_accuracy:.4f}")
    print(f"ROC-AUC: {final_roc_auc:.4f}")
    print(f"Confusion Matrix: {conf_matrix}")
    print(f"Training completed in {total_time:.1f} seconds")

    return model, metrics

if __name__ == "__main__":
    # Train the model
    model, metrics = train_roberta_model()

    print("🎉 RoBERTa Multi-Modal Training Complete!")
    print("Model saved as: roberta_multi_modal_model.pth")
    print("Metrics saved as: roberta_multi_modal_metrics.json")
