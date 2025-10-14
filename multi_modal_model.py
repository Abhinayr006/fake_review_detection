import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertTokenizer
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import pickle
import json
import time

class MultiModalFakeReviewDetector(nn.Module):
    """
    Multi-modal fake review detection model combining text and behavioral features.
    """

    def __init__(self, text_model_name='distilbert-base-uncased', num_behavioral_features=25, hidden_dim=512):
        super(MultiModalFakeReviewDetector, self).__init__()

        # Text encoder (DistilBERT)
        self.text_encoder = DistilBertModel.from_pretrained(text_model_name)
        text_hidden_size = self.text_encoder.config.hidden_size  # 768 for DistilBERT

        # Behavioral features processing
        self.behavioral_encoder = nn.Sequential(
            nn.Linear(num_behavioral_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # Multi-modal fusion
        combined_size = text_hidden_size + hidden_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_size, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 2)  # Binary classification
        )

        # Simplified attention mechanism (2 weights for text and behavioral)
        self.attention = nn.Sequential(
            nn.Linear(combined_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, input_ids, attention_mask, behavioral_features):
        # Encode text
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = text_outputs.last_hidden_state.mean(dim=1)  # Mean pooling

        # Encode behavioral features
        behavioral_embeddings = self.behavioral_encoder(behavioral_features)

        # Concatenate features for attention
        combined_features = torch.cat([text_embeddings, behavioral_embeddings], dim=1)

        # Apply attention-based fusion (2 weights)
        attention_weights = self.attention(combined_features)  # (batch, 2)

        # Weight the embeddings
        text_weighted = text_embeddings * attention_weights[:, 0:1]
        behavioral_weighted = behavioral_embeddings * attention_weights[:, 1:2]
        fused_features = torch.cat([text_weighted, behavioral_weighted], dim=1)

        # Apply fusion layer
        fused_features = self.fusion_layer(fused_features)

        # Final classification
        logits = self.classifier(fused_features)

        return logits, attention_weights

class MultiModalTrainer:
    """
    Trainer class for the multi-modal fake review detection model.
    """

    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.7, patience=3
        )

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        predictions = []
        true_labels = []

        for batch in train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            behavioral_features = batch['behavioral_features'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(input_ids, attention_mask, behavioral_features)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Collect predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(true_labels, predictions)

        return avg_loss, accuracy

    def evaluate(self, test_loader):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        all_logits = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                behavioral_features = batch['behavioral_features'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits, _ = self.model(input_ids, attention_mask, behavioral_features)
                loss = self.criterion(logits, labels)

                # Collect predictions
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
                total_loss += loss.item()

        # Calculate metrics
        avg_loss = total_loss / len(test_loader)
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions, output_dict=True)
        cm = confusion_matrix(true_labels, predictions).tolist()
        roc_auc = roc_auc_score(true_labels, [p[1] for p in all_logits])

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'predictions': predictions,
            'true_labels': true_labels,
            'logits': all_logits
        }

    def train(self, train_loader, test_loader, num_epochs=10, patience=5):
        """Full training loop with early stopping."""
        print("Starting multi-modal model training...")
        print(f"Training on {self.device}")

        best_accuracy = 0
        best_epoch = 0
        no_improve_count = 0

        training_history = []

        for epoch in range(num_epochs):
            start_time = time.time()

            # Train
            train_loss, train_accuracy = self.train_epoch(train_loader, epoch)

            # Evaluate
            eval_metrics = self.evaluate(test_loader)

            # Update scheduler
            self.scheduler.step(eval_metrics['accuracy'])

            # Track best model
            if eval_metrics['accuracy'] > best_accuracy:
                best_accuracy = eval_metrics['accuracy']
                best_epoch = epoch
                no_improve_count = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_multi_modal_model.pth')
            else:
                no_improve_count += 1

            # Record history
            epoch_time = time.time() - start_time
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': eval_metrics['loss'],
                'val_accuracy': eval_metrics['accuracy'],
                'val_roc_auc': eval_metrics['roc_auc'],
                'epoch_time': epoch_time
            })

            # Print progress
            print(f"Epoch {epoch + 1}/{num_epochs} ({epoch_time:.2f}s)")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            print(f"  Val Loss: {eval_metrics['loss']:.4f}, Val Acc: {eval_metrics['accuracy']:.4f}")
            print(f"  Val ROC-AUC: {eval_metrics['roc_auc']:.4f}")

            # Early stopping
            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(f"\nBest validation accuracy: {best_accuracy:.4f} at epoch {best_epoch + 1}")

        # Load best model
        self.model.load_state_dict(torch.load('best_multi_modal_model.pth'))

        return training_history, eval_metrics

def create_multi_modal_dataset(df, tokenizer, max_length=128, batch_size=16):
    """
    Create PyTorch dataset for multi-modal training.
    """
    from torch.utils.data import Dataset, DataLoader

    class MultiModalDataset(Dataset):
        def __init__(self, texts, behavioral_features, labels):
            self.texts = texts
            self.behavioral_features = behavioral_features
            self.labels = labels

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = str(self.texts.iloc[idx])
            features = self.behavioral_features.iloc[idx].values.astype(np.float32)
            label = self.labels.iloc[idx]

            # Tokenize text
            encoded = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )

            return {
                'input_ids': encoded['input_ids'].flatten(),
                'attention_mask': encoded['attention_mask'].flatten(),
                'behavioral_features': torch.tensor(features),
                'labels': torch.tensor(label, dtype=torch.long)
            }

    # Prepare features (exclude text and label columns)
    feature_columns = [col for col in df.columns
                      if col not in ['text', 'deceptive']
                      and df[col].dtype in ['int64', 'float64']]

    behavioral_features = df[feature_columns]
    labels = df['deceptive']

    # Create datasets
    dataset = MultiModalDataset(df['text'], behavioral_features, labels)

    # Split into train/test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train_multi_modal_model(df, model_save_path='multi_modal_model.pth',
                           tokenizer_path='multi_modal_tokenizer.pkl',
                           metrics_path='multi_modal_metrics.json'):
    """
    Train the multi-modal fake review detection model.
    """
    print("🚀 Starting Multi-Modal Fake Review Detection Training")
    print("=" * 80)

    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Determine number of behavioral features
    feature_columns = [col for col in df.columns
                      if col not in ['text', 'deceptive']
                      and df[col].dtype in ['int64', 'float64']]
    num_behavioral_features = len(feature_columns)

    print(f"Number of behavioral features: {num_behavioral_features}")
    print(f"Feature columns: {feature_columns}")

    # Create model
    model = MultiModalFakeReviewDetector(num_behavioral_features=num_behavioral_features)

    # Create datasets
    train_loader, test_loader = create_multi_modal_dataset(df, tokenizer)

    # Initialize trainer
    trainer = MultiModalTrainer(model)

    # Train model
    training_history, final_metrics = trainer.train(train_loader, test_loader, num_epochs=30)

    # Save model and tokenizer
    torch.save(model.state_dict(), model_save_path)
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)

    # Save metrics
    metrics = {
        'accuracy': final_metrics['accuracy'],
        'classification_report': final_metrics['classification_report'],
        'confusion_matrix': final_metrics['confusion_matrix'],
        'roc_auc': final_metrics['roc_auc'],
        'training_history': training_history,
        'num_behavioral_features': num_behavioral_features,
        'feature_columns': feature_columns
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    print("✅ Multi-modal model training completed!")
    print(f"📊 Final Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"📊 Final ROC-AUC: {final_metrics['roc_auc']:.4f}")
    print(f"💾 Model saved to: {model_save_path}")
    print(f"💾 Metrics saved to: {metrics_path}")

    return model, tokenizer, metrics

if __name__ == "__main__":
    # Example usage
    print("Multi-Modal Fake Review Detection Model")
    print("This module provides the multi-modal architecture for combining text and behavioral features.")
    print("\nTo train the model:")
    print("1. Load your enhanced dataset with engineered features")
    print("2. Call train_multi_modal_model(df)")
    print("3. Use the trained model for predictions")
