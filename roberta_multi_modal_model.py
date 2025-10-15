import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
import torch.nn.functional as F

class RobertaMultiModalFakeReviewDetector(nn.Module):
    """
    Multi-modal fake review detector using RoBERTa for text encoding
    and neural networks for behavioral feature processing.
    """
    def __init__(self, num_behavioral_features=7, hidden_dim=512, dropout_rate=0.3):
        super(RobertaMultiModalFakeReviewDetector, self).__init__()

        # Text encoder using RoBERTa
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        # Freeze RoBERTa base layers to prevent overfitting
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Unfreeze the last layer for fine-tuning
        for param in self.text_encoder.encoder.layer[-1].parameters():
            param.requires_grad = True

        # Behavioral features encoder
        self.behavioral_encoder = nn.Sequential(
            nn.Linear(num_behavioral_features, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        # Attention mechanism for fusion
        self.attention_weights = nn.Linear(768 + 256, 2)  # RoBERTa hidden size is 768

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(768 + 256, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, input_ids, attention_mask=None, behavioral_features=None):
        # Text encoding
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = text_outputs.last_hidden_state[:, 0, :]  # CLS token

        # Behavioral encoding
        behavioral_embeddings = self.behavioral_encoder(behavioral_features)

        # Concatenate features
        combined = torch.cat([text_embeddings, behavioral_embeddings], dim=1)

        # Attention-based fusion
        attention_logits = self.attention_weights(combined)
        attention_weights = F.softmax(attention_logits, dim=1)

        # Apply attention weights
        weighted_text = attention_weights[:, 0].unsqueeze(1) * text_embeddings
        weighted_behavioral = attention_weights[:, 1].unsqueeze(1) * behavioral_embeddings
        fused_features = torch.cat([weighted_text, weighted_behavioral], dim=1)

        # Classification
        logits = self.classifier(fused_features)
        return logits, attention_weights

    def tokenize_text(self, texts, max_length=128):
        """Tokenize input texts for RoBERTa"""
        return self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
