"""
Pretrained Transformer & From-Scratch Transformer Classification Pipeline (LAB 08).

Implements:
  1. PretrainedSentimentClassifier:
       Input text -> Tokenizer -> Input IDs & Attention Mask -> Pretrained Transformer Encoder
       -> Contextual representations -> [CLS] representation -> Linear classification head
       -> Class logits -> Softmax -> Prediction & Confidence
     Supports both Mode A (Frozen encoder / Feature Extraction) and Mode B (Fine-tuning).
  2. FromScratchTransformerClassifier:
       Compact Transformer Encoder trained from random initialization using primitive PyTorch
       layers (consistent with LAB 07) for Experiment 1 & Experiment 3 baseline comparisons.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"


def load_tokenizer(model_name: str = DEFAULT_MODEL_NAME):
    """Load the HuggingFace pretrained tokenizer matching the chosen backbone."""
    return AutoTokenizer.from_pretrained(model_name)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Return total and trainable parameter counts for a PyTorch model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


class PretrainedSentimentClassifier(nn.Module):
    """
    BERT-style Sequence Classifier built on top of a Pretrained Transformer Encoder.

    Pipeline:
        input_ids, attention_mask
          -> Pretrained Transformer Encoder (outputs.last_hidden_state: [B, T, H])
          -> [CLS] token representation (index 0: [B, H])
          -> Dropout
          -> Task-specific Linear Classification Head ([B, num_labels])
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        num_labels: int = 2,
        dropout: float = 0.2,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.encoder.config.hidden_size)

        # Task-specific classification head (randomly initialized)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self.set_encoder_trainable(not freeze_encoder)

    def set_encoder_trainable(self, trainable: bool) -> None:
        """Toggle gradient updates on the pretrained encoder (Frozen vs Fine-tuning)."""
        for param in self.encoder.parameters():
            param.requires_grad = trainable

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return raw class logits of shape (batch_size, num_labels)."""
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None and "token_type_ids" in self.encoder.forward.__code__.co_varnames:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)
        # Extract [CLS] representation at sequence position 0: (B, H)
        cls_representation = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_representation))
        return logits

    @torch.no_grad()
    def predict_with_confidence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run forward pass and compute softmax probabilities and max confidence.

        Returns:
            preds: (B,) predicted class indices
            confidences: (B,) max softmax probability per sample
            probs: (B, num_labels) full softmax probability distribution
            logits: (B, num_labels) raw logits
        """
        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        probs = F.softmax(logits, dim=-1)
        confidences, preds = probs.max(dim=-1)
        return preds, confidences, probs, logits

    @torch.no_grad()
    def predict_texts(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 128,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """End-to-end inference helper from raw text strings to predictions & confidences."""
        self.eval()
        target_device = device or next(self.parameters()).device
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(target_device) for k, v in encoded.items()}
        preds, confidences, probs, _ = self.predict_with_confidence(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            token_type_ids=encoded.get("token_type_ids"),
        )
        return preds.cpu(), confidences.cpu(), probs.cpu()


# =============================================================================
# From-Scratch Transformer Baseline (consistent with LAB 07 primitive layers)
# =============================================================================


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to token embeddings."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class MultiHeadSelfAttention(nn.Module):
    """Multi-head scaled dot-product self-attention implemented from primitive layers."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.w_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.w_o(context)


class TransformerEncoderBlock(nn.Module):
    """Single Transformer Encoder block: MHA -> Add&Norm -> FFN -> Add&Norm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm1(x + self.dropout1(self.attn(x, mask=mask)))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x


class FromScratchTransformerClassifier(nn.Module):
    """
    Small Transformer Encoder trained from random initialization (Experiment 1 & 3 baseline).
    Uses masked mean pooling over non-padding tokens followed by a linear classification head.
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        max_len: int = 512,
        num_labels: int = 2,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.embed_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, n_heads, d_ff, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_idx).long()

        # Shape (B, 1, 1, T) for attention broadcasting
        attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        x = self.embed_dropout(self.pos_encoding(self.embedding(input_ids)))
        for layer in self.layers:
            x = layer(x, mask=attn_mask)

        # Masked mean pooling across valid tokens
        valid_mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
        pooled = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1e-6)
        return self.classifier(pooled)


if __name__ == "__main__":
    print("=== Smoke Test: Pretrained & From-Scratch Classifiers ===")
    tokenizer = load_tokenizer(DEFAULT_MODEL_NAME)
    sample_texts = [
        "This movie was wonderfully directed and genuinely moving.",
        "Total waste of time, predictable plot and wooden acting.",
    ]
    batch = tokenizer(
        sample_texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    print("Tokenizer keys:", list(batch.keys()))
    print("input_ids shape:", tuple(batch["input_ids"].shape))
    print("attention_mask shape:", tuple(batch["attention_mask"].shape))

    frozen_model = PretrainedSentimentClassifier(
        model_name=DEFAULT_MODEL_NAME, freeze_encoder=True
    )
    finetune_model = PretrainedSentimentClassifier(
        model_name=DEFAULT_MODEL_NAME, freeze_encoder=False
    )
    scratch_model = FromScratchTransformerClassifier(vocab_size=10000)

    print("Pretrained (Frozen) params:", count_parameters(frozen_model))
    print("Pretrained (Fine-tuning) params:", count_parameters(finetune_model))
    print("From-Scratch Transformer params:", count_parameters(scratch_model))

    finetune_model.eval()
    preds, confs, probs, logits = finetune_model.predict_with_confidence(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_type_ids=batch.get("token_type_ids"),
    )
    print("Pretrained logits shape:", tuple(logits.shape))
    print("Pretrained probs:", probs.tolist())
    print("Predictions:", preds.tolist(), "Confidences:", [round(c, 4) for c in confs.tolist()])

    scratch_ids = torch.randint(2, 10000, batch["input_ids"].shape)
    scratch_logits = scratch_model(
        input_ids=scratch_ids, attention_mask=batch["attention_mask"]
    )
    print("From-Scratch logits shape:", tuple(scratch_logits.shape))
