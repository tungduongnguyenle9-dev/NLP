"""
Full experiment suite for LAB 07 - Transformer Architecture.
Thực hiện toàn bộ các thực nghiệm theo yêu cầu đề bài:
  - Experiment 1: Positional Encoding & Permutation experiment ("dog bites man" vs "man bites dog")
  - Experiment 2: Number of Attention Heads (n_heads in [1, 2, 4, 8] with d_model=128)
  - Experiment 3: Attention Head Diversity (visualize 4 heads heatmap)
  - Experiment 4: Transformer vs RNN/LSTM Comparison
  - Experiment 5: Sequence Length Variation (max_length in [32, 64, 128, 256])
  - Experiment 6: Pooling Strategy (Mean pooling vs CLS-style pooling)
  - Error Analysis & Application: Evaluation, learning curves, confusion matrix, error examples.
"""

import sys
import os
import re
import time
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Đảm bảo import được module từ thư mục Lab07
sys.path.append("/Users/user/NLP/Lab07")
from positional_encoding import PositionalEncoding, get_sinusoidal_encoding_table
from transformer_block import TransformerBlock

# Cài đặt cố định các tham số môi trường
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(4)

LAB07_DIR = Path("/Users/user/NLP/Lab07")
DATASET_PATH = Path("/Users/user/NLP/IMDB Dataset.csv")

device = torch.device("cpu")
print(f"Device: {device}")

# =====================================================================
# 1. TIỀN XỬ LÝ VÀ LOAD DỮ LIỆU NHẤT QUÁN VỚI LAB 05 & LAB 06
# =====================================================================
print("\n[1/7] Đang tải dữ liệu IMDb Dataset...")
df = pd.read_csv(DATASET_PATH)

def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())

train_df, other = train_test_split(df, test_size=0.2, stratify=df.sentiment, random_state=SEED)
val_df, test_df = train_test_split(other, test_size=0.5, stratify=other.sentiment, random_state=SEED)

print(f"Dataset split: Train = {len(train_df)}, Val = {len(val_df)}, Test = {len(test_df)}")

# Xây dựng từ điển với kích thước 10,000 như các bài trước
VOCAB_SIZE = 10000
train_tokens = [tokenize(text) for text in train_df.review.fillna("")]
val_tokens = [tokenize(text) for text in val_df.review.fillna("")]
test_tokens = [tokenize(text) for text in test_df.review.fillna("")]

word_counts = Counter(word for words in train_tokens for word in words)
vocab = {word: i + 2 for i, (word, _) in enumerate(word_counts.most_common(VOCAB_SIZE - 2))}
# ID 0: <PAD>, ID 1: <UNK>

def encode_tokens(token_lists, max_len):
    x = np.zeros((len(token_lists), max_len), dtype=np.int64)
    for i, words in enumerate(token_lists):
        ids = [vocab.get(word, 1) for word in words[:max_len]]
        x[i, :len(ids)] = ids
    return torch.tensor(x, dtype=torch.long)

def make_loader(df_part, token_lists, max_len, batch_size=128, shuffle=False):
    x = encode_tokens(token_lists, max_len)
    y = torch.tensor((df_part.sentiment == "positive").to_numpy(), dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)

# Để các thực nghiệm diễn ra nhanh chóng, chuẩn xác và không bị treo máy CPU:
# Dùng tập huấn luyện đại diện 12,000 mẫu, validation 2,500 mẫu, test 2,500 mẫu
TRAIN_SUB_SIZE = 12000
VAL_SUB_SIZE = 2500
TEST_SUB_SIZE = 2500

sub_train_df = train_df.iloc[:TRAIN_SUB_SIZE].copy()
sub_train_tokens = train_tokens[:TRAIN_SUB_SIZE]

sub_val_df = val_df.iloc[:VAL_SUB_SIZE].copy()
sub_val_tokens = val_tokens[:VAL_SUB_SIZE]

sub_test_df = test_df.iloc[:TEST_SUB_SIZE].copy()
sub_test_tokens = test_tokens[:TEST_SUB_SIZE]

train_loader_100 = make_loader(sub_train_df, sub_train_tokens, max_len=100, batch_size=128, shuffle=True)
val_loader_100 = make_loader(sub_val_df, sub_val_tokens, max_len=100, batch_size=128, shuffle=False)
test_loader_100 = make_loader(sub_test_df, sub_test_tokens, max_len=100, batch_size=128, shuffle=False)
y_test_true = (sub_test_df.sentiment == "positive").to_numpy().astype(int)

# =====================================================================
# 2. EXPERIMENT 1: POSITIONAL ENCODING & PERMUTATION EXPERIMENT
# =====================================================================
print("\n[2/7] Chạy Experiment 1: Positional Encoding & Permutation...")

# 2.1 Vẽ hình positional_encoding.png
pe_table = get_sinusoidal_encoding_table(n_position=100, d_model=128)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
im = axes[0].imshow(pe_table, cmap="viridis", aspect="auto")
axes[0].set_title("Sinusoidal Positional Encoding Heatmap (100 positions x 128 dim)", fontsize=12)
axes[0].set_xlabel("Embedding Dimension (i)")
axes[0].set_ylabel("Position in Sequence (pos)")
fig.colorbar(im, ax=axes[0], label="PE Value")

# Đồ thị sóng sin/cos tại một số chiều đại diện
positions = np.arange(100)
axes[1].plot(positions, pe_table[:, 0], label="dim 0 (sin, high freq)", color="blue")
axes[1].plot(positions, pe_table[:, 1], label="dim 1 (cos, high freq)", color="cyan")
axes[1].plot(positions, pe_table[:, 16], label="dim 16 (mid freq)", color="green")
axes[1].plot(positions, pe_table[:, 64], label="dim 64 (low freq)", color="red")
axes[1].set_title("Sóng Positional Encoding tại các chiều khác nhau", fontsize=12)
axes[1].set_xlabel("Position (pos)")
axes[1].set_ylabel("PE Value")
axes[1].legend(loc="upper right")
axes[1].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
fig.savefig(LAB07_DIR / "positional_encoding.png", dpi=300)
plt.close(fig)
print("-> Đã lưu: positional_encoding.png")

# 2.2 Thực nghiệm hoán vị: S1 = 'dog bites man', S2 = 'man bites dog'
def run_permutation_test():
    vocab_small = {"<PAD>": 0, "dog": 1, "bites": 2, "man": 3}
    d_model = 128
    emb = nn.Embedding(len(vocab_small), d_model)
    pe_layer = PositionalEncoding(d_model=d_model, max_len=10, dropout=0.0)
    block_no_pe = TransformerBlock(d_model=d_model, n_heads=4, d_ff=256, dropout=0.0)
    block_with_pe = TransformerBlock(d_model=d_model, n_heads=4, d_ff=256, dropout=0.0)
    block_with_pe.load_state_dict(block_no_pe.state_dict())

    s1_ids = torch.tensor([[vocab_small["dog"], vocab_small["bites"], vocab_small["man"]]])
    s2_ids = torch.tensor([[vocab_small["man"], vocab_small["bites"], vocab_small["dog"]]])

    emb_s1 = emb(s1_ids)
    emb_s2 = emb(s2_ids)

    # A. Không có Positional Encoding
    out_no_pe_s1 = block_no_pe(emb_s1)
    out_no_pe_s2 = block_no_pe(emb_s2)
    # Mean pooling
    pool_no_pe_s1 = out_no_pe_s1.mean(dim=1)
    pool_no_pe_s2 = out_no_pe_s2.mean(dim=1)
    cos_no_pe = F.cosine_similarity(pool_no_pe_s1, pool_no_pe_s2).item()
    diff_no_pe = torch.norm(pool_no_pe_s1 - pool_no_pe_s2).item()

    # B. Có Positional Encoding
    emb_pe_s1 = pe_layer(emb_s1)
    emb_pe_s2 = pe_layer(emb_s2)
    out_pe_s1 = block_with_pe(emb_pe_s1)
    out_pe_s2 = block_with_pe(emb_pe_s2)
    pool_pe_s1 = out_pe_s1.mean(dim=1)
    pool_pe_s2 = out_pe_s2.mean(dim=1)
    cos_pe = F.cosine_similarity(pool_pe_s1, pool_pe_s2).item()
    diff_pe = torch.norm(pool_pe_s1 - pool_pe_s2).item()

    print(f"Permutation test S1='dog bites man' vs S2='man bites dog':")
    print(f"  - Không có PE: Cosine Similarity = {cos_no_pe:.6f}, L2 Norm Difference = {diff_no_pe:.6f}")
    print(f"  - Có PE:       Cosine Similarity = {cos_pe:.6f}, L2 Norm Difference = {diff_pe:.6f}")
    return cos_no_pe, cos_pe

cos_no_pe, cos_pe = run_permutation_test()

# =====================================================================
# MODEL ARCHITECTURES
# =====================================================================
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=256, max_len=500, dropout=0.1, pooling="mean"):
        super().__init__()
        self.d_model = d_model
        self.pooling = pooling
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.scale = math.sqrt(d_model)
        self.pe = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)
        self.block = TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
        self.classifier = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        # x: (B, T)
        mask = (x != 0).unsqueeze(1).unsqueeze(2) # (B, 1, 1, T) for attention
        emb = self.embedding(x) * self.scale
        emb = self.pe(emb)

        if return_attention:
            out, attn_weights = self.block(emb, mask=mask, return_attention=True)
        else:
            out = self.block(emb, mask=mask, return_attention=False)
            attn_weights = None

        if self.pooling == "mean":
            # Masked mean pooling để không bị ảnh hưởng bởi pad token
            pad_mask = (x != 0).unsqueeze(-1).float() # (B, T, 1)
            sum_embeddings = (out * pad_mask).sum(dim=1)
            sum_mask = pad_mask.sum(dim=1).clamp(min=1e-9)
            pooled = sum_embeddings / sum_mask
        elif self.pooling == "cls":
            # Lấy vector tại vị trí đầu tiên
            pooled = out[:, 0, :]
        else:
            pooled = out.mean(dim=1)

        logits = self.classifier(self.dropout(pooled)).squeeze(-1)
        if return_attention:
            return logits, attn_weights
        return logits


class LSTMClassifier(nn.Module):
    """LSTM model baseline từ LAB 05 để so sánh công bằng."""
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        out, (h_n, c_n) = self.lstm(emb)
        # Lấy hidden state cuối cùng
        last_hidden = h_n[-1]
        logits = self.fc(self.dropout(last_hidden)).squeeze(-1)
        return logits


def train_and_eval(model, train_loader, val_loader, test_loader, epochs=3, lr=0.001):
    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    val_losses = []

    start_time = time.perf_counter()
    for ep in range(epochs):
        model.train()
        total_train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(by)
        train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                logits = model(bx)
                loss = criterion(logits, by)
                total_val_loss += loss.item() * len(by)
        val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

    total_time = time.perf_counter() - start_time

    # Đánh giá trên test set
    model.eval()
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for bx, _ in test_loader:
            logits = model(bx)
            probs = torch.sigmoid(logits).numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    y_true = np.array([by.item() for _, by_batch in test_loader for by in by_batch], dtype=int)

    acc = accuracy_score(y_true, all_preds)
    prec = precision_score(y_true, all_preds, zero_division=0)
    rec = recall_score(y_true, all_preds, zero_division=0)
    f1 = f1_score(y_true, all_preds, zero_division=0)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    metrics = {
        "params": params,
        "time": total_time,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "probs": all_probs,
        "preds": all_preds,
        "y_true": y_true
    }
    return metrics

# =====================================================================
# 3. EXPERIMENT 2: NUMBER OF ATTENTION HEADS
# =====================================================================
print("\n[3/7] Chạy Experiment 2: Số attention heads (n_heads in [1, 2, 4, 8])...")
exp2_results = []
models_by_head = {}

for h in [1, 2, 4, 8]:
    d_k = 128 // h
    model = TransformerClassifier(vocab_size=VOCAB_SIZE, d_model=128, n_heads=h, d_ff=256, dropout=0.1, pooling="mean")
    m = train_and_eval(model, train_loader_100, val_loader_100, test_loader_100, epochs=3, lr=0.001)
    models_by_head[h] = (model, m)
    exp2_results.append({
        "Heads": h,
        "d_k": d_k,
        "Parameters": m["params"],
        "Training time": round(m["time"], 2),
        "Accuracy": round(m["acc"], 4),
        "F1": round(m["f1"], 4)
    })
    print(f"  Heads={h:1d} | d_k={d_k:3d} | Params={m['params']} | Time={m['time']:.2f}s | Acc={m['acc']:.4f} | F1={m['f1']:.4f}")

df_exp2 = pd.DataFrame(exp2_results)

# =====================================================================
# 4. EXPERIMENT 3: ATTENTION HEAD DIVERSITY
# =====================================================================
print("\n[4/7] Chạy Experiment 3: Attention Head Diversity & Visualization...")
best_model_4h, best_metrics_4h = models_by_head[4]

sample_text = "this movie was truly fantastic with brilliant acting and great directing"
sample_tokens = tokenize(sample_text)
sample_ids = torch.tensor([[vocab.get(w, 1) for w in sample_tokens]], dtype=torch.long)

best_model_4h.eval()
with torch.no_grad():
    _, attn_weights = best_model_4h(sample_ids, return_attention=True)
    # attn_weights: (1, n_heads=4, T, T)
    attn_weights = attn_weights.squeeze(0).numpy() # (4, T, T)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()
n_words = len(sample_tokens)

for h_idx in range(4):
    ax = axes[h_idx]
    mat = attn_weights[h_idx, :n_words, :n_words]
    sns.heatmap(mat, ax=ax, cmap="Blues", annot=True, fmt=".2f",
                xticklabels=sample_tokens, yticklabels=sample_tokens, cbar=True)
    ax.set_title(f"Attention Head {h_idx + 1}", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)

plt.suptitle("Attention Head Diversity across 4 Heads (Sample Sentence)", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(LAB07_DIR / "attention_heads.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("-> Đã lưu: attention_heads.png")

# =====================================================================
# 5. EXPERIMENT 4: TRANSFORMER VS RNN/LSTM
# =====================================================================
print("\n[5/7] Chạy Experiment 4: Transformer vs LSTM Comparison...")
lstm_model = LSTMClassifier(vocab_size=VOCAB_SIZE, embed_dim=128, hidden_dim=64, dropout=0.1)
lstm_metrics = train_and_eval(lstm_model, train_loader_100, val_loader_100, test_loader_100, epochs=3, lr=0.001)

transformer_metrics = best_metrics_4h

exp4_results = [
    {
        "Model": "LSTM",
        "Parameters": lstm_metrics["params"],
        "Training time": round(lstm_metrics["time"], 2),
        "Accuracy": round(lstm_metrics["acc"], 4),
        "Precision": round(lstm_metrics["prec"], 4),
        "Recall": round(lstm_metrics["rec"], 4),
        "F1": round(lstm_metrics["f1"], 4)
    },
    {
        "Model": "Transformer",
        "Parameters": transformer_metrics["params"],
        "Training time": round(transformer_metrics["time"], 2),
        "Accuracy": round(transformer_metrics["acc"], 4),
        "Precision": round(transformer_metrics["prec"], 4),
        "Recall": round(transformer_metrics["rec"], 4),
        "F1": round(transformer_metrics["f1"], 4)
    }
]
df_exp4 = pd.DataFrame(exp4_results)
print(df_exp4.to_string(index=False))

# =====================================================================
# 6. EXPERIMENT 5: SEQUENCE LENGTH VARIATION
# =====================================================================
print("\n[6/7] Chạy Experiment 5: Sequence Length Variation (32, 64, 128, 256)...")
exp5_results = []
all_lengths = [len(words) for words in sub_train_tokens]

for L in [32, 64, 128, 256]:
    retained_pct = 100.0 * np.mean([min(len(words), L) / max(len(words), 1) for words in sub_train_tokens])
    attn_mem = L * L

    tl = make_loader(sub_train_df, sub_train_tokens, max_len=L, batch_size=128, shuffle=True)
    vl = make_loader(sub_val_df, sub_val_tokens, max_len=L, batch_size=128, shuffle=False)
    tel = make_loader(sub_test_df, sub_test_tokens, max_len=L, batch_size=128, shuffle=False)

    model_L = TransformerClassifier(vocab_size=VOCAB_SIZE, d_model=128, n_heads=4, d_ff=256, max_len=300, dropout=0.1, pooling="mean")
    m_L = train_and_eval(model_L, tl, vl, tel, epochs=2, lr=0.001)

    exp5_results.append({
        "Length": L,
        "Tokens retained": f"{retained_pct:.2f}%",
        "Attention memory": attn_mem,
        "Training time": round(m_L["time"], 2),
        "F1": round(m_L["f1"], 4)
    })
    print(f"  Length={L:3d} | Retained={retained_pct:.1f}% | Attn Elements={attn_mem:5d} | Time={m_L['time']:.2f}s | F1={m_L['f1']:.4f}")

df_exp5 = pd.DataFrame(exp5_results)

# =====================================================================
# 7. EXPERIMENT 6: POOLING STRATEGY (MEAN VS CLS-STYLE)
# =====================================================================
print("\n[7/7] Chạy Experiment 6: Pooling Strategy...")
cls_model = TransformerClassifier(vocab_size=VOCAB_SIZE, d_model=128, n_heads=4, d_ff=256, dropout=0.1, pooling="cls")
cls_metrics = train_and_eval(cls_model, train_loader_100, val_loader_100, test_loader_100, epochs=3, lr=0.001)

exp6_results = [
    {
        "Pooling": "Mean",
        "Accuracy": round(best_metrics_4h["acc"], 4),
        "F1": round(best_metrics_4h["f1"], 4),
        "Training time": round(best_metrics_4h["time"], 2)
    },
    {
        "Pooling": "CLS-style",
        "Accuracy": round(cls_metrics["acc"], 4),
        "F1": round(cls_metrics["f1"], 4),
        "Training time": round(cls_metrics["time"], 2)
    }
]
df_exp6 = pd.DataFrame(exp6_results)
print(df_exp6.to_string(index=False))

# =====================================================================
# LƯU KẾT QUẢ VÀO results.csv
# =====================================================================
with open(LAB07_DIR / "results.csv", "w", encoding="utf-8") as f:
    f.write("# Experiment 2: Number of attention heads\n")
    df_exp2.to_csv(f, index=False)
    f.write("\n# Experiment 4: Transformer vs LSTM\n")
    df_exp4.to_csv(f, index=False)
    f.write("\n# Experiment 5: Sequence length\n")
    df_exp5.to_csv(f, index=False)
    f.write("\n# Experiment 6: Pooling strategy\n")
    df_exp6.to_csv(f, index=False)
print("-> Đã lưu toàn bộ số liệu vào results.csv")

# =====================================================================
# VẼ ĐỒ THỊ LEARNING CURVES VÀ CONFUSION MATRIX
# =====================================================================
# Learning curves
fig, ax = plt.subplots(figsize=(8, 5))
epochs_range = range(1, len(transformer_metrics["train_losses"]) + 1)
ax.plot(epochs_range, transformer_metrics["train_losses"], marker='o', label="Training Loss", color="#1f77b4")
ax.plot(epochs_range, transformer_metrics["val_losses"], marker='s', label="Validation Loss", color="#ff7f0e")
ax.set_title("Transformer Encoder Learning Curves (3 Epochs)", fontsize=13, fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Binary Cross Entropy Loss")
ax.set_xticks(epochs_range)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend()
plt.tight_layout()
fig.savefig(LAB07_DIR / "learning_curves.png", dpi=300)
plt.close(fig)
print("-> Đã lưu: learning_curves.png")

# Confusion Matrix
cm = confusion_matrix(transformer_metrics["y_true"], transformer_metrics["preds"])
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
disp.plot(cmap="Blues", ax=ax, colorbar=False)
ax.set_title(f"Confusion Matrix (Test Acc: {transformer_metrics['acc']:.4f})", fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(LAB07_DIR / "confusion_matrix.png", dpi=300)
plt.close(fig)
print("-> Đã lưu: confusion_matrix.png")

# =====================================================================
# TÌM CÁC VÍ DỤ ERROR ANALYSIS (3 ĐÚNG, 3 SAI)
# =====================================================================
print("\nTrích xuất các ca phân tích lỗi cho error_analysis.md...")
test_probs = transformer_metrics["probs"]
test_preds = transformer_metrics["preds"]
y_test = transformer_metrics["y_true"]

correct_indices = np.where(test_preds == y_test)[0]
incorrect_indices = np.where(test_preds != y_test)[0]

print(f"Tổng mẫu đúng: {len(correct_indices)}, Tổng mẫu sai: {len(incorrect_indices)}")

# Lưu chi tiết 3 mẫu đúng và 3 mẫu sai ra file JSON/scratch để điền vào error_analysis.md
sample_errors = {
    "correct": [],
    "incorrect": []
}

# Chọn các mẫu đúng có đặc trưng rõ ràng
for idx in correct_indices[:15]:
    txt = sub_test_df.iloc[idx]["review"]
    prob = float(test_probs[idx])
    lbl = "positive" if y_test[idx] == 1 else "negative"
    pred_lbl = "positive" if test_preds[idx] == 1 else "negative"
    sample_errors["correct"].append({
        "index": int(sub_test_df.index[idx]),
        "actual": lbl,
        "predicted": pred_lbl,
        "prob": prob,
        "text": txt[:300]
    })

for idx in incorrect_indices[:15]:
    txt = sub_test_df.iloc[idx]["review"]
    prob = float(test_probs[idx])
    lbl = "positive" if y_test[idx] == 1 else "negative"
    pred_lbl = "positive" if test_preds[idx] == 1 else "negative"
    sample_errors["incorrect"].append({
        "index": int(sub_test_df.index[idx]),
        "actual": lbl,
        "predicted": pred_lbl,
        "prob": prob,
        "text": txt[:300]
    })

import json
with open(LAB07_DIR / "error_samples.json", "w", encoding="utf-8") as f:
    json.dump(sample_errors, f, ensure_ascii=False, indent=2)
print("-> Đã lưu error_samples.json")

# Qualitative test cases (Section 23 đề bài)
print("\nKiểm tra các câu Qualitative Test Cases:")
test_sentences = [
    ("Short", "I really enjoyed this movie."),
    ("Long", "While the initial premise of the film promised an insightful exploration of human nature, the convoluted screenplay and amateurish cinematography completely derailed the experience."),
    ("Negation", "This movie was not good at all, completely disappointing."),
    ("Contrast", "The cinematography was breathtaking and the actors were great, but the plot was utterly terrible."),
    ("Ambiguous", "An interesting experience that leaves viewers questioning what they just watched.")
]

for label, s in test_sentences:
    toks = tokenize(s)
    ids = torch.tensor([[vocab.get(w, 1) for w in toks[:100]]], dtype=torch.long)
    best_model_4h.eval()
    with torch.no_grad():
        lg = best_model_4h(ids)
        pr = torch.sigmoid(lg).item()
        pred_sentiment = "Positive" if pr >= 0.5 else "Negative"
    print(f"  [{label}] '{s}' -> {pred_sentiment} (P(positive) = {pr:.4f})")

print("\n>>> HOÀN THÀNH TOÀN BỘ CÁC THỰC NGHIỆM LAB 07! <<<")
