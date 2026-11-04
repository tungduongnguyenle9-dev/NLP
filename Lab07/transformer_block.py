"""
Transformer Block Implementation from Scratch.
Xây dựng khối Transformer Encoder từ các primitive layers (Linear, LayerNorm, Dropout, Softmax),
TUYỆT ĐỐI KHÔNG dùng nn.TransformerEncoderLayer có sẵn.

Kiến trúc chuẩn theo 'Attention Is All You Need':
    x -> Multi-Head Self-Attention -> Dropout -> Residual + LayerNorm (Add & Norm)
      -> Position-wise Feed-Forward -> Dropout -> Residual + LayerNorm (Add & Norm)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module.
    
    Shapes trong quá trình tính toán:
      - x: (B, T, d_model)
      - Q, K, V sau linear & split: (B, n_heads, T, d_k) với d_k = d_model // n_heads
      - Attention scores S = Q @ K^T / sqrt(d_k): (B, n_heads, T, T)
      - Attention weights A = softmax(S): (B, n_heads, T, T)
      - Context = A @ V: (B, n_heads, T, d_k)
      - Sau concat các heads: (B, T, d_model)
      - Output qua W_O: (B, T, d_model)
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) phải chia hết cho n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections cho Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Output projection W_O
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, return_attention: bool = False):
        """
        x: (B, T, d_model)
        mask: optional (B, 1, 1, T) hoặc (B, 1, T, T)
        """
        B, T, _ = x.shape

        # 1. Chiếu tuyến tính: (B, T, d_model) -> (B, T, d_model)
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # 2. Reshape & transpose thành multi-head: (B, n_heads, T, d_k)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # 3. Scaled dot-product attention scores: (B, n_heads, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Áp dụng mask nếu có (ví dụ: padding mask)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 4. Softmax tính attention weights: (B, n_heads, T, T)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)

        # 5. Nhân với Value V: (B, n_heads, T, d_k)
        context = torch.matmul(attn_weights_dropped, v)

        # 6. Concat các heads lại: (B, T, d_model)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # 7. Output projection: (B, T, d_model)
        output = self.w_o(context)

        if return_attention:
            return output, attn_weights
        return output


class PositionWiseFeedForward(nn.Module):
    """
    Position-Wise Feed-Forward Network:
        FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
    
    Shapes:
      - Input x: (B, T, d_model)
      - Hidden layer: (B, T, d_ff)
      - Output: (B, T, d_model)
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, d_model) -> (B, T, d_ff) -> ReLU -> (B, T, d_model)
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    Khối Transformer Encoder Block hoàn chỉnh:
        Input X
          │
          ▼
        Multi-Head Self-Attention
          │
          ▼
        Residual + LayerNorm (Add & Norm 1)
          │
          ▼
        Position-Wise Feed-Forward Network
          │
          ▼
        Residual + LayerNorm (Add & Norm 2)
          │
          ▼
        Output
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, return_attention: bool = False):
        """
        x: (B, T, d_model)
        mask: optional (B, 1, 1, T)
        """
        # --- Sub-layer 1: Multi-Head Self-Attention + Residual + LayerNorm ---
        if return_attention:
            attn_out, attn_weights = self.attention(x, mask=mask, return_attention=True)
        else:
            attn_out = self.attention(x, mask=mask, return_attention=False)
            attn_weights = None

        # Residual connection + LayerNorm: y = LayerNorm(x + Dropout(Attention(x)))
        x = self.norm1(x + self.dropout1(attn_out))

        # --- Sub-layer 2: Feed-Forward Network + Residual + LayerNorm ---
        ffn_out = self.ffn(x)
        # Residual connection + LayerNorm: y = LayerNorm(x + Dropout(FFN(x)))
        x = self.norm2(x + self.dropout2(ffn_out))

        if return_attention:
            return x, attn_weights
        return x


if __name__ == "__main__":
    # Shape verification theo Mục 18 của đề bài:
    # B = 8, T = 100, d_model = 128, n_heads = 4, d_ff = 512
    # d_k = 128 / 4 = 32
    B, T, d_model, n_heads, d_ff = 8, 100, 128, 4, 512

    print("=== KIỂM TRA SHAPE THEO MỤC 18 ĐỀ BÀI ===")
    print(f"Tham số: B={B}, T={T}, d_model={d_model}, n_heads={n_heads}, d_k={d_model // n_heads}, d_ff={d_ff}")

    block = TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=0.1)
    dummy_input = torch.randn(B, T, d_model)

    output, attn_weights = block(dummy_input, return_attention=True)

    print(f"X (Input shape):             {tuple(dummy_input.shape)}")
    print(f"Attention Weights shape:     {tuple(attn_weights.shape)}  (Kỳ vọng: (8, 4, 100, 100))")
    print(f"Output shape:                {tuple(output.shape)}  (Kỳ vọng: (8, 100, 128))")

    assert tuple(dummy_input.shape) == (8, 100, 128)
    assert tuple(attn_weights.shape) == (8, 4, 100, 100)
    assert tuple(output.shape) == (8, 100, 128)
    print(">>> TẤT CẢ SHAPES ĐỀU CHÍNH XÁC 100%!")
