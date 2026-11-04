"""
Positional Encoding for Transformer Architecture.
Calculates sinusoidal positional encoding as defined in 'Attention Is All You Need':
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
"""

import math
import torch
import torch.nn as nn
import numpy as np


def get_sinusoidal_encoding_table(n_position: int, d_model: int) -> np.ndarray:
    """
    Sinh ma trận positional encoding dạng NumPy array kích thước (n_position, d_model).
    Dùng cho thực nghiệm, visualize hoặc tính toán độc lập.
    """
    pe_table = np.zeros((n_position, d_model), dtype=np.float32)
    for pos in range(n_position):
        for i in range(0, d_model, 2):
            div_term = math.pow(10000.0, 2 * i / d_model)
            pe_table[pos, i] = math.sin(pos / div_term)
            if i + 1 < d_model:
                pe_table[pos, i + 1] = math.cos(pos / div_term)
    return pe_table


class PositionalEncoding(nn.Module):
    """
    PyTorch Module thực hiện cộng Sinusoidal Positional Encoding vào word embeddings:
        X' = X + PE[:, :T, :]
    
    Shapes:
        Input X:  (Batch_size, Seq_len, d_model)
        Output:   (Batch_size, Seq_len, d_model)
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Tính toán ma trận PE
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        ) # (d_model / 2,)

        # Cột chẵn là sin, cột lẻ là cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Lưu dưới dạng buffer (không cập nhật gradient trong quá trình training)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        Returns: (B, T, d_model)
        """
        # Cộng PE tương ứng với độ dài chuỗi T vào embedding x
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


if __name__ == "__main__":
    # Test shape & giá trị cơ bản
    d_model = 128
    seq_len = 100
    batch_size = 8

    pe_layer = PositionalEncoding(d_model=d_model, max_len=500, dropout=0.0)
    dummy_x = torch.zeros(batch_size, seq_len, d_model)
    output = pe_layer(dummy_x)

    print(f"Input shape:  {dummy_x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, d_model), "Shape không khớp!"
    print("Test PositionalEncoding thành công!")
