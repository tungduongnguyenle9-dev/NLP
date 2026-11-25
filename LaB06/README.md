# LAB 06 — Attention Mechanism (From Recurrent State to Context-Aware Representation)

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ:** Học kỳ I - 2026  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Hạn nộp:** 23:59 - 28/10/2026  

---

## 1. Giới thiệu tổng quan

**LAB 06** chuyển tiếp từ mô hình hóa chuỗi tuần tự dựa trên trạng thái ẩn cuối (RNN/LSTM ở LAB 05) sang **cơ chế Attention**, cho phép mô hình kết hợp có trọng số thông tin từ toàn bộ chuỗi đầu vào thay vì nén toàn bộ câu vào một hidden state duy nhất $h_T$.

Công thức cốt lõi (Scaled Dot-Product Attention):
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

Mục tiêu chính của Lab:
- Tính toán thủ công attention scores, softmax weights, context vector, scaling và self-attention.
- Tự cài đặt `scaled_dot_product_attention` và `softmax` ổn định về số học bằng **NumPy** (không dùng thư viện attention có sẵn).
- Khảo sát ảnh hưởng của **scaling** ($\frac{1}{\sqrt{d_k}}$), **temperature** ($T$), **query**, và độ phức tạp **$O(T^2)$** theo chiều dài chuỗi.
- So sánh trực tiếp **LSTM** vs **Self-Attention** trên bài toán phân loại cảm xúc (Sentiment Classification) và phân tích lỗi kết hợp trực quan hóa attention.

---

## 2. Cấu trúc thư mục & Danh mục nộp bài (Deliverables)

```text
LaB06/
├── README.md                 # Tài liệu tổng quan, kết quả thực nghiệm và hướng dẫn chạy
├── caculation.pdf            # Bản scan 4 trang tính toán thủ công bằng tay
├── prediction.md             # 5 dự đoán trước thực nghiệm (Prediction, Reason, Confidence)
├── attention_numpy.py        # Cài đặt softmax ổn định số học & scaled_dot_product_attention bằng NumPy
├── self_attention.ipynb      # Notebook chạy toàn bộ thực nghiệm, trực quan hóa và đánh giá mô hình
├── results.csv               # Bảng kết quả định lượng so sánh LSTM vs Attention
├── attention_heatmap.png     # Heatmap trực quan hóa ma trận attention weights
├── confusion_matrix.png      # Ma trận nhầm lẫn của mô hình phân loại cảm xúc
├── error_analysis.md         # Phân tích 3 mẫu dự đoán đúng, 3 mẫu sai và nhận xét attention pattern
└── reflection.md             # Tổng kết bài học, hạn chế của Attention và chuyển tiếp sang Transformer
```

---

## 3. Tóm tắt kết quả thực nghiệm chính

### 3.1. So sánh LSTM vs Attention (`results.csv`)

| Model | Accuracy | F1 Score | Training Time (s) | Parameters |
| :--- | :---: | :---: | :---: | :---: |
| **LSTM** | `0.6972` | `0.7369` | `17.74` | `328,481` |
| **Attention** | `0.8306` | `0.8324` | `21.19` | `323,201` |

- **Nhận xét:** Mô hình Attention đạt Accuracy cao hơn **+13.34%** và F1 cao hơn **+9.55%** so với LSTM với số tham số ít hơn (323,201 so với 328,481), đổi lại thời gian huấn luyện tăng nhẹ (+3.45s).

### 3.2. Các quan sát chính từ thực nghiệm
- **Scaling & Temperature:** Chia cho $\sqrt{d_k}$ và tăng temperature $T$ làm phân phối softmax phẳng hơn (giảm độ tập trung quá mức và tránh bão hòa gradient); temperature thấp làm phân phối sắc nhọn hơn.
- **Query Dependence:** Thay đổi query trên cùng một câu làm thay đổi phân phối attention weights — attention phụ thuộc vào thông tin cần truy vấn chứ không phải danh sách từ quan trọng cố định.
- **Computational Cost $O(T^2)$:** Ma trận self-attention có kích thước $T \times T$, khiến chi phí tính toán và bộ nhớ tăng theo bình phương chiều dài chuỗi (ví dụ $T = 1{,}000 \to 10^6$ phần tử; $T = 10{,}000 \to 10^8$ phần tử).
- **Attention Visualization:** Heatmap chỉ phản ánh phân phối trọng số chú ý (*attention pattern*), không tự động đồng nghĩa với giải thích ngôn ngữ học hay khả năng suy luận (*reasoning*).

---

## 4. Hướng dẫn chạy lại mã nguồn

### Yêu cầu môi trường
- Python $\ge 3.10$
- Thư viện: `numpy`, `torch`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter`

### Thực thi
1. **Kiểm tra module cài đặt Attention bằng NumPy:**
   ```bash
   python3 -c "import numpy as np; from attention_numpy import scaled_dot_product_attention; x = np.eye(3, 2); print(scaled_dot_product_attention(x, x, x))"
   ```
2. **Mở và chạy toàn bộ notebook thực nghiệm:**
   ```bash
   jupyter notebook self_attention.ipynb
   ```

---

## 5. AI Assistance Statement

- **Tool:** AI coding assistant
- **Purpose:** Hỗ trợ kiểm tra cấu trúc deliverables, định dạng tài liệu `README.md`, và xử lý scan/gộp ảnh bài làm viết tay thành file PDF (`caculation.pdf`).
- **Generated / Modified:** Tài liệu tổng quan `README.md` được tạo dựa trên các file kết quả và báo cáo có sẵn trong thư mục `LaB06/`.
- **Verified:** Toàn bộ phần tính toán tay (`caculation.pdf`), dự đoán (`prediction.md`), phân tích lỗi (`error_analysis.md`) và đúc kết (`reflection.md`) được sinh viên tự thực hiện và kiểm tra lại theo đúng quy định AI Policy (Mục 30).
