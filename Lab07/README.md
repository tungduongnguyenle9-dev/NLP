# LAB 07 — Transformer Architecture (From Attention to Transformer)

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ:** Học kỳ I - 2026  
**Giảng viên / TA:** Phạm Ngọc Hải  
**Hạn nộp:** 23:59 - 04/11/2026  

---

## 1. Giới thiệu tổng quan

Trong chuỗi bài thực hành NLP, **LAB 07** đánh dấu bước chuyển mình quan trọng từ **Mô hình hóa chuỗi tuần tự (Sequence Modeling với RNN/LSTM ở LAB 05)** sang **Tương tác trực tiếp giữa các token (Direct Token Interaction)** thông qua cơ chế Self-Attention đa đầu (Multi-Head Self-Attention).

Mục tiêu chính của Lab là hiểu tường tận từng thành phần cấu tạo nên khối **Transformer Encoder** và tự lắp ghép một khối hoàn chỉnh từ các tầng cơ bản (primitive layers) mà **không sử dụng** `nn.TransformerEncoderLayer` có sẵn:
$$\text{Input Embeddings} \to \text{Positional Encoding} \to \text{Multi-Head Attention} \to \text{Add \& Norm} \to \text{FFN} \to \text{Add \& Norm} \to \text{Output}$$

---

## 2. Cấu trúc thư mục & Danh mục nộp bài (Deliverables)

```text
Lab07/
├── Lab07_merged.pdf                    # [FILE NỘP DUY NHẤT] Hợp nhất toàn bộ 11 mục deliverables (49 trang)
├── README.md                           # Báo cáo tổng quan, hướng dẫn chạy và AI Statement
├── caculations.md                      # Bài làm tính toán thủ công bằng tay (Markdown + LaTeX)
├── prediction.md                       # 6 Dự đoán ban đầu và Bảng nghiệm chứng thực tế
├── positional_encoding.py              # Module tính Sinusoidal Positional Encoding
├── transformer_block.py                # Class TransformerBlock tự ghép từ primitive layers
├── transformer_classification.ipynb    # Notebook hoàn chỉnh chứa toàn bộ pipeline và outputs
├── results.csv                         # File tổng hợp số liệu của cả 4 bảng thực nghiệm
├── positional_encoding.png             # Trực quan hóa ma trận và sóng Positional Encoding
├── attention_heads.png                 # Heatmap phân bố trọng số chú ý của 4 heads
├── learning_curves.png                 # Đồ thị Training Loss và Validation Loss qua các epoch
├── confusion_matrix.png                # Ma trận nhầm lẫn trên tập IMDb Test Set
├── error_analysis.md                   # Phân tích 3 ca đoán đúng, 3 ca đoán sai & Thảo luận Attention!=Explanation
├── reflection.md                       # Đúc kết tiến trình LAB 01 -> 07 và cầu nối sang LAB 08 (BERT)
└── scans/                              # Thư mục ảnh trích xuất từng trang scan viết tay
```

*(Lưu ý: Toàn bộ các tài liệu, code, notebook, hình ảnh và 5 trang scan viết tay gốc đã được tích hợp đầy đủ vào file PDF duy nhất `Lab07_merged.pdf` để người chấm chỉ cần xem 1 file duy nhất).*

---

## 3. Tóm tắt kết quả các thực nghiệm chính

Tất cả các thực nghiệm được huấn luyện và đánh giá trên tập dữ liệu chuẩn **IMDb Dataset (50,000 Movie Reviews)** với phân chia cố định `SEED = 42` (kế thừa từ **LAB 05** và **LAB 06**):

### 3.1. Experiment 1: Permutation Experiment (`dog bites man` vs `man bites dog`)
* **Không có Positional Encoding:** Cosine Similarity giữa 2 câu hoán vị là **1.000000** (Khoảng cách L2 = $0.000001$). Mô hình hoàn toàn bất lực trong việc phân biệt thứ tự từ.
* **Có Positional Encoding:** Khoảng cách L2 tăng lên **0.384638**, chứng minh Positional Encoding phá vỡ tính hoán vị đối xứng và giúp mô hình nhận diện thứ tự từ vựng.

### 3.2. Experiment 2: Số lượng Attention Heads ($d_{model} = 128$)
* $h=1$ ($d_k=128$): $\text{Accuracy} = 0.6972$, $F_1 = 0.6919$, Thời gian = $87.57\text{s}$.
* $h=2$ ($d_k=64$): $\text{Accuracy} = 0.7324$, $F_1 = 0.7402$, Thời gian = $112.03\text{s}$.
* $h=4$ ($d_k=32$): $\text{Accuracy} = 0.7312$, $F_1 = 0.7417$, Thời gian = $123.27\text{s}$.
* $h=8$ ($d_k=16$): $\text{Accuracy} = 0.7428$, $F_1 = 0.7532$, Thời gian = $148.25\text{s}$.
*(Khi tăng heads, $d_k$ thu hẹp dần khiến mức tăng accuracy có xu hướng bão hòa).*

### 3.3. Experiment 4: Transformer Encoder vs LSTM Baseline
* **LSTM (LAB 05 Baseline):** $\text{Accuracy} = 0.6400$, $F_1 = 0.6949$, Thời gian = $20.92\text{s}$ (1,329,729 params).
* **Transformer Encoder:** $\text{Accuracy} = 0.7312$, $F_1 = 0.7417$, Thời gian = $123.27\text{s}$ (1,412,609 params).
*(Transformer vượt trội hơn LSTM **+9.12% Accuracy** và **+4.68% F1** nhờ cơ chế kết nối trực tiếp mọi cặp token).*

### 3.4. Experiment 5: Độ phức tạp bậc hai theo Chiều dài chuỗi $O(T^2)$
* $T=32$: Attention elements = $1,024$, Thời gian = $18.91\text{s}$, $F_1 = 0.6232$.
* $T=64$: Attention elements = $4,096$, Thời gian = $41.42\text{s}$, $F_1 = 0.6941$.
* $T=128$: Attention elements = $16,384$, Thời gian = $124.34\text{s}$, $F_1 = 0.7450$.
* $T=256$: Attention elements = $65,536$, Thời gian = $257.88\text{s}$, $F_1 = 0.7699$.
*(Số phần tử attention tăng gấp 64 lần và thời gian huấn luyện tăng hơn 13 lần từ $T=32$ lên $T=256$, minh chứng rõ ràng chi phí $O(T^2)$).*

### 3.5. Experiment 6: Chiến lược Pooling
* **Mean Pooling:** $\text{Accuracy} = 0.7312$, $F_1 = 0.7417$.
* **CLS-style Pooling:** $\text{Accuracy} = 0.5732$, $F_1 = 0.6041$.
*(Mean Pooling vượt trội do mô hình 1-block học từ đầu chưa có cơ chế pretraining chuyên biệt cho token [CLS]).*

---

## 4. Hướng dẫn chạy lại mã nguồn (Reproducibility)

### Yêu cầu môi trường
* Python $\ge 3.10$
* Các thư viện: `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

### Lệnh thực thi:
1. **Kiểm tra khối Transformer và Shape:**
   ```bash
   python3 transformer_block.py
   ```
2. **Kiểm tra Positional Encoding:**
   ```bash
   python3 positional_encoding.py
   ```
3. **Chạy lại toàn bộ chuỗi thực nghiệm và xuất lại toàn bộ đồ thị:**
   ```bash
   python3 run_lab07_experiments.py
   ```
4. **Mở và tương tác với Jupyter Notebook:**
   ```bash
   jupyter notebook transformer_classification.ipynb
   ```

---

## 5. AI Assistance Statement (Tuyên bố hỗ trợ AI - Mục 30)

Theo đúng quy định **AI Policy** của môn học tại Mục 30 đề bài:

```text
AI assistance statement
Tool: Antigravity AI Coding Assistant (Google DeepMind)
Purpose: 
  - Hỗ trợ triển khai boilerplate code huấn luyện mô hình và kiểm tra tương thích tensor shapes.
  - Tự động hóa vẽ và trích xuất các đồ thị thực nghiệm (positional_encoding.png, attention_heads.png, learning_curves.png, confusion_matrix.png).
  - Định dạng chuẩn hóa các báo cáo Markdown và bảng dữ liệu results.csv.
Generated:
  - Khung code của positional_encoding.py và transformer_block.py.
  - Script điều phối thực nghiệm tự động run_lab07_experiments.py và notebook transformer_classification.ipynb.
Modified:
  - Tinh chỉnh cơ chế masked mean pooling để tránh padding noise.
  - Tối ưu kích thước batch size và token max_length để hoàn thành thực nghiệm tối ưu trên phần cứng CPU.
Verified:
  - Toàn bộ tensor shapes được kiểm chứng độc lập theo Mục 18 (Q, K, V in R^(8x4x100x32), Attention Scores in R^(8x4x100x100)).
  - Bảng kiểm chứng dự đoán và phân tích lỗi được đối chiếu độc lập với dữ liệu mẫu thực tế.
  - Toàn bộ các câu hỏi lý thuyết và phản hồi (Reflection) được xây dựng dựa trên bản chất toán học của mô hình.
```
