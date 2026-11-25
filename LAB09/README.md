# LAB 09 — Named Entity Recognition and Information Extraction

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng | **Học kỳ:** Học kỳ I - 2026  
**Chủ đề:** From Sequence Classification to Token Classification

---

## 1. Giới thiệu tổng quan

LAB 09 chuyển từ bài toán phân loại chuỗi (`Sequence Classification` — gán 1 nhãn cho cả câu qua `[CLS]`) sang phân loại theo từng token (`Token Classification` — gán nhãn BIO cho từng token trong câu) để giải quyết bài toán **Named Entity Recognition (NER)** trên bộ dữ liệu **CoNLL-2003**, sau đó mở rộng thành pipeline **Information Extraction (IE)** trích xuất dữ liệu có cấu trúc.

Các điểm kỹ thuật trọng tâm được triển khai trong thư mục:
* Xây dựng mô hình **Baseline** dựa trên đặc trưng từ cục bộ + Logistic Regression (`baseline_ner.py`).
* Xử lý **Subword Tokenization & Label Alignment** theo chiến lược **Strategy A** (gán nhãn cho subword đầu tiên, dùng `IGNORE = -100` cho các subword tiếp theo và token đặc biệt) trong `tokenizer_alignment.ipynb`.
* Fine-tune **Pretrained Transformer** (`distilbert-base-cased`, $d=768$, $C=9$ nhãn BIO) không dùng lối tắt `pipeline("ner")`, kiểm tra trực tiếp `tokens`, `input_ids`, `attention_mask`, `word_ids`, `labels`, `logits`, `predictions` và đánh giá ở mức **entity-level** (`seqeval`) trong `ner_finetuning.ipynb`.
* Xây dựng và đánh giá định lượng pipeline **Information Extraction (Option A — News)** trên tập 30 câu có gold annotations (`information_extraction.py`).

---

## 2. Cấu trúc thư mục `LAB09/`

*(Lưu ý: theo yêu cầu riêng, file `calculations.md` không nằm trong gói bàn giao này; thư mục chứa đầy đủ 12 file còn lại theo Mục 29 của đề bài).*

```text
LAB09/
├── README.md                     # Tổng quan, kết quả chính, hướng dẫn chạy và khai báo AI Usage
├── prediction.md                 # 3 dự đoán ban đầu trước khi chạy thực nghiệm (Mục 12)
├── baseline_ner.py               # Mô hình Baseline: Token features + Logistic Regression (Mục 13)
├── tokenizer_alignment.ipynb     # Label alignment (Strategy A) & Experiment 1 trên 60 câu (Mục 8–11, 16–17)
├── ner_finetuning.ipynb          # Fine-tune DistilBERT-cased, inspection, Exp 2–5 & trích xuất lỗi (Mục 14, 16, 18–22)
├── results.csv                   # Bảng so sánh Baseline vs Transformer trên cùng tập test CoNLL-2003 (Mục 18)
├── entity_metrics.csv            # Precision, Recall, F1, Support theo từng loại PER, ORG, LOC, MISC (Mục 21)
├── entity_length_analysis.png    # Biểu đồ hiệu năng entity-level theo độ dài 1, 2, 3, 4+ tokens (Mục 22)
├── confusion_matrix.png          # Ma trận nhầm lẫn 9 nhãn BIO trên toàn bộ tập test
├── error_analysis.md             # Phân tích Exp 4–5, 15 ca lỗi (Exp 6) và Context Analysis (Mục 21–24)
├── information_extraction.py     # Pipeline IE (Option A — News) & đánh giá 30 câu gold (Mục 25–27)
└── reflection.md                 # Bài reflection 300–500 từ (Mục 32) & 10 câu Individual Learning Check (Mục 31)
```

---

## 3. Hướng dẫn chạy code

Môi trường Python 3.12 sử dụng các thư viện: `torch`, `transformers`, `datasets`, `seqeval`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `jupyter`, `nbconvert`.

1. **Chạy Baseline NER:**
   ```bash
   python baseline_ner.py
   ```
2. **Chạy Notebook Alignment & Tokenization Analysis:**
   Mở và chạy toàn bộ các cell trong `tokenizer_alignment.ipynb` (hoặc thực thi bằng `jupyter nbconvert --to notebook --execute --inplace tokenizer_alignment.ipynb`).
3. **Chạy Notebook Fine-tuning & Evaluation:**
   Mở và chạy `ner_finetuning.ipynb`. Notebook này huấn luyện `distilbert-base-cased` trên 6,000 câu train CoNLL-2003 (`seed=42`, 2 epochs), đánh giá trên toàn bộ 3,453 câu test, xuất `results.csv`, `entity_metrics.csv`, `entity_length_analysis.png`, `confusion_matrix.png` và lưu checkpoint model.
4. **Chạy Information Extraction & Evaluation (30 câu News):**
   ```bash
   python information_extraction.py --model-dir <duong-dan-toi-ner_model>
   ```

---

## 4. Tóm tắt kết quả thực nghiệm

### Experiment 1 — Tokenization Analysis (`tokenizer_alignment.ipynb`, 60 câu test: 30 ngắn, 30 dài)
* Tổng hợp 60 câu: `1,247` từ gốc → `1,731` subword tokens (không tính `[CLS]`/`[SEP]`), có `266` từ bị tách (`21.33%`), tỉ lệ chung $R = 1.3881$.
* Câu ngắn (30 câu): $R = 1.9515$ (`48.06%` từ bị tách) vs Câu dài (30 câu): $R = 1.2767$ (`16.04%` từ bị tách).
* Non-Entity (`O`): $R = 1.1933$ (`10.81%` từ bị tách) vs Entity (`PER/ORG/LOC/MISC`): $R = 2.0456$ (`56.84%` từ bị tách; cao nhất là `PER` với `72.63%` từ bị tách).

### Experiment 2 & 3 — Baseline vs Pretrained Transformer (3,453 câu test CoNLL-2003, `results.csv`)

| Model | Entity Precision | Entity Recall | Entity F1 | Token Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| Baseline (Logistic Regression) | 0.6799 | 0.7512 | **0.7138** | 0.9546 |
| Pretrained Transformer (`distilbert-base-cased`) | 0.8579 | 0.8725 | **0.8652** | 0.9741 |

*(Đối chiếu Experiment 3 ở Transformer: Token-level F1 trên các nhãn khác `O` là `0.8874`, cao hơn Entity-level F1 `0.8652` do các lỗi lệch biên từng phần như `1995 World Cup` → `World Cup`).*

### Experiment 4 — Hiệu năng theo từng loại Entity (`entity_metrics.csv`)

| Entity | Precision | Recall | F1 | Support |
| :--- | :---: | :---: | :---: | :---: |
| **PER** | 0.9502 | 0.9326 | **0.9413** | 1,617 |
| **LOC** | 0.8794 | 0.9047 | **0.8918** | 1,668 |
| **ORG** | 0.8152 | 0.8338 | **0.8244** | 1,661 |
| **MISC** | 0.7089 | 0.7493 | **0.7285** | 702 |

### Experiment 5 — Hiệu năng theo độ dài Entity (`entity_length_analysis.png`)
* **1 token:** F1 = `0.8544` (3,574 gold)
* **2 tokens:** F1 = `0.8960` (1,776 gold)
* **3 tokens:** F1 = `0.8112` (236 gold)
* **4+ tokens:** F1 = `0.8160` (62 gold)

### Mini Information Extraction Evaluation — Option A News (`information_extraction.py`, 30 câu gold)
* **Overall Entity-Level:** Precision = **0.9865**, Recall = **0.9733**, F1 = **0.9799** (`TP=73, FP=1, FN=2`).
* **Per-Type:** `PERSON` F1 = `0.9655`, `ORG` F1 = `0.9630`, `LOCATION` F1 = `1.0000`.
* 2 lỗi gặp phải là do khác biệt miền/thời gian so với Reuters 1996 (`Satya Nadella` bị gắn `ORG`, `SpaceX` bị sót).

---

## 5. AI Usage (Mục 28)

* **AI-assisted code:** Có sử dụng trợ lý AI hỗ trợ tra cứu API Hugging Face (`AutoTokenizer.word_ids`, `DataCollatorForTokenClassification`), tạo khung boilerplate cho các script/notebook (`baseline_ner.py`, `tokenizer_alignment.ipynb`, `ner_finetuning.ipynb`, `information_extraction.py`) và viết mã vẽ biểu đồ `matplotlib`/`seaborn`.
* **What was generated:** Khung hàm feature extraction cho baseline, hàm `tokenize_and_align`, vòng lặp fine-tune PyTorch + length-grouped batching, script đánh giá `seqeval` và trực quan hóa confusion matrix / entity length.
* **What was modified:** Điều chỉnh chiến lược lấy subword đầu tiên (Strategy A), cấu hình tập train 6,000 mẫu với length-grouped batching để chạy ổn định trên CPU, bổ sung bước trích xuất ví dụ lỗi thật và kiểm tra trực quan `tokens`, `input_ids`, `attention_mask`, `word_ids`, `labels`, `logits`, `predictions`; tự xây dựng bộ 30 câu News kèm gold annotations trong `information_extraction.py`.
* **What was verified:** Toàn bộ notebook và script đã được chạy trực tiếp để đối chiếu số liệu thật trên tập test CoNLL-2003 (3,453 câu) và 30 câu IE; kiểm tra tính nhất quán giữa `results.csv`, `entity_metrics.csv`, hai biểu đồ PNG và các phần tự viết (`prediction.md`, `error_analysis.md`, `reflection.md`).
