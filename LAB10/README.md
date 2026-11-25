# LAB 10 — Extractive Question Answering

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng | **Học kỳ:** I - 2026  
**Giảng viên / TA:** Phạm Ngọc Hải | **Hạn nộp:** 23:59 - 25/11/2026

---

## 1. Tổng quan

LAB 10 fine-tune **Pretrained Transformer** (`distilbert-base-uncased`) với **Start/End span prediction head** trên tập **SQuAD v1.1** (`rajpurkar/squad`). Toàn bộ pipeline (tokenization, sliding window, `offset_mapping` alignment, QA loss $L = \frac{L_{\text{start}} + L_{\text{end}}}{2}$, span decoding $i \le j$, chuẩn hóa EM/F1 và Keyword-Overlap Baseline) được cài đặt trực tiếp trong `qa_model.py`, **không** dùng `pipeline("question-answering")` có sẵn của HuggingFace.

---

## 2. Cấu trúc thư mục (`LAB10/` — 12 file, tách riêng `calculations.md`)

```text
LAB10/
├── README.md                    # Tổng quan, hướng dẫn chạy, kết quả, Reproducibility & AI Usage
├── prediction.md                # 4 dự đoán trước khi chạy và đối chiếu sau thực nghiệm
├── tokenizer_alignment.ipynb    # Tokenization, sliding window & kiểm thử 4 edge cases alignment
├── qa_model.py                  # ExtractiveQAModel, Baseline, Span Decoding & EM/F1 metrics
├── qa_finetuning.ipynb          # Fine-tuning SQuAD & 5 thí nghiệm có kiểm soát
├── results.csv                  # Số liệu tổng hợp EM, F1, Training time các thí nghiệm
├── learning_curves.png          # Biểu đồ Train/Val Loss và Val EM/F1 qua 3 epochs
├── em_f1_analysis.png           # So sánh Baseline vs Transformer & phân bố EM vs F1
├── answer_length_analysis.png   # Hiệu năng EM/F1 theo 4 nhóm độ dài đáp án (1, 2-3, 4-6, 7+)
├── error_analysis.md            # 6 mẫu EM=0 & F1>0, 15 lỗi chi tiết & #wrong start vs #wrong end
├── document_qa.py               # Mini Document QA System chạy trên model đã fine-tune
└── reflection.md                # Reflection 300-500 từ (LAB 08-10), QA vs RAG & 12 câu Learning Check
```

---

## 3. Cách chạy

Môi trường Python 3.12 (`torch==2.2.2`, `transformers==4.57.6`, `datasets==3.6.0`, `numpy==1.26.4`, `pyarrow==17.0.0`, `pandas==2.2.3`, `matplotlib==3.9.4`):

```bash
python3 qa_model.py                                      # Self-test metrics, baseline, alignment, model
python3 document_qa.py --demo                            # Chạy demo Mini Document QA (3 tài liệu mẫu)
python3 document_qa.py --question "..." --context "..."  # Hỏi đáp trực tiếp qua CLI
```

Hai notebook `tokenizer_alignment.ipynb` và `qa_finetuning.ipynb` đã được thực thi hoàn chỉnh (0 lỗi) với kernel `nlp_lab10`.

---

## 4. Tóm tắt kết quả (`results.csv`)

### Exp 1 — Context Length (Val `n=300`, 600 train samples, `stride=64`, 2 epochs)
| Max length | EM (%) | F1 (%) | Training time (s) |
| :---: | :---: | :---: | :---: |
| 128 | 19.00 | 27.49 | 1467.2 |
| 256 | 5.67 | 12.90 | 714.7 |
| 384 | 4.67 | 12.95 | 877.0 |

### Exp 2 — Main QA Model (`1,000` train samples, `max_length=384, stride=128`, 3 epochs, `851.5s`)
- Epoch 1: `train_loss=4.3489`, `val_loss=3.8734`, `val_EM=7.33%`, `val_F1=13.80%`
- Epoch 2: `train_loss=3.1393`, `val_loss=3.4465`, `val_EM=14.00%`, `val_F1=22.06%`
- Epoch 3: `train_loss=2.0598`, `val_loss=3.2696`, `val_EM=19.67%`, `val_F1=28.12%`

### Exp 3 — Baseline vs Fine-tuned Transformer (Test `n=400`)
| Model | EM (%) | F1 (%) |
| :--- | :---: | :---: |
| Baseline (Keyword Overlap) | 13.50 | 23.70 |
| Fine-tuned Transformer (DistilBERT) | 24.25 | 31.91 |

### Exp 4 — Question/Context Lexical Overlap (Test `n=400`)
| Nhóm | Số mẫu | Baseline EM / F1 (%) | Transformer EM / F1 (%) |
| :--- | :---: | :---: | :---: |
| High Lexical Overlap (`>= 0.6`) | 281 | 14.95 / 25.60 | 25.27 / 33.73 |
| Low Lexical Overlap (`< 0.6`) | 119 | 10.08 / 19.23 | 21.85 / 27.62 |

### Exp 5 — Answer Length (Test `n=400`)
| Answer length | Số mẫu | EM (%) | F1 (%) |
| :---: | :---: | :---: | :---: |
| `1` token | 121 | 26.45 | 30.61 |
| `2–3` tokens | 183 | 26.78 | 35.14 |
| `4–6` tokens | 61 | 16.39 | 29.33 |
| `7+` tokens | 35 | 17.14 | 24.10 |

---

## 5. Reproducibility (Mục 30)

```text
Model: distilbert-base-uncased (ExtractiveQAModel + Linear(768, 2) start/end head)
Tokenizer: AutoTokenizer.from_pretrained("distilbert-base-uncased") (WordPiece fast tokenizer)
Dataset: SQuAD v1.1 (rajpurkar/squad)
Train split: 1,000 mẫu đầu từ train.shuffle(seed=42) (Exp 1 dùng 600 mẫu đầu)
Validation split: 300 mẫu [0:300] từ validation.shuffle(seed=42)
Test split: 400 mẫu [300:700] từ validation.shuffle(seed=42)
Max sequence length: 384 (Exp 1 so sánh 128, 256, 384)
Stride: 128 (Exp 1 dùng 64)
Batch size: 16 (LengthGroupedBatchSampler + dynamic batch padding)
Learning rate: 5e-5 (linear warmup 10% + linear decay)
Epochs: 3 (Exp 1 dùng 2)
Optimizer: AdamW (weight_decay=0.01, grad clip=1.0)
Random seed: 42
Hardware: Intel Core i9-9880H @ 2.30GHz (CPU 8 threads, 32 GB RAM)
Training time: 851.5s (~14.2 phút) cho mô hình chính Exp 2 (Exp 1: 1467.2s / 714.7s / 877.0s)
```

---

## 6. AI Usage (Mục 29)

```text
AI-assisted code:
- Hỗ trợ tra cứu tokenizer API, viết khung boilerplate huấn luyện/đánh giá, xử lý lỗi thư viện và vẽ biểu đồ matplotlib.

What was generated:
- Khung boilerplate cho ExtractiveQAModel, DataLoader dynamic padding, code vẽ 3 biểu đồ PNG và khung CLI trong document_qa.py.

What was modified:
- Chỉnh sửa cơ chế gom batch theo độ dài (LengthGroupedBatchSampler) cho CPU 8 luồng, tinh chỉnh align_answer_to_tokens, KeywordOverlapBaseline và phân loại lỗi wrong start vs wrong end.
- Tự viết gọn và đối chiếu nội dung prediction.md, error_analysis.md và reflection.md theo số liệu thực nghiệm.

What was verified:
- Pass toàn bộ self-test qa_model.py, chạy thực tế 4 edge cases & 500 mẫu SQuAD trong tokenizer_alignment.ipynb.
- Chạy hoàn chỉnh qa_finetuning.ipynb (0 lỗi), kiểm tra trực quan 3 ảnh PNG, đối chiếu results.csv và test chạy document_qa.py trên checkpoint đã lưu.
```
