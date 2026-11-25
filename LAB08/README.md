# LAB 08 — Pretrained Transformers & BERT Fine-Tuning

## Danh mục file
- `README.md`: Tổng hợp kết quả, thông số tái lập (Reproducibility), khai báo AI Usage và 12 câu kiểm tra lý thuyết.
- `prediction.md`: 3 dự đoán trước khi chạy thực nghiệm.
- `tokenization.ipynb`: Thực hành WordPiece tokenizer, subword, special tokens, `input_ids`, `attention_mask`.
- `pretrained_classification.py`: Module `PretrainedSentimentClassifier` (`[CLS]` + Linear head, hỗ trợ `freeze_encoder`) và `FromScratchTransformerClassifier`.
- `fine_tuning.ipynb`: Notebook chạy 5 thực nghiệm (Exp 1–5), confusion matrix và confidence analysis.
- `results.csv`: Bảng số liệu tổng hợp toàn bộ 18 lượt chạy.
- `data_efficiency.png`: Đường cong Accuracy và F1 theo tỷ lệ dữ liệu train (`10% -> 100%`).
- `confusion_matrix.png`: Ma trận nhầm lẫn trên IMDb Test (600 mẫu).
- `confidence_analysis.png`: Độ chính xác theo 3 mức tự tin (`Low / Medium / High`) và phân phối xác suất.
- `error_analysis.md`: Phân tích 8 mẫu lỗi thực tế, đánh giá confidence và trả lời Q1–Q6.
- `reflection.md`: Đúc kết 331 từ về lý do thực tế ưu tiên pretrained model + fine-tuning.

---

## Tóm tắt kết quả thực nghiệm (`results.csv`)

### Exp 1 & 2 — From-Scratch vs Frozen vs Fine-Tuning (`T=128`)
| Mô hình / Chế độ | Trainable Params | Accuracy | F1 | Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| From-Scratch Transformer (1L, d=128) | 1,412,738 | 0.7300 | 0.7445 | 24.58 |
| Pretrained BERT-Tiny (Frozen) | 258 | 0.6967 | 0.6884 | 18.62 |
| Pretrained BERT-Tiny (Fine-tuning) | 4,386,178 | **0.7483** | 0.7156 | 36.56 |

### Exp 3 — Data Efficiency (`10% -> 100%` train)
| % Train (Số mẫu) | BERT Acc | BERT F1 | Scratch Acc | Scratch F1 |
| :---: | :---: | :---: | :---: | :---: |
| 10% (120) | 0.5733 | 0.5803 | 0.5000 | 0.0196 |
| 25% (300) | 0.6017 | 0.6713 | 0.6317 | 0.6783 |
| 50% (600) | 0.6833 | 0.6975 | 0.6083 | 0.6498 |
| 100% (1,200) | 0.7483 | 0.7156 | 0.7300 | 0.7445 |

### Exp 4 — Max Sequence Length ($T \in \{64, 128, 256\}$)
| Max Length ($T$) | $T^2$ | Accuracy | F1 | Time (s) | Peak RSS (MB) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 64 | 4,096 | 0.7033 | 0.6728 | 20.50 | 2074.0 |
| 128 | 16,384 | 0.7483 | 0.7156 | 36.56 | 1982.1 |
| 256 | 65,536 | **0.7750** | **0.7840** | 63.77 | 2325.8 |

### Exp 5 — Domain Shift (Train trên IMDb)
| Tập đánh giá (Domain) | Accuracy | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: | :---: |
| IMDb Movie Reviews (in-domain, n=600) | 0.7483 | 0.8225 | 0.6333 | 0.7156 |
| Amazon Product Reviews (out-of-domain, n=500) | 0.7320 | 0.7320 | 0.7320 | 0.7320 |
| Financial News (out-of-domain, n=662) | 0.5423 | 0.6224 | 0.4772 | 0.5402 |

---

## Cách chạy
```bash
# 1. Kiểm tra module và kiến trúc model
python3 pretrained_classification.py

# 2. Chạy notebook tokenization và fine-tuning (Jupyter / nbclient)
jupyter nbconvert --to notebook --execute --inplace tokenization.ipynb
jupyter nbconvert --to notebook --execute --inplace fine_tuning.ipynb
```

---

## Reproducibility (Mục 23)
```text
Model: google/bert_uncased_L-2_H-128_A-2 (4,386,178 params) & FromScratchTransformerClassifier (1L, d=128, H=4, 1,412,738 params)
Tokenizer: AutoTokenizer WordPiece (google/bert_uncased_L-2_H-128_A-2, vocab_size=30,522)
Dataset: IMDb Dataset (/Users/user/NLP/IMDB Dataset.csv), Amazon Polarity (n=500), Financial News Sentiment (n=662)
Train split: 1,200 mẫu (600 positive / 600 negative, stratified)
Validation split: 400 mẫu (200 positive / 200 negative, stratified)
Test split: 600 mẫu (300 positive / 300 negative, stratified)
Max sequence length: 128 (Exp 4 thử nghiệm thêm 64 và 256)
Batch size: 32 (train) / 64 (eval)
Learning rate: 1.2e-4 (BERT Fine-tuning), 3e-3 (BERT Frozen), 1e-3 (From-Scratch)
Epochs: 4 (BERT Fine-tuning & Frozen), 5 (From-Scratch)
Optimizer: AdamW (weight_decay=0.01, grad clip=1.0)
Random seed: 42
Hardware: 8-Core Intel Core i9 (2.3 GHz, 16 logical cores), 32 GB RAM, CPU-only (torch.set_num_threads(4))
Training time: 36.56s (BERT Fine-tuning T=128), 18.62s (Frozen), 24.58s (From-Scratch); toàn bộ fine_tuning.ipynb: 291.8s
```

---

## AI Usage (Mục 24)
```text
AI-assisted code: Hỗ trợ viết boilerplate PyTorch/HuggingFace, script dựng notebook và vẽ biểu đồ matplotlib/seaborn.
What was generated: Khung class trong pretrained_classification.py, hàm train/eval và code vẽ 3 biểu đồ trong fine_tuning.ipynb.
What was modified: Điều chỉnh kích thước tập dữ liệu (1,200/400/600), chọn backbone BERT-Tiny và tinh chỉnh learning rate/epochs cho vừa CPU.
What was verified: Chạy thực tế toàn bộ pretrained_classification.py, tokenization.ipynb và fine_tuning.ipynb không lỗi, kiểm tra số liệu khớp results.csv và biểu đồ.
```

---

## Individual Learning Check (Mục 27)
1. **Pretraining khác fine-tuning ở đâu?**
   Pretraining học biểu diễn ngôn ngữ tổng quát trên lượng văn bản lớn không nhãn (như MLM), còn fine-tuning huấn luyện tiếp model đó trên tập dữ liệu có nhãn của một bài toán cụ thể.
2. **MLM là gì?**
   Masked Language Modeling là bài toán che ngẫu nhiên một số token bằng `[MASK]` rồi bắt model nhìn ngữ cảnh hai chiều xung quanh để đoán lại đúng token bị che.
3. **Tại sao cần tokenizer?**
   Vì Transformer không đọc trực tiếp chữ viết thô mà chỉ nhận chuỗi số nguyên (`input_ids`) để tra bảng embedding.
4. **Subword token giải quyết vấn đề gì?**
   Nó tách từ hiếm hoặc từ ghép thành các mảnh nhỏ hơn (vd `over` + `##act` + `##ing`), vừa giữ kích thước từ điển gọn vừa giảm tối đa tình trạng gặp `[UNK]`.
5. **`[CLS]` thường được sử dụng như thế nào?**
   Trong bài toán phân loại câu kiểu BERT, vector đầu ra tại vị trí `[CLS]` được lấy làm đại diện cho cả chuỗi rồi đưa qua lớp Linear để tính logits dự đoán.
6. **Attention mask dùng để làm gì?**
   Đánh dấu `1` cho token thật và `0` cho token đệm `[PAD]`, giúp Self-Attention bỏ qua các vị trí padding khi tính trọng số.
7. **Frozen encoder nghĩa là gì?**
   Là tắt cập nhật gradient (`requires_grad=False`) cho toàn bộ phần thân Transformer đã pretrain, chỉ cho phép train lớp classification head ở cuối.
8. **Fine-tuning thay đổi những tham số nào?**
   Thay đổi cả trọng số của classification head mới gắn thêm lẫn trọng số bên trong các tầng encoder của pretrained model.
9. **Vì sao pretrained model có thể hữu ích khi downstream dataset nhỏ?**
   Vì nó đã học sẵn ngữ pháp, nghĩa từ và quan hệ ngữ cảnh ở bước pretraining nên không bị mất khả năng khái quát khi số nhãn downstream ít (như mốc 10% chỉ 120 mẫu).
10. **Pretrained model có đảm bảo không bị domain shift không?**
    Không. Model fine-tune trên review phim (IMDb) sang tin tài chính nhiều thuật ngữ lạ vẫn tụt từ 74.83% xuống 54.23%.
11. **Vì sao tăng sequence length làm chi phí attention tăng?**
    Vì mỗi token đều tính tương tác với mọi token khác tạo ra ma trận $T \times T$, nên khi $T$ tăng thì số phép tính và bộ nhớ tăng theo bậc hai $O(T^2)$.
12. **Confidence cao có đảm bảo prediction đúng không?**
    Không. Thực nghiệm có tới 37 ca model tự tin trên 90% nhưng vẫn đoán sai (`confidence != correctness`).
