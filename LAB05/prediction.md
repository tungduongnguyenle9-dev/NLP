# LAB05 — Prediction trước experiment

## Prediction 1 — Word order

- **Prediction:** RNN + Embedding xử lý thứ tự từ trực tiếp hơn TF-IDF + Logistic Regression.
- **Reason:** RNN đọc từng token theo thứ tự và cập nhật hidden state từ thông tin trước đó. TF-IDF chủ yếu biểu diễn tần suất hoặc mức độ quan trọng của từ, không trực tiếp mô hình hóa thứ tự.
- **Confidence:** High

## Prediction 2 — Long-range dependency

- **Prediction:** LSTM có khả năng xử lý dependency dài tốt hơn RNN thông thường.
- **Reason:** LSTM có cell state và các gates giúp kiểm soát thông tin cần giữ, từ đó giảm khó khăn khi học phụ thuộc xa.
- **Confidence:** High

## Prediction 3 — Sequence length: 50 → 200

- **Prediction:** Training của RNN sẽ chậm hơn khi tăng sequence length từ 50 lên 200.
- **Reason:** Model phải xử lý nhiều timestep hơn, làm tăng khối lượng tính toán và bộ nhớ.
- **Confidence:** High

## Prediction 4 — Truncation về 30 tokens

- **Prediction:** Truncate toàn bộ review về 30 tokens có thể làm giảm chất lượng classification.
- **Reason:** Review dài có thể chứa từ phủ định, ý đối lập hoặc kết luận sentiment ở phần bị cắt, khiến model thiếu context quan trọng.
- **Confidence:** High

## Prediction 5 — LSTM vs RNN accuracy

- **Prediction:** LSTM không chắc chắn đạt accuracy cao hơn RNN trên mọi dataset.
- **Reason:** Kết quả phụ thuộc vào độ dài và độ khó của dữ liệu, cấu hình huấn luyện và khả năng tổng quát hóa. Với sequence ngắn hoặc tác vụ đơn giản, RNN có thể đạt kết quả tương đương.
- **Confidence:** High
