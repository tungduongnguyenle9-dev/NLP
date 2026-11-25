# Dự đoán trước khi chạy thực nghiệm — LAB 10

Em ghi lại 4 dự đoán cá nhân trước khi fine-tune model trên SQuAD (Mục 9), rồi đối chiếu ngắn gọn sau khi chạy xong.

---

## Phần 1: Dự đoán ban đầu (Trước khi chạy)

### Prediction 1 — Nhiều từ trùng giữa question và context
* **Dự đoán:** Model **khá dễ bị nhầm**.
* **Lý do:** Chỗ nào trong context lặp lại nhiều từ khóa giống câu hỏi thì attention dễ dồn vào đó. Nếu có câu bẫy (distractor) trùng nhiều từ nhưng nói về ý khác, model dễ bắt nhầm chỗ đó thay vì đáp án thật.

### Prediction 2 — Answer span càng dài có khó hơn không?
* **Dự đoán:** **Có**, càng dài càng dễ lệch biên (`start` hoặc `end`).
* **Lý do:** Đáp án 1–2 từ kiểu tên riêng, năm sinh thì chỉ cần chấm đúng 1 chỗ. Còn đáp án dài 4–7+ từ rất dễ lấy dư giới từ/dấu câu ở đầu hoặc cắt thiếu vài từ ở cuối, làm **EM tụt mạnh hơn F1**.

### Prediction 3 — Context nhiều thông tin không liên quan / distractor
* **Dự đoán:** Hiệu năng (cả EM lẫn F1) **sẽ giảm rõ rệt**.
* **Lý do:** Câu trả lời đúng bị chìm giữa nhiều chữ thừa, dễ bị phân tán attention vào distractor và dễ bị cắt sang window khác khi vượt `max_length`.

### Prediction 4 — Đổi thứ tự các câu trong context (giữ nguyên câu chứa đáp án)
* **Dự đoán:** Điểm tổng chắc **ít bị ảnh hưởng**, nhưng từng câu cụ thể có thể dao động nhẹ.
* **Lý do:** Self-attention đọc toàn bộ chuỗi cùng lúc nên không phụ thuộc tuyến tính như RNN. Tuy nhiên em hơi phân vân vì đổi thứ tự câu có thể làm đứt mạch tham chiếu (như `he`, `it`, `this`) và đổi vị trí tương đối.

---

## Phần 2: Đối chiếu sau khi chạy thực nghiệm

1. **Prediction 1 — Đúng:** Ở Exp 4, khi độ trùng từ khóa giảm (`>= 0.6` vs `< 0.6`), Transformer giảm từ `EM 25.27% / F1 33.73%` xuống `21.85% / 27.62%` (Baseline còn tụt mạnh hơn: `14.95%` xuống `10.08%`). Nhiều lỗi thật trong `error_analysis.md` đúng là do câu bẫy trùng từ khóa.
2. **Prediction 2 — Đúng sát dự đoán:** Ở Exp 5 (`answer_length_analysis.png`), nhóm `1 token` (`EM 26.45% / F1 30.61%`) và `2–3 tokens` (`26.78% / 35.14%`) cao hơn hẳn nhóm `4–6 tokens` (`EM 16.39% / F1 29.33%`) và `7+ tokens` (`17.14% / 24.10%`). Đúng là EM rớt gắt hơn F1 khi đáp án dài.
3. **Prediction 3 — Đúng (tụt mạnh hơn em nghĩ):** Trên 100 mẫu test probe, khi chèn thêm đoạn nền không liên quan + câu distractor, điểm tụt thẳng từ `EM 32.0% / F1 40.4%` xuống còn `12.0% / 14.2%`.
4. **Prediction 4 — Đúng một nửa:** Trên 100 mẫu test probe, đảo thứ tự câu chỉ làm điểm chung giảm nhẹ (`EM 32.0% → 31.0%`, `F1 40.4% → 39.43%`), nhưng soi từng mẫu thì có tới **`52/100` dự đoán bị đổi chuỗi** — cho thấy vị trí và mạch câu vẫn tác động không nhỏ.
