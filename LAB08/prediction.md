# Prediction Before Experiment — LAB 08

## Prediction 1
**Câu hỏi:** Pretrained Transformer fine-tuned trên IMDb có tốt hơn Transformer nhỏ train from scratch trên cùng dữ liệu không?

- **Prediction:** Có, pretrained Transformer fine-tuned sẽ cho kết quả cao hơn rõ rệt.
- **Reason:** Model train từ đầu phải tự học cả nghĩa từ, ngữ pháp lẫn cách phân loại cảm xúc chỉ từ tập IMDb có hạn, nên dễ bỏ sót các câu phủ định dài hoặc nhận xét lắt léo. Trong khi đó, BERT đã học sẵn biểu diễn ngữ cảnh tiếng Anh từ lượng văn bản khổng lồ qua pretraining, khi đem fine-tune chỉ cần điều chỉnh trọng số theo tác vụ phân loại nên học nhanh và chính xác hơn.
- **Expected evidence:** Trên cùng bộ chia IMDb và cùng `max_length`, Pretrained BERT có Accuracy và F1 vượt hơn mô hình train from scratch (khoảng 5–10%+), đồng thời validation loss giảm nhanh ngay từ 1–2 epoch đầu.

---

## Prediction 2
**Câu hỏi:** Nếu freeze toàn bộ pretrained encoder và chỉ train classification head thì kết quả sẽ như thế nào?

- **Lựa chọn:** **C. kém hơn** (so với fine-tuning toàn bộ).
- **Giải thích:** Khi đóng băng encoder, các lớp BERT giữ nguyên trọng số cũ và chỉ đóng vai trò trích xuất đặc trưng cố định. Vector `[CLS]` vẫn chứa ngữ nghĩa chung đủ để lớp Linear phía trên tách tương đối ổn, nhưng vì BERT được pretrain bằng MLM chứ không phải phân loại review phim nên một số sắc thái khen/chê đặc thù sẽ chưa được làm nổi bật. Mở khóa toàn bộ encoder để fine-tune sẽ giúp attention tập trung đúng tín hiệu cảm xúc hơn, nên cho kết quả tốt hơn chế độ freeze.

---

## Prediction 3
**Câu hỏi:** Nếu chỉ có 10% training data (so với 100%, 50%, 25%), model nào giảm performance mạnh hơn?

- **Dự đoán:** Model **train from scratch** sẽ giảm mạnh hơn nhiều so với pretrained model.
- **Cơ sở từ vai trò của pretraining:** Model train from scratch khởi tạo trọng số ngẫu nhiên nên toàn bộ khả năng hiểu từ vựng và cấu trúc câu đều phụ thuộc vào lượng dữ liệu downstream đưa vào. Khi chỉ còn 10% dữ liệu (vài trăm mẫu), model không đủ mẫu để học biểu diễn tổng quát và rất dễ overfit vào vài từ khóa lẻ tẻ, khiến điểm rơi nhanh. Ngược lại, pretrained model đã có sẵn biểu diễn ngôn ngữ học từ kho dữ liệu không nhãn trước đó, nên lượng nhãn ít ỏi ở mức 10% chủ yếu dùng để căn chỉnh lớp phân loại và tinh chỉnh nhẹ đầu ra, giúp đường cong hiệu năng giữ ổn định hơn hẳn.
