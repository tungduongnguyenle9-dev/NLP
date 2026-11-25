# Dự đoán trước khi chạy thực nghiệm — LAB 09

Ghi lại mấy suy đoán ban đầu của em trước khi bắt đầu code và chạy mô hình NER trên tập CoNLL-2003 (theo Mục 12 của đề bài).

---

### Prediction 1: Nếu dùng pretrained Transformer thay cho classifier đơn giản dựa trên word features thì NER thay đổi như thế nào?

* **Dự đoán:** Chắc chắn pretrained Transformer sẽ làm tốt hơn hẳn, đặc biệt là ở mấy từ hay bị hiểu nhầm nghĩa hoặc từ mới chưa gặp bao giờ.
* **Lý do em nghĩ vậy:**
  * Nếu chỉ dùng classifier đơn giản (như Logistic Regression với mấy đặc trưng thủ công như từ viết hoa, hậu tố, từ đứng trước/sau 1 bước), mô hình gần như chỉ nhìn "bề mặt" của từ và rất dễ đoán sai khi gặp cùng một chữ nhưng ngữ cảnh khác nhau. Ví dụ chữ *Washington* hoặc *Jordan* đứng ở vị trí khác nhau trong câu thì lúc là tên người, lúc là tên nước hay tên tổ chức; mấy feature rời rạc rất khó nắm hết được ý nghĩa cả câu.
  * Còn pretrained Transformer (như BERT hay DistilBERT) đã được đọc rất nhiều văn bản tiếng Anh từ trước rồi, lại có cơ chế self-attention giúp mỗi token "nhìn" được hết các từ xung quanh ở khoảng cách xa. Nhờ vậy vector biểu diễn của từng token mang theo ngữ cảnh cụ thể của câu đó luôn, nên phân biệt `PER`, `ORG`, `LOC` chắc sẽ mượt và ít bị nhầm hơn nhiều. Ngoài ra tokenizer dạng subword cũng giúp xử lý mấy từ lạ (OOV) tốt hơn so với từ điển cố định.

---

### Prediction 2: Các entity dài nhiều token sẽ dễ hay khó hơn entity chỉ có một token?

*(Ví dụ: `Microsoft` so với `United Nations Security Council`)*

* **Dự đoán:** Theo em thì mấy entity dài nhiều token sẽ **khó hơn** nhiều so với entity chỉ có một token.
* **Lý do em nghĩ vậy:**
  * Ở bài toán NER theo chuẩn BIO và chấm điểm ở mức toàn bộ entity (entity-level), một entity dài chỉ cần đoán hụt hoặc lệch **một token thôi** là coi như sai nguyên cả cụm đó luôn rồi. Ví dụ `United Nations Security Council` có 4 từ, mô hình phải đoán đúng liên tiếp `B-ORG I-ORG I-ORG I-ORG`. Nếu chẳng may từ `Council` cuối cùng bị đoán thành `O` hoặc tự dưng tách thành một `B-ORG` mới thì mất điểm cả cụm ngay.
  * Thêm nữa, các entity dài hay chứa mấy từ rất bình thường ở giữa như `of`, `the`, `and`, `for`, `Security`, `Council`... (ví dụ `University of California Berkeley`). Lúc đứng lẻ thì `of` hay `Security` đâu phải tên riêng, nên mô hình rất dễ bị "đứt gánh giữa đường", nhận diện sai ranh giới (entity boundary). Trong khi đó entity ngắn chỉ 1 token như `Microsoft` thì chỉ cần chốt đúng 1 nhãn `B-ORG` là ăn trọn điểm rồi, ít rủi ro lệch biên hơn.
  * *(À nhưng có một chút phân vân nhỏ: đôi khi mấy cụm rất nổi tiếng dài như `United Nations` thì lại có mẫu lặp đi lặp lại khá rõ, nhưng nhìn chung các cụm 3–4+ token chắc vẫn khó giữ chuẩn ranh giới hơn loại 1 token).*

---

### Prediction 3: Entity hiếm xuất hiện trong training data có khả năng gây lỗi nhiều hơn không?

* **Dự đoán:** Có, em nghĩ entity hiếm chắc chắn dễ bị lỗi hơn hẳn so với mấy entity xuất hiện dày đặc trong tập train.
* **Lý do em nghĩ vậy:**
  * Đầu tiên là với model baseline dùng word features, nếu một tên riêng chưa từng thấy hoặc chỉ gặp 1–2 lần trong tập train thì trọng số học được cho từ đó gần như bằng 0 hoặc rất yếu. Model chỉ còn biết bấu víu vào dấu hiệu viết hoa hay hậu tố, nên rất dễ bỏ sót (False Negative) hoặc đoán nhầm sang nhãn phổ biến hơn (ví dụ thấy viết hoa liền đoán đại thành `PER` hoặc `LOC`).
  * Với pretrained Transformer thì đỡ hơn một chút vì bản thân nó đã biết sẵn kiến thức từ lúc pretrain và dựa được vào ngữ cảnh xung quanh (kiểu thấy cụm *"arrived in [X]"* thì đoán được `[X]` là `LOC` dù tên thành phố đó lạ hoắc). Tuy nhiên trong CoNLL-2003 vẫn có nhiều tên người nước ngoài lạ, tên đội bóng nhỏ, hoặc từ viết tắt chuyên ngành vừa hiếm vừa ít xuất hiện lúc pretrain, hoặc hay bị tokenizer cắt vụn ra thành nhiều subword nhỏ lẻ, nên xác suất mô hình đoán sai loại entity hoặc sót luôn chắc chắn vẫn cao hơn nhiều so với mấy thực thể quen mặt.
