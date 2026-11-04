# Báo cáo Phân tích Lỗi (Error Analysis) — LAB 07

Báo cáo phân tích hành vi mô hình **Transformer Encoder** trên tập dữ liệu IMDb Test Set (theo Mục 26 của đề bài).

---

## 1. Phân tích 3 dự đoán đúng (Correct Predictions)

### Bảng tổng hợp các ca đúng tiêu biểu

| ID | Index IMDb | Nhãn thật | Dự đoán | P(positive) | Trích đoạn nhận xét |
|:---:|:---:|:---:|:---:|:---:|:---|
| **C1** | 46799 | negative | negative | 0.190 | *"Although it has been 2 years, I still remember the complete waste that comprises the entire plot of the movie. Unfortunately, I came across this movie..."* |
| **C2** | 45891 | positive | positive | 0.930 | *"THE BLOB is a great horror movie, not merely because of the vividly horrific images of its nearly unstoppable, flesh-dissolving title character, but because it features a real societal message. It is, in many ways, a 'feel-good horror film.' The clever storyline is helped immeasurably by solid performance..."* |
| **C3** | 48397 | positive | positive | 0.829 | *"The saddest thing about this film is that only 8 people cared to leave a review of it and NO-ONE felt it worthwhile leaving a comment on the message boards... this is the film that people REALLY should have seen and given awards..."* |

### Phân tích chi tiết từng ca đúng:
* **C1 (Index 46799 — Negative rõ ràng):**
  * *Hành vi mô hình:* Dự đoán chính xác nhãn Negative với độ chắc chắn cao ($P(\text{positive}) = 0.190$, tương ứng $P(\text{negative}) = 0.810$).
  * *Nguyên nhân:* Câu chứa các từ khóa tiêu cực mang tính định lượng cảm xúc rất mạnh ngay đầu chuỗi: *"complete waste"*, *"unfortunately"*. Nhờ cơ chế Self-Attention và Positional Encoding, các từ này được phân bố trọng số chú ý cao và giữ nguyên ảnh hưởng qua lớp Mean Pooling.
* **C2 (Index 45891 — Positive trực diện):**
  * *Hành vi mô hình:* Dự đoán nhãn Positive cực kỳ tự tin ($P(\text{positive}) = 0.930$).
  * *Nguyên nhân:* Mật độ các từ ngữ tích cực xuất hiện dày đặc và có ngữ cảnh rõ ràng: *"great horror movie"*, *"societal message"*, *"feel-good"*, *"clever storyline"*, *"solid performance"*. Các attention heads phân bổ đều và cộng hưởng tín hiệu cảm xúc tích cực trên toàn bộ câu.
* **C3 (Index 48397 — Positive có sắc thái đánh lừa / Ambiguity):**
  * *Hành vi mô hình:* Dự đoán đúng Positive ($P(\text{positive}) = 0.829$).
  * *Nguyên nhân:* Câu mở đầu bằng cụm từ mang sắc thái buồn bã dễ gây nhầm lẫn (*"The saddest thing"*). Tuy nhiên, Self-Attention đã kết nối từ *"saddest"* với vế sau (*"only 8 people cared to leave a review"*, *"film that people REALLY should have seen and given awards"*), nhận diện được sắc thái nuối tiếc cho một tác phẩm hay thay vì chê bai chất lượng phim.

---

## 2. Phân tích 3 dự đoán sai (Incorrect Predictions)

### Bảng tổng hợp các ca sai tiêu biểu

| ID | Index IMDb | Nhãn thật | Dự đoán | P(positive) | Trích đoạn nhận xét | Hiện tượng lỗi chính |
|:---:|:---:|:---:|:---:|:---:|:---|:---|
| **I1** | 32483 | negative | **positive** | **0.884** | *"This is a film that has garnered any interest or praise it has received simply on the merit of being a lesbian interest piece. The performances are mostly emotionless compared to better films..."* | **Surface polarity dominance & Rare words** |
| **I2** | 10202 | positive | **negative** | **0.234** | *"This movie is pretty cheesy, but I do give it credit for at least trying to provide some characterization for it's principles. There are some great moments in the film and the dialogue has some great..."* | **Contrast sentence ("but", "give credit")** |
| **I3** | 18798 | positive | **negative** | **0.468** | *"I'm not going to tell anyone what happens in the end, but it did not fit with the movie. The rest of Imaginary Heroes, though, portrayed a realistic family going through realistic issues, such as death..."* | **Negation & Boundary ambiguity** |

### Phân tích chi tiết từng ca sai:

* **I1 (Index 32483 — Sai tự tin cao / Overconfident Error):**
  * *Hiện tượng:* Người viết phê bình gay gắt tác phẩm (*"emotionless"*), nhận định phim chỉ được chú ý vì yếu tố chủ đề. Tuy nhiên mô hình dự đoán Positive với xác suất rất cao ($0.884$).
  * *Cơ chế lỗi:* Các từ vựng có cực tính bề mặt tích cực như *"praise"*, *"interest"*, *"merit"*, *"better"* xuất hiện ở vế đầu đã thu hút attention weights rất lớn từ nhiều head. Transformer chưa được pretrain trên tập ngữ liệu khổng lồ nên chỉ nhận diện được sự tương quan bề mặt (bag-of-words bias) mà không hiểu được cấu trúc mỉa mai/phê bình tinh vi.
* **I2 (Index 10202 — Câu tương phản / Contrast clause):**
  * *Hiện tượng:* Người viết ban đầu chê phim nhẹ (*"pretty cheesy"*), nhưng dùng liên từ chuyển ý (*"but I do give it credit"*, *"great moments"*, *"great dialogue"*) để khẳng định cảm xúc tích cực. Mô hình dự đoán nhãn Negative ($P = 0.234$).
  * *Cơ chế lỗi:* Cụm từ tiêu cực xuất hiện ở đầu câu và từ *"cheesy"* nhận được sự chú ý lớn từ các positional encodings đầu tiên. Mặc dù có liên từ *"but"*, một Transformer block nông (1 layer) chưa đủ độ sâu để đảo ngược hoàn toàn biểu diễn ngữ nghĩa của mệnh đề tương phản.
* **I3 (Index 18798 — Phủ định & Gần ngưỡng quyết định):**
  * *Hiện tượng:* Review khen ngợi câu chuyện của bộ phim là chân thực (*"realistic family"*, *"realistic issues"*), chỉ không thích phần kết. Mô hình dự đoán Negative ($P = 0.468$), rất sát ngưỡng 0.5.
  * *Cơ chế lỗi:* Câu mở đầu bằng nhiều từ phủ định (*"not going to tell"*, *"did not fit"*). Tín hiệu tiêu cực này bị khuếch tán vào vector biểu diễn chung, khiến xác suất tích cực bị kéo tụt xuống dưới ngưỡng 0.5 một chút.

---

## 3. Thảo luận quan trọng: Attention $\neq$ Explanation

Trong quá trình phân tích lỗi của Transformer, một sai lầm phổ biến là đồng nhất hóa:
$$\text{Attention Weight cao} \iff \text{Lý do model đưa ra quyết định}$$

> [!WARNING]
> **Cần phân biệt rạch ròi giữa Hành vi mô hình (Model Behavior) và Diễn giải con người (Human Interpretation):**
>
> 1. **Bản chất của Attention weights:** Attention weights $A = \text{softmax}(QK^T / \sqrt{d_k})$ đơn thuần là các hệ số pha trộn tuyến tính (mixing coefficients) giữa các vector Value $V$. Nó thể hiện luồng tập hợp thông tin (information routing) tại một layer cụ thể chứ không phải là lời giải thích nhân quả (causal explanation).
> 2. **Tác động của các thành phần phi tuyến:** Sau bước Attention, vector còn phải đi qua các thành phần phi tuyến phức tạp gồm:
>    * Residual connection ($x + \text{Attention}(x)$)
>    * Layer Normalization (chuẩn hóa mean & variance trên feature dimension)
>    * Position-wise Feed-Forward Network với hàm kích hoạt phi tuyến ReLU ($W_2 \text{ReLU}(W_1 x + b_1) + b_2$)
>    * Pooling và Linear Classifier.
>    Do đó, một từ có attention weight lớn hoàn toàn có thể bị triệt tiêu hoặc biến đổi ngược hướng sau FFN và LayerNorm.
> 3. **Tính đa nghĩa của Attention:** Một attention head có thể chú ý vào một từ đơn giản vì từ đó là dấu ngắt câu (như dấu chấm, dấu phẩy) hoặc từ chức năng đóng vai trò "kho chứa thông tin trung gian" (information sink), hoàn toàn không mang ý nghĩa cảm xúc theo cách con người suy nghĩ.

---

## 4. Kết luận

Mô hình Transformer Encoder tối giản (1 block) mang lại sự cải thiện rõ rệt về độ chính xác ($73.12\%$ so với $64.00\%$ của LSTM baseline) nhờ khả năng tương tác trực tiếp giữa mọi cặp từ. Tuy nhiên, các lỗi sai chủ yếu xuất phát từ:
1. Độ sâu mô hình còn mỏng (1 layer) nên chưa phân tầng tốt các cấu trúc câu lồng ghép phức tạp.
2. Thiếu tri thức tiền huấn luyện (pretrained knowledge) về ngữ nghĩa sâu của từ vựng, dẫn đến việc bị đánh lừa bởi các từ mang cực tính bề mặt (như ca I1).
Điều này đặt ra nhu cầu tất yếu để chuyển sang mô hình **Pretrained Transformer (như BERT)** ở bài **LAB 08**.
