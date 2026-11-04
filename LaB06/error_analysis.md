# Error Analysis — LAB 06

## 1. Phân tích các dự đoán đúng

- **C1:** Review có những từ mang nghĩa tiêu cực khá rõ như "complete waste" và "unfortunately". Mô hình dự đoán đúng nhãn negative, tuy nhiên xác suất positive là 0.405 nên độ chắc chắn chưa cao.
- **C2:** Mô hình dự đoán positive với xác suất 0.990. Trong review có những cụm từ tích cực như "great horror movie", nên kết quả này khá hợp lý.
- **C3:** Review bắt đầu bằng "The saddest thing", dễ gây nhầm là tiêu cực. Tuy nhiên, nội dung này đang nói về việc bộ phim có ít người đánh giá chứ không trực tiếp chê chất lượng phim. Mô hình vẫn dự đoán đúng positive với xác suất 0.839.

## 2. Phân tích các dự đoán sai

- **I1:** Mô hình dự đoán positive với xác suất khá cao (0.844) dù nhãn thật là negative. Có thể những từ như "interest" và "praise" đã khiến mô hình nghiêng về tích cực mà chưa nắm được ý phê bình của review.
- **I2:** Đoạn review chủ yếu giới thiệu nhân vật và nội dung phim, chưa thể hiện cảm xúc rõ ràng. Mô hình dự đoán positive với xác suất 0.575, khá gần ngưỡng 0.5 nên có thể thấy dự đoán chưa chắc chắn.
- **I3:** Review nhắc đến những điểm yếu của một bộ phim kinh phí thấp. Có thể mô hình tập trung vào các nhận xét tiêu cực này mà chưa đánh giá đúng thái độ chung của người viết. Xác suất positive là 0.466, chỉ thấp hơn ngưỡng phân loại một chút.

## 3. Nhận xét chung

Mô hình hoạt động tốt với những review có từ ngữ thể hiện cảm xúc rõ ràng. Tuy nhiên, khi review có cả ý tích cực và tiêu cực hoặc chủ yếu mô tả nội dung, mô hình vẫn có thể nhầm lẫn.

Trong ba dự đoán sai, I1 đáng chú ý nhất vì mô hình dự đoán sai với xác suất khá cao. Còn I2 và I3 đều có xác suất gần 0.5, cho thấy mô hình chưa phân biệt rõ hai nhãn.

Attention weights có thể giúp quan sát mô hình đang chú ý vào những token nào, nhưng chưa đủ để khẳng định đó là nguyên nhân trực tiếp khiến dự đoán sai.

## 4. Kết luận

Attention giúp mô hình khai thác thông tin từ nhiều vị trí trong câu, nhưng vẫn có hạn chế khi xử lý những review có cách diễn đạt phức tạp hoặc cảm xúc không rõ ràng.


### 3 dự đoán đúng

| ID | Dòng IMDb | Nhãn thật | Dự đoán | P(positive) | Trích review |
|---|---:|---|---|---:|---|
| C1 | 46799 | negative | negative | 0,405 | Although it has been 2 years, I still remember the complete waste that comprises the entire plot of the movie. Unfortunately... |
| C2 | 45891 | positive | positive | 0,990 | THE BLOB is a great horror movie, not merely because of the vividly horrific images... |
| C3 | 48397 | positive | positive | 0,839 | The saddest thing about this film is that only 8 people cared to leave a review of it... |

### 3 dự đoán sai

| ID | Dòng IMDb | Nhãn thật | Dự đoán | P(positive) | Trích review |
|---|---:|---|---|---:|---|
| I1 | 32483 | negative | positive | 0,844 | This is a film that has garnered any interest or praise it has received simply on the merit of being a lesbian interest piece... |
| I2 | 25851 | negative | positive | 0,575 | Gene Hackman is a former Marine Corps colonel who musters a handful of private Vietnam vets... |
| I3 | 36838 | positive | negative | 0,466 | We're talking about a low budget film, and it's understandable that there are some weaknesses... |

