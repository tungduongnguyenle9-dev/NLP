# Reflection — LAB 05

## 1. Kết quả thực nghiệm

Thực nghiệm sử dụng IMDb 50.000 review (25.000 positive, 25.000 negative), chia thành 40.000 train, 5.000 validation và 5.000 test. Các mô hình được huấn luyện với vocabulary 10.000 từ và 3 epochs.

| Model | Accuracy | F1 | Training time |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.897 | 0.897 | 10.35 s |
| RNN | 0.559 | 0.503 | 24.92 s |
| LSTM | 0.778 | 0.780 | 29.17 s |
| GRU | 0.814 | 0.800 | 48.98 s |

TF-IDF + Logistic Regression cho kết quả tốt nhất trong lần chạy này. Trong nhóm sequence models, GRU có F1 cao nhất, tiếp đến là LSTM và RNN. Vì vậy, sử dụng mô hình có khả năng xử lý thứ tự từ không đồng nghĩa với accuracy cao hơn baseline trên mọi bài toán.

## 2. RNN, LSTM và GRU

RNN cập nhật hidden state theo thứ tự token, nên có khả năng biểu diễn ngữ cảnh và thứ tự từ mà Average Word2Vec không giữ được. Tuy nhiên, thông tin phải truyền qua nhiều timestep; RNN có thể gặp vanishing gradient khi học dependency dài. LSTM dùng cell state và các gates để điều khiển thông tin được giữ hoặc bỏ. GRU dùng các gates với kiến trúc gọn hơn LSTM, không có cell state riêng.

Trong biểu đồ learning curves, loss của LSTM và GRU giảm rõ sau 3 epochs, còn validation loss của RNN gần như không giảm. Chưa thấy dấu hiệu validation loss tăng trở lại trong ba epochs; do đó chưa đủ cơ sở kết luận mô hình đã overfit.

## 3. Ảnh hưởng của sequence length

| LSTM max_length | Tokens retained | F1 | Training time |
|---|---:|---:|---:|
| 50 | 21.45% | 0.757 | 17.31 s |
| 100 | 41.37% | 0.780 | 29.17 s |
| 200 | 68.45% | 0.778 | 73.61 s |

Tăng `max_length` giúp giữ lại nhiều token hơn nhưng làm thời gian train tăng. F1 tốt nhất ở độ dài 100 trong lần chạy này; tăng lên 200 không tiếp tục cải thiện F1. Truncation có thể làm mất các từ thể hiện cảm xúc, phủ định hoặc phần kết luận quan trọng của review. Cần cân bằng lượng context giữ lại với chi phí tính toán.

## 4. Context và lỗi dự đoán

Trong thử nghiệm thay đổi vị trí từ khóa, xác suất positive với `excellent` lần lượt là 0.404, 0.313, 0.303 (đầu, giữa, cuối); với `terrible` là 0.124, 0.153, 0.118. Kết quả thay đổi theo vị trí nhưng chưa thể dùng thử nghiệm nhỏ này để chứng minh vanishing gradient.

Ở phần application, LSTM dự đoán đúng chiều cảm xúc cho câu ngắn, câu có `not` và câu tương phản có `but`; tuy nhiên câu tương phản chỉ có P(positive) = 0.524, khá gần ngưỡng 0.5. Khi phân tích lỗi trên IMDb, cần kiểm tra toàn bộ review, đặc biệt các đoạn bị cắt ở 100 token, phủ định và tương phản, thay vì mặc định mọi dự đoán sai đều do mô hình không hiểu thứ tự từ.

## 5. Từ RNN đến Attention

Sequence models có thể sử dụng thứ tự và trạng thái ngữ cảnh, nhưng phải xử lý tuần tự và việc truyền thông tin qua nhiều bước có thể khiến dependency xa khó học. Attention cho phép mô hình tập trung trực tiếp vào những token liên quan thay vì chỉ phụ thuộc vào hidden state truyền qua chuỗi. Vì vậy, Attention là hướng tiếp theo phù hợp để xử lý ngữ cảnh dài và là nền tảng cho Transformer.

**Kết luận:** Trong thực nghiệm này, LSTM và GRU cải thiện đáng kể so với RNN, nhưng TF-IDF + Logistic Regression vẫn đạt F1 cao nhất. Việc chọn mô hình cần dựa trên kết quả thực nghiệm, độ dài văn bản và chi phí tính toán, không chỉ dựa trên độ phức tạp của kiến trúc.
