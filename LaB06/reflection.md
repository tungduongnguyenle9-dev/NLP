# Reflection — LAB 06

### 1. Attention và RNN/LSTM

Attention có thể lấy thông tin từ nhiều token thay vì chỉ dựa vào hidden state cuối như các mô hình đã học ở buổi trước. Nhờ vậy, thông tin ở xa có thể được chú ý trực tiếp hơn.

### 2. Kết quả thực nghiệm

Khi thay đổi query, các trọng số cũng thay đổi. Khi tăng temperature, trọng số phân bố đều hơn, không còn tập trung quá nhiều vào một token.

### 3. Hạn chế

Self-attention tốn tài nguyên khi sequence dài. Ngoài ra, attention weights không chứng minh được model thực sự hiểu ngôn ngữ.

### 4. So sánh mô hình

Trong kết quả chạy, Attention cao hơn LSTM 13,34 điểm phần trăm về Accuracy và khoảng 9,55 điểm phần trăm về F1. Đổi lại, thời gian huấn luyện dài hơn 3,45 giây, dù số tham số ít hơn 5.280. Vì vậy, thời gian chạy không chỉ phụ thuộc vào số tham số mà còn vào cách mô hình xử lý chuỗi. Bảng kết quả cho thấy Attention hoạt động tốt hơn, nhưng một lần chạy không đủ để kết luận nó luôn tốt hơn LSTM hoặc xác định chính xác nguyên nhân chênh lệch. Với chuỗi dài hơn, ma trận self-attention tăng theo bình phương độ dài chuỗi, nên cần cân nhắc chi phí tính toán và bộ nhớ.

### 5. Kết luận

Điểm quan trọng nhất là cách Q, K, V tạo ra context vector. Phần cần tìm hiểu thêm là cách Transformer sử dụng nhiều attention head và thông tin vị trí.
