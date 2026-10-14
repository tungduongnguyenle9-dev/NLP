# Reflection

Qua bài lab, em nhận thấy cách biểu diễn văn bản có ảnh hưởng đến kết quả phân loại. Với bộ dữ liệu IMDB, TF-IDF cho kết quả tốt hơn Average Word2Vec. Khi thêm bigram, số lượng đặc trưng tăng mạnh nhưng F1 gần như không cải thiện, trong khi thời gian train lâu hơn nhiều. Điều này cho thấy thêm nhiều đặc trưng chưa chắc làm mô hình tốt hơn.

Một số câu có phủ định hoặc nhiều ý vẫn dễ bị dự đoán sai vì các cách biểu diễn hiện tại chưa nắm bắt tốt thứ tự và ngữ cảnh của từ. Đây là lý do cần chuyển sang các mô hình có khả năng xử lý chuỗi và ngữ cảnh tốt hơn như RNN, LSTM hoặc GRU.
