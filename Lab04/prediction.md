# Prediction trước experiment

### Prediction 1

1. **Prediction:** TF-IDF + Logistic Regression có khả năng hoạt động tốt.
2. **Reason:** Vì TF-IDF bắt từ khóa đặc trưng tốt, còn LR xử lý tốt việc phân loại trên không gian từ vựng khổng lồ do TF-IDF tạo ra.
3. **Confidence:** High

### Prediction 2

1. **Prediction:** Average Word2Vec không chắc chắn tốt hơn TF-IDF.
2. **Reason:** Average Word2Vec giữ thông tin ngữ nghĩa, nhưng việc lấy trung bình có thể làm mất các từ quan trọng và trật tự từ.
3. **Confidence:** Cao

### Prediction 3

1. **Prediction:** Tăng vocabulary size không chắc chắn làm accuracy tăng.
2. **Reason:** Thêm quá nhiều từ vựng có thể kéo theo cả những từ hiếm và từ rác.
3. **Confidence:** High

### Prediction 4

1. **Prediction:** Bỏ stopwords không chắc chắn làm performance tăng.
2. **Reason:** Một số stopwords có thể mang thông tin quan trọng cho classification, đặc biệt là các từ phủ định như “not”.
3. **Confidence:** High

### Prediction 5

1. **Prediction:** Accuracy và F1 có thể cho kết luận khác nhau.
2. **Reason:** Accuracy đo tỷ lệ dự đoán đúng tổng thể, còn F1 cân bằng giữa Precision và Recall nên có thể rất khác biệt khi dataset mất cân bằng.
3. **Confidence:** High
