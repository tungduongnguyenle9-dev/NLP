# Dự đoán trước thực nghiệm — LAB 07

Ghi lại các suy đoán cá nhân trước khi chạy code thực nghiệm theo Mục 8 và đối chiếu lại kết quả ở Mục 28.

---

## 1. Dự đoán ban đầu (Trước khi chạy)

### Prediction 1
* **Câu hỏi:** Một Transformer có thể xử lý các token trong sequence song song hơn RNN không?
* **Prediction:** Có.
* **Reason:** Em thấy bảo Transformer tính toán bằng phép nhân ma trận trên toàn bộ câu luôn chứ không phải chạy vòng lặp tuần tự từng từ một như RNN, nên chắc chắn là tận dụng phần cứng để tính song song tốt hơn nhiều.
* **Confidence:** High

---

### Prediction 2
* **Câu hỏi:** Nếu bỏ positional encoding, Transformer có còn biết thứ tự token không?
* **Prediction:** Không.
* **Reason:** Cơ chế self-attention chỉ so sánh độ tương quan giữa các cặp từ với nhau. Nếu không có gì đánh dấu vị trí thì đảo từ ra chỗ khác phép nhân vẫn ra vậy, mô hình sẽ không biết từ nào đứng trước từ nào đứng sau.
* **Confidence:** High

---

### Prediction 3
* **Câu hỏi:** Nếu hai câu có cùng tập token nhưng khác thứ tự, Transformer không có positional information có thể phân biệt chúng không?
* **Prediction:** Không.
* **Reason:** Vì hai câu có cùng tập từ ngữ, không có vị trí thì mô hình coi như một túi từ. Sau khi gom lại qua pooling chắc chắn vector đại diện cho hai câu sẽ giống nhau, nên mô hình không thể phân biệt được.
* **Confidence:** High

---

### Prediction 4
* **Câu hỏi:** Hai attention heads có nhất thiết học cùng một attention pattern không?
* **Prediction:** Không nhất thiết.
* **Reason:** Chia ra nhiều head là để mỗi head phụ trách quan sát một kiểu quan hệ khác nhau (ví dụ head nhìn từ lân cận, head nhìn từ cảm xúc). Nếu head nào cũng học y hệt nhau thì chia nhiều head làm gì.
* **Confidence:** Medium

---

### Prediction 5
* **Câu hỏi:** Tăng số attention heads có chắc chắn làm accuracy tăng không?
* **Prediction:** Chắc chắn có.
* **Reason:** Càng có nhiều head thì mô hình càng nhìn câu dưới nhiều góc độ khác nhau, lại có thêm nhiều bộ trọng số để học nên em đoán là điểm accuracy sẽ tăng lên.
* **Confidence:** Medium

---

### Prediction 6
* **Câu hỏi:** Transformer có luôn nhanh hơn RNN trên mọi sequence length không?
* **Prediction:** Không.
* **Reason:** Vì ma trận attention có kích thước là $T \times T$, câu mà dài ra thì số phép tính tăng theo bình phương nên với mấy câu dài ngoằng chắc Transformer sẽ bị chậm đi hoặc ngốn RAM.
* **Confidence:** Medium

---

## 2. Bảng nghiệm chứng sau thực nghiệm (Mục 28)

Sau khi chạy xong các thực nghiệm và lấy số liệu từ `results.csv`, em đối chiếu lại các dự đoán ban đầu:

| Prediction | Dự đoán ban đầu | Kết quả thực tế | Đánh giá | Giải thích thực tế |
|:---:|:---|:---|:---:|:---|
| **1** | Có, xử lý song song hơn RNN | Đúng | **Đúng** | Transformer tính toàn bộ chuỗi cùng lúc qua ma trận, không cần chạy vòng lặp theo từng timestep như RNN. |
| **2** | Không biết thứ tự nếu bỏ PE | Đúng | **Đúng** | Khi không có PE, mô hình coi chuỗi như một tập từ rời rạc. |
| **3** | Không phân biệt được "dog bites man" vs "man bites dog" nếu thiếu PE | Đúng | **Đúng** | Chạy thực nghiệm hoán vị: không có PE thì cosine similarity ra đúng **1.000000** (vector giống hệt nhau). Có PE thì khoảng cách L2 mới tách ra được $0.3846$. |
| **4** | Không nhất thiết học giống nhau | Đúng | **Đúng** | Xem hình `attention_heads.png` thấy rõ: Head 1 chú ý vào từ liền kề, các Head khác nhìn tản ra hoặc nhìn vào từ khóa chính. |
| **5** | Chắc chắn có (accuracy sẽ tăng) | Không chắc chắn tăng | **SAI** | *(Giải thích bên dưới)* |
| **6** | Không phải lúc nào cũng nhanh hơn | Đúng | **Đúng** | Khi sequence dài 256, thời gian train tăng vọt lên **257.88s** (ma trận attention lên tới 65,536 phần tử), chậm hơn nhiều so với chuỗi ngắn. |

---

### Giải thích dự đoán sai (Prediction 5):
* **Lý do ban đầu em đoán sai:** Em cứ nghĩ đơn giản là càng nhiều head thì mô hình càng "thông minh" hơn và accuracy sẽ tăng đều đặn.
* **Bản chất thực tế sau khi làm bài:** Khi cố định kích thước mô hình $d_{model} = 128$, mình tăng số head lên thì số chiều của mỗi head ($d_k = d_{model} / n_{heads}$) lại bị chia nhỏ ra:
  * 1 head: $d_k = 128$ $\to$ Acc = 0.6972
  * 2 heads: $d_k = 64$ $\to$ Acc = 0.7324
  * 4 heads: $d_k = 32$ $\to$ Acc = 0.7312
  * 8 heads: $d_k = 16$ $\to$ Acc = 0.7428
  
  Khi lên tới 4 heads và 8 heads, $d_k$ chỉ còn 32 và 16 chiều. Không gian biểu diễn của từng head bị hẹp lại, thông tin bị chia vụn ra nên kết quả accuracy chỉ dao động quanh mức $73\% - 74\%$ chứ không tăng vọt như em tưởng. Như vậy tăng head không đồng nghĩa với việc chắc chắn accuracy sẽ tăng.
