# Reflection & Individual Learning Check — LAB 10

## 1. Reflection: LAB 08, LAB 09 và LAB 10 (Mục 34)

```text
                       Pretrained Transformer
                                 │
                 ┌───────────────┼────────────────┐
                 ↓               ↓                ↓
         [CLS] representation  Token reps     Token reps
                 ↓               ↓                ↓
           Classification       NER          Start / End
                 ↓               ↓                ↓
               LAB 08          LAB 09           LAB 10
```

Cả LAB 08, LAB 09 và LAB 10 đều dùng chung một backbone **Pretrained Transformer** (như BERT hoặc DistilBERT). Vai trò của backbone này giống hệt nhau ở cả ba bài: đọc chuỗi token đầu vào và tạo ra các vector biểu diễn có ngữ cảnh (contextual representations) nhờ self-attention. Điểm khác biệt nằm hoàn toàn ở **prediction head** gắn phía trên và **target** dùng để tính loss:

- **LAB 08 — Sequence Classification:** Mục tiêu là gán **1 nhãn** cho cả đoạn văn (ví dụ: tích cực / tiêu cực). Vì chỉ cần thông tin tóm tắt toàn câu, mô hình lấy duy nhất vector `[CLS]` rồi đưa qua một lớp Linear để phân loại. Target là một nhãn lớp duy nhất.
- **LAB 09 — Token Classification (NER):** Mục tiêu là gán **1 nhãn cho từng token** (như `B-PER`, `I-PER`, `O`). Mô hình giữ lại toàn bộ vector đầu ra của từng token và áp dụng cùng một lớp Linear lên từng vị trí. Target là chuỗi nhãn có độ dài bằng số token.
- **LAB 10 — Extractive Question Answering:** Đầu vào ghép chung `(Question, Context)` để self-attention đối chiếu trực tiếp câu hỏi với đoạn văn. Thay vì gán nhãn rời rạc cho từng từ như NER, bài toán cần cắt ra một đoạn chữ liên tiếp (`answer span`). Do đó mô hình dùng **hai bộ phân loại** trên token representations: một đầu dự đoán xác suất điểm bắt đầu $P_{\text{start}}(i)$ và một đầu dự đoán xác suất điểm kết thúc $P_{\text{end}}(j)$ với $i \le j$. Target là cặp chỉ số `(start_position, end_position)` và loss là trung bình cộng hai Cross-Entropy: $L = (L_{\text{start}} + L_{\text{end}})/2$.

Như vậy, Pretrained Transformer không phải là ứng dụng cụ thể mà là bộ mã hóa ngữ cảnh dùng chung. Cùng một backbone nhưng chỉ cần đổi prediction head và target là giải quyết được ba lớp bài toán khác nhau: phân loại chuỗi, phân loại token và dự đoán span.

---

## 2. Phân biệt Extractive QA và RAG (Mục 28)

- **Extractive QA (LAB 10):** `Question + Context → QA Model → Answer span`. Context **đã được đưa sẵn**, mô hình chỉ tìm vị trí `(start, end)` và cắt nguyên văn câu trả lời trong đó.
- **RAG (LAB 11 & LAB 12):** `Question → Retriever → Top-K Relevant Docs (Context) → Generator / QA → Answer`. Chưa biết câu trả lời nằm ở tài liệu nào nên phải có bước **Retriever** tìm đoạn liên quan từ kho tài liệu lớn trước, rồi mới sinh hoặc trích câu trả lời.

---

## 3. Individual Learning Check (Mục 33)

1. **Extractive QA khác classification như thế nào?**  
   Classification chọn 1 nhãn cố định cho cả văn bản (từ `[CLS]`); Extractive QA nhận `(Question, Context)` và dự đoán 2 vị trí `(start, end)` trong context để cắt ra đáp án.
2. **Tại sao QA cần start và end position?**  
   Đáp án là một đoạn chữ liên tiếp có độ dài thay đổi, nên cần cả điểm bắt đầu `i` và điểm kết thúc `j` ($i \le j$) để khoanh đúng đoạn trả lời.
3. **Character offset khác token index như thế nào?**  
   `Character offset` đếm theo từng ký tự trong chuỗi gốc; `token index` đếm theo vị trí subword sau khi tokenize (đã thêm `[CLS]`, question, `[SEP]` và có thể tách 1 từ thành nhiều token).
4. **Tại sao cần offset mapping?**  
   SQuAD cung cấp vị trí theo ký tự còn Transformer tính theo token. `offset_mapping` giúp đổi character span sang token span khi train, và đổi ngược từ token dự đoán về chuỗi gốc khi suy luận.
5. **`stride` dùng để làm gì?**  
   Tạo phần chồng lấn (overlap) giữa 2 window liên tiếp để câu trả lời nằm sát mép cắt không bị đứt đôi.
6. **Vì sao context dài có thể phải chia thành nhiều window?**  
   Transformer có giới hạn độ dài đầu vào (`max_length`) và chi phí $O(L^2)$. Chia nhiều window trượt giúp giữ nguyên question và không mất đáp án ở cuối đoạn văn.
7. **Exact Match đo điều gì?**  
   Đo tỷ lệ câu hỏi mà dự đoán (sau khi chuẩn hóa chữ thường, bỏ dấu câu và `a/an/the`) khớp giống hệt 100% đáp án chuẩn.
8. **F1 của QA khác gì với accuracy?**  
   Accuracy chỉ chấm đúng/sai toàn bộ; Token-level F1 tính theo Precision và Recall trên tập từ trùng nhau nên vẫn cho điểm thành phần (0–1) khi đoán đúng một phần.
9. **Vì sao prediction có thể có `F1 > 0` nhưng `EM = 0`?**  
   Dự đoán trùng một phần từ với đáp án chuẩn nhưng bị dư hoặc thiếu từ ở biên (ví dụ `1945 and 1970` vs `between 1945 and 1970`).
10. **Wrong-start và wrong-end khác nhau như thế nào?**  
    `Wrong-start` là đúng điểm cuối nhưng sai điểm đầu; `wrong-end` là đúng điểm đầu nhưng sai điểm cuối.
11. **Extractive QA khác RAG ở đâu?**  
    Extractive QA có sẵn context và chỉ cắt nguyên văn đáp án; RAG phải tự truy xuất tài liệu liên quan từ kho lớn trước rồi mới trả lời.
12. **Khi nào extractive QA không thể trả lời câu hỏi?**  
    Khi đáp án không có trong context (unanswerable), cần suy luận/tính toán hoặc tổng hợp nhiều câu rời rạc, câu hỏi dạng Có/Không không có nguyên văn, hoặc đáp án bị cắt mất do truncation.
