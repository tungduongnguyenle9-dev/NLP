# Reflection & Individual Learning Check — LAB 09

## Phần 1: Reflection (Mục 32)

> **Câu hỏi:** Tại sao cùng một pretrained Transformer có thể được sử dụng cho cả sequence classification và token classification chỉ bằng cách thay đổi phần prediction head?

Một pretrained Transformer có thể dùng chung cho cả sequence classification và token classification vì phần thân (backbone) của nó không học riêng một bài toán đầu ra cố định, mà chỉ làm một việc cốt lõi: tạo ra **biểu diễn theo ngữ cảnh (contextual representation)** cho từng token.

Khi một chuỗi gồm $T$ token đi qua các tầng self-attention, đầu ra cuối cùng của Transformer là ma trận $H \in \mathbb{R}^{T \times d}$. Ở đây, mỗi token $i$ có một vector $h_i \in \mathbb{R}^d$ đã tổng hợp thông tin từ toàn bộ các từ xung quanh theo cả hai chiều. Vì ma trận $H$ đã chứa đầy đủ ngữ nghĩa ở cả mức từng từ lẫn mức cả câu, bài toán đầu ra thực chất chỉ do **prediction head** và cách định nghĩa nhãn quyết định:

* **Sequence Classification (LAB 08):** Mục tiêu là gán một nhãn duy nhất cho cả câu (như `POSITIVE` / `NEGATIVE`). Lúc này, ta chỉ lấy vector đại diện của token đầu chuỗi `[CLS]` (vốn đã gom ngữ cảnh của cả câu qua self-attention) rồi đưa qua một lớp tuyến tính để dự đoán ra một vector logits chung cho cả chuỗi.
* **Token Classification (LAB 09):** Mục tiêu của NER là gán nhãn BIO cho từng token. Thay vì chỉ nhìn `[CLS]`, ta giữ nguyên toàn bộ các vector $h_1, h_2, \dots, h_T$ và áp dụng cùng một lớp tuyến tính ($Z = HW + b \in \mathbb{R}^{T \times C}$) lên từng vị trí. Mỗi token lúc này tự tạo ra phân phối xác suất nhãn riêng (`B-PER`, `I-PER`, `O`, ...). Nhờ $h_i$ đã mang ngữ cảnh xung quanh, cùng một từ như *Barcelona* có thể ra `B-LOC` trong câu chỉ địa điểm và ra `B-ORG` trong bảng tỷ số bóng đá.

Tóm lại, pretrained Transformer đóng vai trò bộ trích xuất đặc trưng ngữ cảnh dùng chung, còn prediction head quyết định cách đọc đầu ra: lấy `[CLS]` cho nhãn cả câu, hoặc lấy từng token cho nhãn mức từ. Nhờ cấu trúc tách bạch này, chỉ cần đổi lớp tuyến tính cuối là ta tái sử dụng được cùng một mô hình nền cho nhiều bài toán khác nhau.

---

## Phần 2: Individual Learning Check (Mục 31)

1. **Sequence classification khác token classification như thế nào?**
   Sequence classification gán một nhãn duy nhất cho cả chuỗi ($X \to y$), còn token classification gán một nhãn riêng cho từng token trong chuỗi ($X=(x_1,\dots,x_T) \to Y=(y_1,\dots,y_T)$).

2. **BIO tagging dùng để làm gì?**
   Dùng để mã hóa đồng thời **loại thực thể** và **ranh giới thực thể** ở mức token: `B-X` là điểm bắt đầu, `I-X` là phần bên trong, `O` là ngoài thực thể.

3. **B-PER khác I-PER như thế nào?**
   `B-PER` đánh dấu token đầu tiên của tên một người, còn `I-PER` đánh dấu các token nối tiếp phía sau. Nhờ đó mô hình tách được hai tên người đứng liền kề nhau thay vì gộp làm một.

4. **Tại sao tokenizer gây ra label alignment problem?**
   Dữ liệu NER gán nhãn theo **từ (word)**, trong khi subword tokenizer của Transformer cắt từ hiếm thành nhiều mảnh con (`subword`) và chèn thêm `[CLS]`, `[SEP]`, làm số token không còn khớp 1–1 với số nhãn gốc.

5. **IGNORE trong loss có ý nghĩa gì?**
   `IGNORE` (`-100`) chỉ bỏ qua vị trí đó khi tính hàm loss, không hề xóa token khỏi chuỗi đầu vào; token đó vẫn có `attention_mask = 1` và vẫn tham gia self-attention bình thường.

6. **Tại sao NER không nên chỉ đánh giá bằng token accuracy?**
   Vì nhãn `O` chiếm phần lớn câu (trên 80%). Một mô hình đoán toàn bộ là `O` vẫn đạt token accuracy rất cao nhưng hoàn toàn không trích xuất được thực thể nào.

7. **Entity-level F1 khác token-level F1 như thế nào?**
   Token-level F1 chấm đúng/sai từng token rời rạc, còn entity-level F1 yêu cầu khớp trọn vẹn cả **loại thực thể** lẫn **ranh giới đầu–cuối**; lệch một token là sai toàn bộ thực thể đó.

8. **Vì sao entity boundary khó?**
   Vì thực thể nhiều từ hay chứa từ thường, số năm hoặc dấu câu (`of`, `1995`, `'s`), và các danh từ chung viết hoa đứng cạnh tên riêng dễ làm mô hình cắt sớm hoặc gộp dư biên.

9. **Vì sao contextual representation hữu ích cho NER?**
   Vì cùng một từ có thể mang nhiều loại thực thể khác nhau tùy ngữ cảnh (ví dụ *Washington* vừa có thể là `PER`, `LOC` hoặc `ORG`). Biểu diễn ngữ cảnh giúp vector của từ thay đổi linh hoạt theo cả câu thay vì cố định như static embedding.

10. **NER có thể được sử dụng như thế nào trong Information Extraction?**
    NER là bước trung gian nhận diện và định vị các thực thể quan trọng (`PERSON`, `ORG`, `LOCATION`, ...) trong văn bản thô để gom thành dữ liệu có cấu trúc (JSON, bảng) phục vụ tìm kiếm hoặc trích xuất quan hệ.
