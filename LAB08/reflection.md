# Reflection — LAB 08

Biết cách tự ghép Transformer ở LAB 07 giúp mình hiểu kiến trúc bên trong, nhưng ngoài thực tế người ta ít khi train Transformer từ đầu cho một bài toán mới vì ba lý do chính sau:

Thứ nhất là vấn đề **data** và **pretrained representation**. Một Transformer khởi tạo ngẫu nhiên phải tự học từ đầu cả nghĩa từ, ngữ pháp lẫn quy luật phân loại chỉ từ tập dữ liệu gán nhãn có hạn. Khi lượng **data** giảm xuống còn 10% (120 mẫu), mô hình train từ đầu gần như sụp đổ hoàn toàn (`Acc = 0.50, F1 = 0.02`). Ngược lại, BERT đã học trước trên hàng tỷ token không nhãn qua Masked Language Modeling, tạo sẵn **pretrained representation** giàu ngữ cảnh. Khi áp dụng vào **downstream task** như phân loại cảm xúc IMDb, mô hình đã có nền ngôn ngữ sẵn nên cần ít dữ liệu hơn nhiều mà đường cong hiệu năng vẫn tăng ổn định.

Thứ hai là **computational cost**. Chi phí phần cứng và thời gian để tự pretrain một mô hình lớn từ con số 0 rất đắt đỏ do độ phức tạp attention $O(T^2)$. Dùng checkpoint có sẵn rồi thực hiện **fine-tuning** vài epoch với learning rate nhỏ vừa tiết kiệm tài nguyên, vừa đạt độ chính xác cao hơn hẳn so với train từ đầu hay giữ encoder đóng băng.

Tuy nhiên, pretrained model vẫn có nhiều **limitation**, điển hình nhất là **domain shift**. Khi mang mô hình vừa fine-tune trên review phim IMDb sang đánh giá tin tài chính có văn phong và thuật ngữ riêng, độ chính xác tụt mạnh từ `74.83%` xuống `54.23%`. Ngoài ra còn có giới hạn về độ dài chuỗi và tình trạng model tự tin cao nhưng vẫn đoán sai. Vì vậy, hiểu cấu trúc từ LAB 07 là nền tảng, còn làm thực tế thì bắt đầu từ pretrained Transformer, **fine-tuning** hợp lý và kiểm tra lỗi kỹ vẫn hiệu quả nhất.
