# Error & Confidence Analysis — LAB 08

## 1. Error Analysis (Mục 20)

Dưới đây là 8 mẫu lỗi thực tế của mô hình Fine-tuned BERT-Tiny (`T=128`) trên tập IMDb Test (600 mẫu: tổng cộng 41 False Positives và 110 False Negatives): gồm 3 False Positives, 3 False Negatives và 2 ca dự đoán cực kỳ tự tin nhưng sai.

| Text | True | Pred | Confidence | Error type | Hypothesis |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `[idx=518]` *"It's hard to know what is the point in UP AT THE VILLA, a gorgeous but shallow period piece, one of those made with the only objective of earning Oscar nominations for best costume design..."* | Negative (0) | Positive (1) | 0.9791 | `ambiguous sentiment` | Phim bị chê rỗng tuếch (*"shallow"*), nhưng câu nhắc nhiều cụm khen bề ngoài (*"gorgeous"*, *"Oscar nominations"*, *"best costume design"*), khiến model nghiêng về Positive. |
| `[idx=133]` *"...the film I saw was absolutely dire, with wooden acting... one of the few plus points has to be the stunning scenery... great sweeping vistas and dramatic skies... best to avoid."* | Negative (0) | Positive (1) | 0.8965 | `long-range dependency` | Đoạn đầu chê diễn xuất (*"dire, wooden acting"*), đoạn giữa lại khen cảnh đẹp (*"stunning scenery, great sweeping vistas"*), còn câu chốt *"best to avoid"* nằm ở cuối bị cắt bởi giới hạn 128 token. |
| `[idx=17]` *"This could have been a breakout role for Valeria Golino but the film instead decided to shift its attention to another area. The film is about a woman named Grazia..."* | Negative (0) | Positive (1) | 0.7716 | `context` | Phần đầu kể tóm tắt nội dung trung tính và có cụm giả định *"could have been a breakout role"* mang từ tích cực nên model chưa nhận ra lời chê. |
| `[idx=187]` *"I find it sad that just because Edward Norton did not want to be in the film... people automatically think the movie sucks... It wasn't mind blowing... but it was not a bad. It was fun to watch."* | Positive (1) | Negative (0) | 0.9636 | `negation` | Mở đầu có nhiều từ tiêu cực (*"sad"*, *"sucks"*) và dùng phủ định kép (*"not a bad"*) để khen phim xem giải trí tốt, làm model gom nhầm thành Negative. |
| `[idx=237]` *"There's a legion of Mick Garris haters out there who feel he couldn't direct a horror film of quality... I like SLEEPWALKERS... Forget character or plot development, who needs them anyway."* | Positive (1) | Negative (0) | 0.9734 | `sarcasm` | Người viết thuật lại lời chê của *"haters"* và khen phim kinh dị theo kiểu đùa cợt (*"forget character or plot development"*), model hiểu theo nghĩa đen thành chê bai. |
| `[idx=28]` *"I also saw this at the cinema in the 80s and have never forgotten it... comparing it to some of the terrible modern films... I should think it would stand up very well."* | Positive (1) | Negative (0) | 0.8701 | `negation` | Câu dùng phủ định để khen (*"never forgotten"*) và nhắc tới *"terrible modern films"* để so sánh, khiến model bắt nhầm từ *"terrible"* áp cho phim đang đánh giá. |
| `[idx=511]` *"Summary: Not worth the film... Luckily, the film on its own was a wonderful piece... However, I consider the six hours I spent watching the 'Scarlett' miniseries to be some of the worst-spent hours..."* | Negative (0) | Positive (1) | **0.9928** | `long-range dependency` | Người viết khen phim gốc (*"wonderful piece"*, *"good story"*, *"enjoyable"*) ở nửa đầu để làm nền rồi mới chê nặng bản miniseries ở nửa sau, làm phần đầu kéo confidence lên 99.28%. |
| `[idx=106]` *"I honestly didn't think at first that this movie would make me laugh like hell and admire the great gore effects, boy was I wrong! ... accompanied by spoofs from 'Evil Dead' and 'Gremlins'."* | Positive (1) | Negative (0) | **0.9803** | `domain-specific expression` | Mở đầu bằng đảo ngữ phủ định (*"didn't think... boy was I wrong!"*) kèm nhiều từ vựng dòng phim kinh dị máu me (*"hell"*, *"gore"*, *"splatter"*), khiến model đoán sai với độ tự tin 98.03%. |

### Nhận xét nhanh
- **False Positives:** Hay gặp khi bài chê vẫn khen vài khía cạnh phụ (trang phục, cảnh quay, bản phim gốc) hoặc kể nội dung trước khi chê ở cuối (vượt quá 128 token).
- **False Negatives:** Xảy ra nhiều ở review phim kinh dị/hành động dùng từ bạo lực để khen, hoặc câu khen dùng phủ định kép (*"not bad"*, *"never forgotten"*).

---

## 2. Confidence Analysis (Mục 21)

Kết quả chia 600 dự đoán trên IMDb Test theo `confidence = max(softmax(logits))` (khớp với `confidence_analysis.png`):

| Confidence Tier | Số lượng | Đúng | Sai | Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| Low (`0.50 – 0.70`) | 142 | 80 | 62 | 56.34% |
| Medium (`0.70 – 0.90`) | 192 | 140 | 52 | 72.92% |
| High (`>= 0.90`) | 266 | 229 | 37 | 86.09% |

**Confidence cao có đồng nghĩa với prediction đúng không?**
Không ($\text{confidence} \neq \text{correctness}$). Dù nhóm High confidence (`>= 0.90`) có độ chính xác cao hơn hai nhóm còn lại (86.09% so với 72.92% và 56.34%), vẫn có tới **37/266 ca** model tự tin trên 90% nhưng đoán sai (như `[idx=511]` ở mức 99.28% và `[idx=106]` ở mức 98.03%). Softmax thường khuếch đại chênh lệch logits khi gặp vài từ khóa quen thuộc mạnh, nên độ tự tin cao chỉ phản ánh model thấy tín hiệu khớp với phân bố lúc train chứ không đảm bảo dự đoán đúng.

---

## 3. Câu hỏi bắt buộc (Mục 28)

### Q1. Model nào hưởng lợi nhiều hơn khi training data ít?
- **Observation:** Ở mốc 10% (120 mẫu), Pretrained BERT-Tiny giữ được hiệu năng trên mức ngẫu nhiên, còn Transformer train từ đầu sụp đổ.
- **Evidence:** Từ `results.csv` (Exp 3), tại `10%` data, BERT-Tiny đạt `Acc = 0.5733, F1 = 0.5803`, trong khi Transformer from scratch chỉ đạt `Acc = 0.5000, F1 = 0.0196` (gần như chỉ đoán 1 lớp).
- **Interpretation:** Pretrained model hưởng lợi rõ rệt khi ít dữ liệu vì đã có sẵn biểu diễn ngôn ngữ từ pretraining, còn model train từ đầu thiếu nền tảng ngữ nghĩa nên không đủ tín hiệu để hội tụ với 120 mẫu.

### Q2. Nếu pretrained model tốt hơn from-scratch, điều đó chứng minh điều gì?
- **Observation:** Ở `T=128` (1,200 mẫu), Pretrained BERT-Tiny đạt Accuracy và Precision cao hơn From-Scratch (`Acc = 0.7483` vs `0.7300`; `Prec = 0.8225` vs `0.7066`), nhưng Recall thấp hơn khiến F1 tạm thấp hơn (`0.7156` vs `0.7445`); khi tăng lên `T=256`, BERT-Tiny vượt trội cả hai (`Acc = 0.7750, F1 = 0.7840`).
- **Evidence:** BERT-Tiny tách đúng 259/300 mẫu Negative và đạt độ ổn định mượt qua các mốc dữ liệu `10% -> 100%`, còn From-Scratch dao động mạnh ở các mốc thấp.
- **Interpretation:** Điều này chứng minh biểu diễn khởi tạo từ pretraining giúp mô hình thích nghi với bài toán downstream ổn định và ít lệ thuộc từ khóa rời rạc hơn trên cùng dữ liệu, nhưng không chứng minh model "hiểu ngôn ngữ hoàn toàn" vì vẫn còn 110 False Negatives ở `T=128`.

### Q3. Nếu frozen model gần bằng fine-tuned model, điều đó có ý nghĩa gì?
- **Observation:** Trên IMDb Test, chế độ Frozen encoder đạt `Acc = 0.6967, F1 = 0.6884` — thấp hơn Fine-tuning (`Acc = 0.7483, F1 = 0.7156`) nhưng vẫn vượt xa mức ngẫu nhiên 50% dù chỉ train 258 tham số ở Linear head.
- **Evidence:** Khoảng cách F1 giữa Frozen (`0.6884`) và Fine-tuning (`0.7156`) chỉ chênh 2.72%, trong khi thời gian train Frozen nhanh gần gấp đôi (`18.62s` vs `36.56s`).
- **Interpretation:** Khi frozen đã tiệm cận fine-tuning, điều đó cho thấy vector `[CLS]` từ pretraining vốn đã chứa sẵn các đặc trưng ngữ nghĩa tách biệt tương đối tốt cho bài toán hiện tại, nên một bộ phân loại tuyến tính đơn giản phía trên cũng khai thác được phần lớn tín hiệu.

### Q4. Nếu fine-tuning tốt hơn frozen model rất nhiều, điều đó gợi ý điều gì?
- **Observation:** Khi mở khóa toàn bộ encoder, Accuracy tăng từ `0.6967` lên `0.7483` (+5.16%) và Precision tăng từ `0.7077` lên `0.8225` (+11.48%).
- **Evidence:** Cập nhật 4,386,178 tham số thay vì chỉ 258 tham số ở head giúp hiệu năng tách lớp cải thiện rõ rệt trên cùng tập test 600 mẫu.
- **Interpretation:** Điều này gợi ý biểu diễn tổng quát học từ MLM lúc pretraining chưa tối ưu cho ranh giới cảm xúc đặc thù của review phim; cần điều chỉnh các tầng attention bên trong encoder để tập trung đúng vào cấu trúc khen/chê và phủ định của downstream task.

### Q5. Nếu performance giảm mạnh trên domain khác, nguyên nhân có thể là gì?
- **Observation:** Đem model fine-tune trên IMDb sang đánh giá zero-shot trên tập khác thì hiệu năng giảm, đặc biệt giảm rất sâu ở tin tài chính.
- **Evidence:** Trong Exp 5 (`results.csv`), IMDb in-domain đạt `Acc = 0.7483, F1 = 0.7156`; sang Amazon Product Reviews giảm nhẹ về Accuracy `0.7320` (`F1 = 0.7320`), nhưng sang Financial News rơi mạnh xuống `Acc = 0.5423, F1 = 0.5402` (Recall chỉ còn `0.4772`).
- **Interpretation:** Nguyên nhân do phân bố từ vựng, văn phong và cách thể hiện cảm xúc khác biệt hoàn toàn (domain shift): Amazon vẫn là review khen/chê nên từ khóa cảm xúc còn gần gũi, còn tin tài chính ngắn gọn và dùng thuật ngữ chuyên ngành (*bullish*, *bearish*, *price target*, *cuts to Sell*) mà model đã adapt cho phim ảnh không nhận diện được.

### Q6. Nếu sequence length tăng nhưng performance gần như không đổi trong khi training time tăng mạnh, có nên tiếp tục tăng sequence length không?
- **Observation:** Trong thực nghiệm, tăng `T` giúp model cải thiện rõ ở chặng `64 -> 128 -> 256` (`Acc: 0.7033 -> 0.7483 -> 0.7750`, `F1: 0.6728 -> 0.7156 -> 0.7840`), nhưng thời gian chạy tăng từ `20.50s` lên `36.56s` rồi `63.77s` và peak RSS lên `2325.8 MB`.
- **Evidence:** Số phần tử ma trận attention $T^2$ tăng gấp 16 lần (`4,096 -> 65,536`), làm thời gian train tăng hơn 3.1 lần dù mới ở độ dài 256.
- **Interpretation:** Nếu bước tăng tiếp theo không còn mang lại cải thiện tương xứng (hiệu suất bão hòa) trong khi chi phí $O(T^2)$ tăng vọt thì **không nên** tiếp tục tăng `T`. Khi đó nên chọn độ dài vừa đủ giữ phần thông tin chính của văn bản để cân bằng giữa accuracy và chi phí tính toán.
