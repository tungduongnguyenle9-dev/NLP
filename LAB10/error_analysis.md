# Phân tích lỗi — LAB 10 (Extractive QA)

Đánh giá mô hình `distilbert-base-uncased` đã fine-tune (`max_length=384, stride=128, 3 epochs`) trên **SQuAD Test Split (400 mẫu)**:
- **Exact Match (`EM = 1`):** `97 / 400` (`24.25%`)
- **Partial Overlap (`EM = 0, F1 > 0`):** `84 / 400` (`21.00%`)
- **No Overlap (`EM = 0, F1 = 0`):** `219 / 400` (`54.75%`)

---

## 1. Ví dụ `EM = 0` nhưng `F1 > 0` (Mục 20)

Dưới đây là 6 mẫu thực tế cho thấy vì sao F1 (`31.91%`) cao hơn hẳn EM (`24.25%`):

| Question | Gold | Prediction | F1 | Vì sao `EM = 0` nhưng `F1 > 0`? |
| :--- | :--- | :--- | :---: | :--- |
| What helped support the Command Module with a propulsion engine and propellants? | `Service Module (SM)` | `cylindrical Service Module (SM` | 0.857 | Đúng phần lõi nhưng lấy dư tính từ `cylindrical` ở đầu và thiếu dấu `)` ở cuối. |
| With what body must a pharmacy technician register? | `the General Pharmaceutical Council (GPhC) register` | `General Pharmaceutical Council` | 0.857 | Trúng tên cơ quan chính nhưng dừng sớm trước `(GPhC) register`. |
| When was the heyday of secular Arab nationalism? | `between 1945 and 1970` | `1945 and 1970` | 0.857 | Trúng mốc năm nhưng thiếu giới từ `between` ở đầu nên EM = 0. |
| In what does the doctrine of legitimate expectations have roots? | `the principles of legal certainty and good faith` | `legal certainty and good faith` | 0.833 | Bỏ cụm dẫn `the principles of`, đi thẳng vào hai nguyên tắc chính. |
| Who were the sculpture galleries that opened in 2006 named after? | `Dorothy and Michael Hintze` | `Dorothy and Michael Hintze sculpture galleries` | 0.800 | Đúng tên người nhưng kéo dư cụm `sculpture galleries` phía sau. |
| How high are many of the buildings with turrets? | `four stories` | `four` | 0.667 | Đúng số `four` nhưng cắt cụt mất đơn vị `stories`. |

---

## 2. Phân tích 15 lỗi tiêu biểu (Mục 24)

### 2.1. 5 lỗi `wrong start` (đúng `end`, sai `start`)

| Question | Gold | Prediction | Error type | Explanation |
| :--- | :--- | :--- | :--- | :--- |
| What helped support the Command Module with a propulsion engine and propellants? | `Service Module (SM)` | `cylindrical Service Module (SM` | `wrong start` (`tokenization`) | `start` lùi thêm tính từ `cylindrical` phía trước (và bị ngắt ở dấu `)` cuối). |
| What does the First Minister deliver at the beginning of each parliamentary year? | `a statement` | `the First Minister delivers a statement` | `wrong start` (`wrong boundary`) | Dừng đúng tại `statement` nhưng `start` kéo luôn chủ ngữ + động từ phía trước. |
| After rejecting compulsory confession, what did Luther call for? | `private confession and absolution` | `compulsory confession and encouraged private confession and absolution` | `wrong start` (`distractor`) | `end` đúng tại `absolution`, nhưng `start` bị hút nhầm về cụm `compulsory confession` ngay trước đó. |
| What is the name of the process which confirms the primality of a number n? | `trial division` | `primality. A simple but slow method of verifying the primality of a given number n is known as trial division` | `wrong start` (`wrong boundary`) | Dừng đúng ở `trial division` nhưng chưa cắt gọn được tên thuật ngữ nên kéo cả câu định nghĩa. |
| What are the main threats facing the Amazon rainforest in the current century? | `climate change in addition to deforestation` | `deforestation` | `wrong start` (`partial answer`) | Bắt đúng cuối câu `deforestation` nhưng đặt `start` trễ nên bỏ mất vế `climate change`. |

### 2.2. 5 lỗi `wrong end` (đúng `start`, sai `end`)

| Question | Gold | Prediction | Error type | Explanation |
| :--- | :--- | :--- | :--- | :--- |
| Along with Canada and the United Kingdom, what country generally doesn't refer to universities as private schools? | `Australia` | `Australia and Canada` | `wrong end` (`wrong boundary`) | Bắt đúng `Australia` nhưng không dừng lại mà lấy dư `and Canada` (đã có trong câu hỏi). |
| Who was the star of The Sarah Jane Adventures? | `Elisabeth Sladen` | `Elisabeth Sladen who reprised her role as investigative journalist Sarah Jane Smith` | `wrong end` (`wrong boundary`) | Bắt đúng tên diễn viên ở đầu nhưng kéo lố cả mệnh đề quan hệ giải thích vai diễn. |
| How many years have imperialistic practices existed? | `thousands` | `thousands of years` | `wrong end` (`wrong boundary`) | Đúng `thousands` nhưng lấy thừa `of years` vốn đã nhắc trong câu hỏi. |
| When was Manning picked #1 in the NFL Draft? | `1998` | `1998 NFL draft, while Newton was picked first in 2011` | `wrong end` (`distractor`) | `start` trúng `1998`, nhưng `end` kéo sang tận năm `2011` ở vế nói về Newton. |
| What do counties sometimes offer to get more teachers? | `alternative licensing programs` | `alternative licensing` | `wrong end` (`partial answer`) | Đúng đầu cụm nhưng dừng sớm, bị thiếu danh từ chính `programs` ở cuối. |

### 2.3. 5 lỗi `wrong span` (chọn sai vùng/thực thể)

| Question | Gold | Prediction | Error type | Explanation |
| :--- | :--- | :--- | :--- | :--- |
| How many yards was the field goal that made the score 13-7 in Super Bowl 50? | `33` | `Jonathan Stewart` | `distractor` | Bị nhiễu bởi pha ghi điểm khác ở đầu đoạn thay vì con số `33` ở cuối đoạn. |
| What series years are available on Blu-ray? | `series from 2009 onwards` | `Eight original series serials have been released on Laserdisc and many have also been released on Betamax tape and Video 2000` | `long context` | Đoạn văn liệt kê nhiều định dạng cũ (DVD, VHS, Laserdisc, Betamax) khiến model chọn nhầm câu ngay trước câu chứa `Blu-ray`. |
| Who other than Tesla did Westinghouse consider for the patents? | `Galileo Ferraris` | `Thomas Commerford Martin` | `multiple plausible spans` | Đoạn có nhiều tên người; model chưa hiểu ý phủ định *"other than Tesla"* nên bắt nhầm tên xuất hiện sớm ở đầu đoạn. |
| Who came up with the theory of relativity? | `Albert Einstein` | `general relativity` | `question ambiguity` | Bỏ qua từ hỏi *Who* (tên người) mà bám vào từ khóa `theory of` để lấy tên lý thuyết. |
| Why did CBS turn down the pitch for The Untouchables? | `its use of violence` | `ABC` | `distractor` | Cùng câu nhắc 2 nhà đài (`CBS` và `ABC`); model chưa tách rõ quan hệ nhân quả *"because of"* nên chọn nhầm tên đài. |

---

## 3. Thống kê `#wrong start` vs `#wrong end` & Giả thuyết (Mục 25)

Trên tổng số **303 lỗi** (trong 400 mẫu test):
- **`#wrong start`:** **25** (`6.25%`)
- **`#wrong end`:** **21** (`5.25%`)
- **`#wrong span`:** **257** (`64.25%`)

**Giả thuyết vì sao `#wrong start` (25) hơi cao hơn `#wrong end` (21):**
1. **Đầu span dễ bị mờ ranh giới hơn cuối span:** Trước đáp án thường có các từ dẫn ngữ pháp (`between`, `since`, `the principles of`) hoặc cụm chủ–vị (`the First Minister delivers...`), nên `start classifier` hay bị kéo lùi vài token hoặc cắt thiếu từ dẫn.
2. **Cuối span hay trùng điểm dừng tự nhiên:** Nhiều đáp án kết thúc ngay trước dấu phẩy, dấu chấm hoặc cuối cụm danh từ ngắn, giúp `end classifier` dễ xác định điểm dừng hơn một chút.
3. **Tỷ lệ `wrong span` còn cao (257/400)** chủ yếu do model mới fine-tune trên 1,000 mẫu (~1.1% SQuAD train) trong 3 epoch trên CPU nên vẫn dễ nhầm khi đoạn văn có nhiều thực thể bẫy.
