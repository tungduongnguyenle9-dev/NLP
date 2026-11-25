# Phân tích kết quả & Lỗi mô hình — LAB 09

## 1. Experiment 4 — Phân tích hiệu năng theo loại Entity (Mục 21)

Kết quả entity-level của `distilbert-base-cased` trên tập test CoNLL-2003 (`entity_metrics.csv`, tổng 5,648 thực thể):

| Entity | Precision | Recall | F1 | Support (Test) | Số mẫu trong tập train (6,000 câu) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PER** | 0.9502 | 0.9326 | **0.9413** | 1,617 | 2,793 |
| **LOC** | 0.8794 | 0.9047 | **0.8918** | 1,668 | 3,091 |
| **ORG** | 0.8152 | 0.8338 | **0.8244** | 1,661 | 2,635 |
| **MISC** | 0.7089 | 0.7493 | **0.7285** | 702 | 1,509 |

Nguyên nhân chênh lệch giữa 4 nhóm entity:
1. **Số lượng training examples:** `LOC` (3,091), `PER` (2,793) và `ORG` (2,635) có dữ liệu train gần gấp đôi `MISC` (1,509), nên `MISC` học kém ổn định nhất.
2. **Lexical patterns:** `PER` có khuôn từ vựng đều (1–2 từ viết hoa đi cạnh động từ hoặc chức danh), `LOC` nhiều quốc gia/thành phố lặp lại; còn `MISC` gộp nhiều loại không đồng nhất (quốc tịch, giải đấu, múi giờ, giải thưởng).
3. **Ambiguity:** `ORG` và `LOC` nhập nhằng mạnh khi tên thành phố/địa danh được dùng làm tên đội bóng. Trên `confusion_matrix.png`, có **104 token `B-ORG` nhầm thành `B-LOC` (6.3%)**, **58 `B-LOC` nhầm thành `B-ORG` (3.5%)**, và **31 `I-LOC` nhầm thành `I-ORG` (12.1%)**.
4. **Context:** Trong câu tường thuật đầy đủ (*"meeting in Barcelona"*), ngữ cảnh giúp phân biệt rõ `LOC`/`ORG`. Nhưng trong bảng tỷ số hoặc tiêu đề rút gọn thiếu động từ, ngữ cảnh bị thu hẹp nên dễ đoán nhầm.
5. **Entity boundary:** `PER` thường ngắn (1–2 từ) nên giữ biên rất tốt (`I-PER` recall đạt **98.4%**). Ngược lại, `ORG` và `MISC` hay dài 3–4 từ, chứa năm hoặc danh từ thường nên dễ đứt biên (`I-MISC` bị rơi về `O` tới **24.1%**).
6. **Domain:** Dữ liệu Reuters 1996 có nhiều tiêu đề thể thao viết **IN HOA TOÀN BỘ (`ALL CAPS`)**, làm mất dấu hiệu viết hoa tên riêng và khiến tokenizer `distilbert-base-cased` chia nhỏ từ thành nhiều subword lẻ.

---

## 2. Experiment 5 — Phân tích theo độ dài Entity (Mục 22)

Kết quả entity-level theo số token (`entity_length_analysis.png`):

| Entity length | Gold Count | Pred Count | TP | Precision | Recall | F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 token** | 3,574 | 3,687 | 3,102 | 0.8413 | 0.8679 | **0.8544** |
| **2 tokens** | 1,776 | 1,764 | 1,586 | 0.8991 | 0.8930 | **0.8960** |
| **3 tokens** | 236 | 230 | 189 | 0.8217 | 0.8008 | **0.8112** |
| **4+ tokens** | 62 | 63 | 51 | 0.8095 | 0.8226 | **0.8160** |

**Đối chiếu `Prediction 2`:**

* **Đúng với dự đoán ở cụm dài:** Nhóm **3 tokens (F1 = 0.8112)** và **4+ tokens (F1 = 0.8160)** thấp hơn rõ so với nhóm ngắn (giảm ~4–8% F1) vì dễ sai biên ở đầu/cuối cụm.
* **Điểm khác dự đoán:** Nhóm **2 tokens (0.8960)** cao hơn nhóm **1 token (0.8544)** vì phần lớn cụm 2 tokens là `Họ + Tên` thuộc lớp `PER` (lớp dễ nhất, F1 = 0.9413), còn nhóm 1 token chứa nhiều tên viết tắt, mã đội bóng và từ `ALL CAPS` dễ đoán nhầm hoặc dư (Precision chỉ 0.8413).

---

## 3. Experiment 6 — Error Analysis (Mục 23)

15 lỗi thực tế trên tập test CoNLL-2003 (`5 False Positives`, `5 False Negatives`, `5 Boundary Errors`):

| Nhóm | Text | Gold | Prediction | Error type | Explanation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| FP 1 | `Defender Hassan Abbas rose to intercept a long ball...` | `O` (`Defender`) | `ORG` (`Defender`) | `spurious entity` / `context-dependent entity` | Vị trí thi đấu viết hoa đầu câu (`freq=0` trong train), đứng ngay trước tên cầu thủ nên model đoán nhầm là tên CLB. |
| FP 2 | `Centre Jason Little will miss Australia 's end-of-tour fixture...` | `O` (`Centre`) | `PER` (`Centre`) | `spurious entity` / `context-dependent entity` | Vị trí `Centre` viết hoa đầu câu trước `Jason Little`, bị nhận nhầm thành một phần tên người. |
| FP 3 | `Japan began the defence of their Asian Cup title ... in a Group C championship match on Friday .` | `O` (`Group C`) | `MISC` (`Group C`) | `spurious entity` | Cụm `Group C` viết hoa hai chữ đứng cạnh giải đấu nên model đoán dư thành `MISC`. |
| FP 4 | `Tour manager Clive Lloyd on Wednesday apologised...` | `O` (`Tour`) | `ORG` (`Tour`) | `spurious entity` | Từ `Tour` viết hoa đầu câu trước `manager`, dễ bị hiểu nhầm là tên tổ chức. |
| FP 5 | `" FIFA has named the Liberian for its 1996 Fair Play award ... "` | `O` (`Fair Play`) | `MISC` (`Fair Play`) | `spurious entity` / `ambiguous entity` | Tên giải thưởng viết hoa hai chữ đi cùng `1996 ... award`, mô hình nhận thành `MISC` dù gold gán `O`. |
| FN 1 | `SOCCER - LATE GOALS GIVE JAPAN WIN OVER SYRIA .` | `LOC` (`JAPAN`) | `O` | `missing entity` / `subword alignment` | Tiêu đề `ALL CAPS`, `JAPAN` có `freq=0` và bị tách thành `['J', '##AP', '##AN']`, dẫn đến bỏ sót thành `O`. |
| FN 2 | `SOCCER - LEEDS ' BOWYER FINED FOR PART IN FAST-FOOD FRACAS .` | `PER` (`BOWYER`) | `O` | `rare entity` / `subword alignment` | Tên riêng hiếm (`freq=0`) ở dạng chữ hoa toàn bộ, bị chia thành 4 mảnh `['B', '##OW', '##Y', '##ER']`. |
| FN 3 | `RUGBY UNION - LITTLE TO MISS CAMPESE FAREWELL .` | `PER` (`CAMPESE`) | `O` | `rare entity` / `subword alignment` | `CAMPESE` hiếm (`freq=0`), bị tách thành `['CA', '##MP', '##ES', '##E']` trong tiêu đề ngắn viết hoa hết. |
| FN 4 | `Astle 9-0-53-1 ( w-1 nb-1 )` | `PER` (`Astle`) | `O` | `rare entity` / `context-dependent entity` | Dòng thông số cricket chỉ có số liệu kỹ thuật, `Astle` lại hiếm (`freq=0`, tách `As`, `##tle`) nên thiếu ngữ cảnh để nhận `PER`. |
| FN 5 | `[ CORRECTED 14:06 GMT ]` | `MISC` (`GMT`) | `O` | `missing entity` | Nằm trong chuỗi ghi chú thời gian `[ ... ]`, model xem toàn bộ là metadata (`O`) thay vì tách `GMT` là `MISC`. |
| Boundary 1 | `Cuttitta announced his retirement after the 1995 World Cup , ...` | `MISC` (`1995 World Cup`) | `MISC` (`World Cup`) | `wrong boundary` | Bỏ sót năm `1995` ở đầu cụm vì số năm thường đứng ngoài thực thể. |
| Boundary 2 | `... at a McDonald 's fast-food restaurant .` | `ORG` (`McDonald 's`) | `ORG` (`McDonald`) | `wrong boundary` | Dừng sớm trước `'s` do sở hữu cách thông thường mang nhãn `O`, trong khi gold gộp `'s` vào tên hãng. |
| Boundary 3 | `... was set to make his Sheffield Wednesday comeback against Liverpool...` | `ORG` (`Sheffield Wednesday`) | `ORG` (`Sheffield`) + `MISC` (`Wednesday`) | `wrong boundary` / `ambiguous entity` | Chữ `Wednesday` trùng tên thứ trong tuần nên bị cắt tách khỏi tên CLB `Sheffield Wednesday`. |
| Boundary 4 | `Australia beat West Indies ... at the Melbourne Cricket Ground on Friday .` | `LOC` (`Melbourne Cricket Ground`) | `LOC` (`Melbourne Cricket`) + `ORG` (`Ground`) | `wrong boundary` / `wrong entity type` | Cụm địa danh 3 từ bị cắt đôi: 2 từ đầu gán `LOC`, từ cuối `Ground` bị tách thành `ORG`. |
| Boundary 5 | `Semifinals ( on Saturday ) : Fung Permadi ( Taiwan ) v Indra` | `PER` (`Fung Permadi`) | `ORG` (`Fung`) + `PER` (`Permadi`) | `wrong boundary` / `subword alignment` | Tên vận động viên châu Á ít gặp, bị tách thành `Fun`, `##g`, `Per`, `##mad`, `##i`; model tách nhầm thành `ORG` + `PER`. |

---

## 4. Context Analysis (Mục 24)

4 ví dụ thực tế trong tập test CoNLL-2003 có **cùng surface form nhưng khác loại entity**, mô hình đều dự đoán đúng theo ngữ cảnh:

1. **`Barcelona`** (`LOC` vs `ORG`)
    * **Test #146 (`LOC`):** `"FIFA 's players ' status committee , meeting in Barcelona , ..."` → Gold: `LOC` | Pred: **`B-LOC`** (địa điểm họp).
    * **Test #3417 (`ORG`):** `"Real Madrid 2 Barcelona 0"` → Gold: `ORG` | Pred: **`B-ORG`** (đội bóng trong bảng tỷ số).

2. **`Philadelphia`** (`LOC` vs `ORG`)
    * **Test #428 (`LOC`):** `"A 39-yard interference penalty against Philadelphia 's Troy Vincent ..."` → Gold: `LOC` | Pred: **`B-LOC`**.
    * **Test #422 (`ORG`):** `"The loss by Philadelphia allowed the idle Green Bay Packers ( 10-3 ) ..."` → Gold: `ORG` | Pred: **`B-ORG`**.

3. **`Leicester`** (`LOC` vs `ORG`)
    * **Test #202 (`LOC`):** `"... 3 - Darren Garforth ( Leicester ) , 2 - Norm Hewitt ( New Zealand ) ..."` → Gold: `LOC` | Pred: **`B-LOC`** (địa phương trong ngoặc).
    * **Test #2878 (`ORG`):** `"Leicester 1 ( Marshall 78 ) Blackburn 1 ( Sutton 34 ) ."` → Gold: `ORG` | Pred: **`B-ORG`** (CLB thi đấu).

4. **`WASHINGTON`** (`LOC` vs `ORG`)
    * **Test #1792 (`LOC`):** `"WASHINGTON 1996-12-06"` → Gold: `LOC` | Pred: **`B-LOC`** (nơi phát tin).
    * **Test #544 (`ORG`):** `"WASHINGTON AT TAMPA BAY"` → Gold: `ORG` | Pred: **`B-ORG`** (đội thể thao).

### Lợi thế của Contextual Representation so với Static Embedding

* **LAB 03 (Static Embedding):** Mỗi từ chỉ có đúng **một vector cố định**, nên `Barcelona` hay `WASHINGTON` luôn có cùng biểu diễn dù là địa danh hay đội bóng — rất dễ nhầm về nhãn phổ biến nhất.
* **LAB 08 → LAB 09 (Contextual Representation → Token-level Prediction):** Nhờ self-attention hai chiều, vector $h_i$ của mỗi token thay đổi theo toàn bộ câu xung quanh (*"meeting in ... "* nghiêng về `LOC`, còn đứng cạnh tỷ số nghiêng về `ORG`). Đây là lý do chính giúp Transformer đạt Entity F1 = **0.8652**, vượt xa Baseline dùng đặc trưng từ cục bộ (**0.7138**).
