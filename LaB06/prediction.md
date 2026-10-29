# Predictions

## Prediction 1

**Prediction:** Key nào có dot product với query lớn hơn thì attention weight của nó cũng lớn hơn.

**Reason:** Vì score lớn hơn thì sau softmax phần đó sẽ được ưu tiên hơn các key còn lại.

**Confidence:** High

## Prediction 2

**Prediction:** Nếu tất cả score đều bằng 1 thì bốn attention weight sẽ bằng nhau, mỗi cái là 0.25.

**Reason:** Vì Softmax không có lý do để ưu tiên phần tử nào khi tất cả score giống nhau.

**Confidence:** High

## Prediction 3

**Prediction:** Với score [1, 1, 10], attention sẽ gần như tập trung hết vào phần tử thứ ba.

**Reason:** Số 10 lớn hơn hẳn hai số còn lại, nên sau khi lấy hàm mũ thì sự chênh lệch càng lớn.

**Confidence:** Medium

## Prediction 4

**Prediction:** Nếu sequence tăng từ 10 lên 1000 token thì chi phí tính attention tăng  100 lần.

**Reason:** Số token tăng 100 lần nên lượng tính toán có thể tăng tương ứng theo số token.

**Confidence:** Medium

## Prediction 5

**Prediction:** Attention không phải lúc nào cũng chọn đúng từ mà con người thấy quan trọng nhất.

**Reason:** Weight được tính từ các vector và query của model, còn con người đánh giá dựa trên nghĩa của câu nên có thể khác nhau.

**Confidence:** Medium
