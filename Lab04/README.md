# Lab 04: Phân loại cảm xúc IMDB

Lab này thử nghiệm một số cách biểu diễn văn bản để phân loại review phim IMDB thành hai nhãn `positive` và `negative`.

## Thử nghiệm

- TF-IDF kết hợp Logistic Regression.
- Average Word2Vec kết hợp Logistic Regression.
- TF-IDF kết hợp Linear SVM.
- So sánh TF-IDF unigram với unigram và bigram.

Dữ liệu được chia thành tập train, validation và test. Các kết quả trong `results.csv` được tính trên tập validation.

## Cách chạy

1. Cài các thư viện cần thiết:

	```bash
	python3 -m pip install numpy pandas matplotlib scikit-learn gensim jupyter
	```

2. Tải `IMDB Dataset.csv` và đặt trong thư mục `Lab04`.
3. Mở `classification.ipynb`, đổi đường dẫn CSV ở phần đầu notebook thành `IMDB Dataset.csv`.
4. Mở terminal tại thư mục `Lab04` và chạy:

	```bash
	jupyter notebook classification.ipynb
	```

Chạy các cell theo thứ tự từ trên xuống. Notebook sẽ tạo hoặc cập nhật `results.csv` và `confusion_matrix.png` trong thư mục đang chạy.

## Kết quả chính

| Mô hình | F1 |
|---|---:|
| TF-IDF + Logistic Regression | 0.9042 |
| Average Word2Vec + Logistic Regression | 0.8219 |
| TF-IDF + Linear SVM | 0.9045 |
| TF-IDF unigram + bigram + Logistic Regression | 0.9029 |

TF-IDF cho kết quả tốt hơn Average Word2Vec. Thêm bigram làm vocabulary tăng từ 92,647 lên 2,110,390 và thời gian train tăng từ khoảng 10.93 lên 60.81 giây, nhưng F1 giảm nhẹ.

## Tệp trong thư mục

- `classification.ipynb`: kiểm tra dữ liệu, chạy thử nghiệm và đánh giá mô hình.
- `preprocessing.py`: chia dữ liệu và xử lý Average Word2Vec.
- `results.csv`: chỉ số đánh giá các mô hình.
- `confusion_matrix.png`: ma trận nhầm lẫn của mô hình baseline.
- `caculations.pdf`: tài liệu tính toán.
- `prediction.md`: dự đoán trước thí nghiệm.
- `error_analysis.md`: ví dụ phân loại đúng và sai.
- `reflection.md`: nhận xét và hướng phát triển tiếp theo.
