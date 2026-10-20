# Error analysis — IMDb 50K (Kaggle) / LSTM

Phân tích từ dự đoán thật của model với `max_length=100`.

| Kết quả | Text (trích) | True | Predicted | Analysis |
|---|---|---|---|---|
| Correct | Although it has been 2 years, I still remember the complete waste that comprises the entire plot of the movie. Unfortunately, I came across this movie after my friends and I selected it while browsing through the new rel... | negative | negative | Có phủ định; cần xem phạm vi và ngữ cảnh. |
| Correct | THE BLOB is a great horror movie, not merely because of the vividly horrific images of its nearly unstoppable, flesh-dissolving title character, but because it features a real societal message. It is, in many ways, a "fe... | positive | positive | Có phủ định; cần xem phạm vi và ngữ cảnh. |
| Correct | The saddest thing about this film is that only 8 people cared to leave a review of it and NO-ONE felt it worthwhile leaving a comment on the message boards.<br /><br />Made the same year as Philadelphia...the Tom Hanks O... | positive | positive | Có phủ định; cần xem phạm vi và ngữ cảnh. |
| Incorrect | This is a film that has garnered any interest or praise it has received simply on the merit of being a lesbian interest piece. The performances are mostly emotionless compared to better films in the GLBT interest genre. ... | negative | positive | Có phủ định; cần xem phạm vi và ngữ cảnh. |
| Incorrect | If this movie had a point I never discovered it. A very depressing movie which supposedly is about the final evacuation of the residents living in a dam site area on the Northfork River in Montana. The problem is that th... | negative | positive | Có phủ định; cần xem phạm vi và ngữ cảnh. |
| Incorrect | This movie is pretty cheesy, but I do give it credit for at least trying to provide some characterization for it's principles. There are some great moments in the film and the dialogue has some great moments as well.<br ... | positive | negative | Có phủ định; cần xem phạm vi và ngữ cảnh. |

## Nhận xét

- Các dấu hiệu ở cột Analysis là giả thuyết, không phải nguyên nhân đã được chứng minh.
- Cần đọc toàn bộ 6 review để phân biệt lỗi sequence, từ hiếm, nhãn mơ hồ và truncation.
- Không thể kết luận nguyên nhân chỉ từ accuracy hoặc một prediction.
