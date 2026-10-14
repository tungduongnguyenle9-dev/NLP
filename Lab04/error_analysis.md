# Error Analysis

## 1. Correctly Classified

| Text | True | Predicted |
|---|---|---|
| I, for one, absolutely loved this movie... | positive | positive |
| This movie is just another average action flick... | negative | negative |
| This movie scared heck out of me when I was just... | positive | positive |

## 2. Incorrectly Classified

| Text | True | Predicted | Analysis |
|---|---|---|---|
| "House of Dracula" isn't all that bad of a film... | positive | negative | Câu có phủ định, dễ làm model hiểu sai. |
| The NYT review says that Sigourney Weaver's character... | positive | negative | Câu dài và có nhiều ý nên khó xác định cảm xúc chính. |
| I missed the beginning but I did see most of it... | negative | positive | Nội dung khá mơ hồ, có cả ý trung tính và tiêu cực. |
