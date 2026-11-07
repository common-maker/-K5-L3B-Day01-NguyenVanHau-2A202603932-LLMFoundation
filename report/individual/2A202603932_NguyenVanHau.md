# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:2A202603932**
- **Họ và tên:Nguyễn Văn Hậu**
- **SEED** (in ở ô đầu notebook):4733

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | [x] |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | [x] |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | [x] |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*
| Mô hình | F1 (test) | F1 (SUB 40) | p95 ms/tin | USD / 1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.809 | 0.710 | 1.826 | 0.000 |
| 2. BoW + MLP (deep learning) | 0.886 | 0.788 | 2.502 | 0.000 |
| 3a. Encoder (MiniLM) + kNN | 0.913 | 0.895 | 254.341 | 0.000 |
| 3b. Encoder (MiniLM) + LogReg | 0.934 | 0.919 | 37.719 | 0.000 |
| 3c. Encoder (MiniLM) + MLP head | 0.915 | 0.919 | 56.041 | 0.000 |
| 4a. LLM nhỏ zero-shot | 0.000 | 0.000 | 932.891 | 0.006 |
| 4b. LLM lớn zero-shot | 0.000 | 0.000 | 904.845 | 0.057 |
| 4c. LLM nhỏ few-shot | 0.000 | 0.000 | 1693.303 | 0.021 |


## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*
  1. predict("ml_lr", ".lr=0.001 thì đường loss giảm rất chậm ở những epoch đầu, sau đó giảm đều tiệm cận đến giá trị thấp, lr=50 thì đường loss hỗn loạn xu hướng bị divergence") ; kết quả : Lr chỉ nhảy 1 lần ở epoch đầu sau đó bị kéo lại bởi sigmoid ; lệch vì chưa hiểu rõ bản chất của hàm loss với hàm sigmoid
  
## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*
Nhóm mô hình dựa trên Bag-of-Words bị tụt nhiều nhất. natural_spam: BoW + LogReg đoán đúng 0% (0/10 case), BoW + MLP  đúng 10% (1/10 case). obfuscation: BoW + MLP đúng 0% (0/6 case), BoW + LogReg đúng 17% (1/6 case). manipulation : BoW + LogReg đúng 12% (1/8 case), BoW + MLP đúng 25%.

case : "C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift.c0m" - các chữ cái bị thay thành số, làm các từ trong spam không nằm ngoài phạm vi của model, vì tập train bị mất cân bằng vè phía ham, mô hình đoán nhầm thành ham.
case : ham looks spam : "lol I just got a text saying I won £1000 and to call some 0906 number, people still fall for that?" - mô hình không hiểu ngữ nghĩa, chỉ biết mail chứa nhiều từ ngữ thuộc nhóm đặc trung của spam nên đoán nhầm ham thành spam. 

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*
ưu tiên chất lượng tối đa trong SLA, chọn 3c. Encoder (MiniLM) + MLP head.  
Đạt F1-score 0.915 (F1 SUB = 0.919), Đánh đổi: Độ trễ p95 tăng lên đến 37.00 ms gần với ngưỡng 50ms, làm tăng chi phí hạ tầng.
## 7. AI log và điều đã học
|---|---|---|---|
|4 |Rewrite TODO 4.1 – 4.4 and find why response = '', label = None, which lead to F1 = 0.0 | 1. Cung cấp đoạn sưa code cho ll_openai, build_classification_prompt, parse_labevà llm_classify_batch. 2. Cung cấp hướng debug để tìm nguyên nhân lỗi  | Đã chạy unit test lại _run_check, kiểm tra các biến liên quan đến phần response, thay đổi giá trị cho phù hợp |  
Sử dụng llm, llm mạnh chưa hẳn là hiệu quả nhất, những bài toán đơn giản chỉ cần cách xử lý đơn giản.
*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
