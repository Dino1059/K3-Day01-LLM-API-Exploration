# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu ustrả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature tăng từ 0.0 lên 1.5, câu trả lời càng "bay bướm" và ít an toàn hơn: 0.0 cho phép phản hồi ổn định, nếu lặp lại nhiều lần thì kết quả gần giống nhau, trong khi 1.5 tăng độ ngẫu nhiên, mỗi lần chạy ra một cách diễn đạt khác (và đôi khi tự đưa vào chi tiết không chính xác). Temperature thấp phù hợp khi cần câu trả lời chính xác, cao phù hợp khi cần sáng tạo.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Mức temperature thường từ 0.0 đến 0.2 vì khách hàng cần thông tin chính xác, nhất quán và đáng tin cậy (như chính sách đổi trả, bảng giá, quy trình xử lý lỗi). Đặt temperature thấp giúp mô hình hoạt động theo cơ chế suy luận gần như định hình (deterministic), hạn chế tối đa hiện tượng "bịa" thông tin (hallucination) và đảm bảo các câu trả lời cho cùng một thắc mắc luôn giữ đúng quy chuẩn của doanh nghiệp.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> GPT-4o đắt hơn GPT-4o-mini khoảng 16 đến 17 lần cho cùng một dung lượng workload này (chưa tính token đầu vào)
> Với bảng giá : GPT-4o: ≈$10.00/1M output tokens. GPT-4o-mini: ≈$0.60/1M output tokens. Nên dùng GPT_4o cho những câu hỏi  cần phân tích suy luận chuyên sâu, ngược lại với gpt-4o-mini
---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Phản hồi từ "Giáo viên tiểu học" dùng từ ngữ đơn giản, ngắn gọn cùng hình ảnh ẩn dụ quen thuộc như "cuốn sổ tay dùng chung của cả lớp", trong khi "Chuyên gia tài chính" trả lời dài hơn với các thuật ngữ chuyên sâu như sổ cái phân tán (DLT), mã hóa mật mã và cơ chế đồng thuận (consensus). System prompt đóng vai trò như một khung định hướng ngữ cảnh (contextual boundary) giúp thiết lập "persona" và điều phối tập từ vựng của mô hình. Nhờ đó, nó điều chỉnh độ sâu kiến thức, văn phong và đối tượng hướng tới của câu trả lời mà không cần người dùng phải thay đổi nội dung câu hỏi gốc.
### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
>chênh lệch nhau khoảng 28% – 39%
---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
>Stream quan trọng trong trường hợp ứng dụng cần tương tác trực tiếp với người dùng, còn non-streaming phù hợp cho tác vụ ngầm

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
>Việc tăng khoảng thời gian chờ theo cấp số nhân giúp giãn mật độ request gửi tới hệ thống,API tự phục hồi tài nguyên sau sự cố quá tải.
---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
>Persona tôi chọn: "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt.". Vì trợ lý chạy trên CLI, user không muốn đọc màn hình dài; "ngắn gọn" ép model đi thẳng vào ý chính, tiết kiệm token output → giảm chi phí.
### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**


>History chỉ giữ 3 lượt gần nhất (history[-6:]), nên trợ lý "quên" ngữ cảnh sau vài câu — không có bộ nhớ dài hạn sau khi tắt CLI Lưu history vào file JSON sau mỗi lượt, và nạp lại khi khởi động.
---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
