# Hướng Dẫn Lab K3-Day01 — Khám Phá LLM API (Bản Dễ Hiểu)

> File này là bản **tóm tắt dễ hiểu** của `LAB_GUIDE.md`, đi kèm **note giải thích bản chất** ("vì sao phải làm vậy") để bạn hiểu sâu, không chỉ làm theo máy móc.
> Toàn bộ code viết trong `template.py`. Mỗi hàm có dạng `raise NotImplementedError(...)` — bạn xóa dòng đó và điền code thật vào.

---

## BẢN CHẤT CỦA BUỔI LAB (đọc trước khi bắt tay)

Vì sao lab này tồn tại? Khi bạn dùng ChatGPT, nó **đang gọi một API** ở phía sau.
API đó nhận vào **danh sách messages** (hội thoại), trả ra **text phản hồi**, và **tính tiền theo token** (không phải theo từ).
Buổi lab đưa bạn về "ngồi ghế lập trình viên" — tự gọi API đó, tự đo timer, tự tính tiền, tự chịu lỗi mạng. Hiểu được 4 điều này = bạn đã hiểu 80% cách mọi ứng dụng AI trên đời hoạt động.

| Khái niệm | Bản chất 1 câu |
|---|---|
| **Chat Completions API** | Gửi list messages → nhận text về. Giống gọi một hàm số trả về chuỗi. |
| **messages** | List các dict `{"role": ..., "content": ...}`. `role` có 3 loại: `system` (chỉ đạo), `user` (câu hỏi), `assistant` (câu trả lời). |
| **system prompt** | "Lời dặn đạo diễn" đặt ở đầu messages → định hình thái độ/ngôn ngữ của model. |
| **temperature / top_p** | Hai núm chỉnh độ "sáng tạo". 0 = ổn định (câu nào cũng ra giống nhau), cao = ngẫu nhiên hơn. |
| **max_tokens** | Hạn ngạch độ dài output ⇒ kiểm soát chi phí & tránh trả lời lan man. |
| **token** | Đơn vị "chữ" mà model tính tiền. **Không phải từ** — 1 từ tiếng Việt có thể = 3-5 token. |
| **tiktoken** | Bộ đếm token chính thức của OpenAI — đếm đúng như cách họ tính tiền bạn. |
| **streaming** | Thay vì đợi trả xong mới show, server bắn từng chunk nhỏ → người dùng đọc dọc ≈ cảm giác "nó đang gõ". |
| **retry + backoff** | Khi API lỗi tạm thời (quá tải, mạng) → thử lại, mỗi lần chờ lâu hơn một chút để không "dùng thân" đánh server đang nghẽn. |
| **mock test** | Test thay API thật bằng "đồ giả" → chạy không tốn tiền, không cần key. Đây là lý do có quy tắc import bên trong hàm (xem dưới). |

---

## ⚠️ QUY TẮC VÀNG (vi phạm = test fail mặc dù chạy thật OK)

> Import `OpenAI` **BÊN TRONG hàm**, KHÔNG đặt ở đầu file.

```python
# ĐÚNG:
def call_openai(...):
    from openai import OpenAI      # <-- trong hàm
    client = OpenAI(...)

# SAI (test sẽ fail):
from openai import OpenAI          # đầu file
def call_openai(...): ...
```

**Vì sao?** Test dùng `unittest.mock.patch("openai.OpenAI")` để **thay thế class OpenAI bằng đồ giả** ngay tại thời điểm import.
- Nếu bạn import ở đầu file → module của bạn đã giữ chặt class thật từ lúc file load xong → mock không tác dụng → test sẽ gọi API thật → fail vì không có key (và tốn tiền!).
- Nếu import trong hàm → mỗi lần hàm chạy, Python mới đi tìm `openai.OpenAI` → mock kịp chặn → test dùng đồ giả. ✅

> Quy tắc tương tự cho `tiktoken` trong `count_tokens` (cũng bọc `import tiktoken` ở trong hàm, hoặc cũng OK nếu chỉ có thử/except). An toàn nhất: **tất cả import thư viện bên ngoài → để trong hàm.**

---

## CHUẨN BỊ MÔI TRƯỜNG (CP0)

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

cp .env.example .env               # rồi mở .env dán key thật vào
pytest tests/ -v                   # phải BÁO FAIL hàng loạt (= đúng)
```

> **Note bản chất:** Kỳ vọng CP0 là "fail chứ không phải lỗi import" — fail kiểu `NotImplementedError` nghĩa là môi trường đã sẵn sàng, chỉ thiếu code của bạn. Nếu gặp `ModuleNotFoundError: openai` → venv chưa activate hoặc chưa `pip install`.

**Không có key OpenAI?** Lấy key NVIDIA NIM **miễn phí** (xem phụ lục cuối file này). Test và `grade.py` **không cần key** vì toàn bộ test đều mock.

---

## BLOCK 1 — API CƠ BẢN (10h00 – 10h40) · CP1

Mục tiêu: gọi được API, đo độ trễ, hiểu 3 tham số sinh text, so sánh 2 model.

### Task 1.1 — `call_openai`

```python
def call_openai(prompt, model=OPENAI_MODEL, temperature=0.7, top_p=0.9, max_tokens=256):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start
    return response.choices[0].message.content, latency
```

> **Note bản chất:**
> - `time.time()` phải ôm sát lời gọi `create(...)` — nếu đặt ngoài phạm vi mạng thì `latency` không còn là "độ trễ API" nữa, nó sẽ lẫn cả thời gian xử lý code của bạn.
> - `response.choices[0].message.content` là "đường hầm" cố định để localStorage text. OpenAI trả một cấu trúc lồng nhau; `choices` là list các phương án (mặc định chỉ 1), `[0]` lấy phần tử đầu, `.message.content` lấy text. Vứt đường hầm này = bài về sai kiểu trả về.
> - Trả về **tuple** `(text, latency)` — chữ ký này là "hợp đồng" với test, đừng sửa.

**Test ngay:** `pytest tests/test_part1.py -k CallOpenAI -v`

### Task 1.2 — `call_openai_mini`

```python
def call_openai_mini(prompt, temperature=0.7, top_p=top_p, max_tokens=256):
    return call_openai(prompt, model=OPENAI_MINI_MODEL,
                       temperature=temperature, top_p=top_p, max_tokens=max_tokens)
```

> **Note bản chất:** Đây là bài học về **tái sử dụng (DRY)**. Copy-paste hai khối giống nhau là "mỏ mìn": ngày nào bạn sửa `call_openai` (đổi endpoint, log metrics...) thì chỉ 1 trong 2 chỗ được sửa → bug khó tìm. Gọi lại hàm cũ + đổi tham số `model` = 1 dòng, sửa 1 lần cập nhật cả hai.

### Task 1.3 — `compare_models`

```python
def compare_models(prompt):
    gpt4o_text, gpt4o_latency = call_openai(prompt)
    mini_text, mini_latency = call_openai_mini(prompt)

    cost = (len(gpt4o_text.split()) / 0.75) / 1000 \
           * PRICING_PER_1K_TOKENS["gpt-4o"]["output"]

    return {
        "gpt4o_response": gpt4o_text,
        "mini_response": mini_text,
        "gpt4o_latency": gpt4o_latency,
        "mini_latency": mini_latency,
        "gpt4o_cost_estimate": cost,
    }
```

> **Note bản chất:**
> - **Tên key phải khớp từng ký tự** với docstring — test dùng `assertIn(key, result)`, gõ `gpt4o_response` thành `gpt_4o_response` = fail mà thôi. Đây là cách dạy "API hợp đồng": hàm tiện ích phải trả đúng shape người ta mong đợi.
> - Ước lượng `0.75 từ ≈ 1 token` là **thô** — chỉ dành cho tiếng Anh. Block 2 sẽ thay bằng tiktoken (chính xác).
> - Test *mock* cả `call_openai` lẫn `call_openai_mini` (không gọi API thật) — vì vậy nếu bạn **không** tái sử dụng mà tự gọi `OpenAI` riêng, vẫn pass vì test patch ở tầng hàm bạn dùng — nhưng bài về sau sẽ khổ.

### ✅ CHECKPOINT 1
```bash
pytest tests/test_part1.py -v          # mong đợi: 10 passed
```

---

## BLOCK 2 — SYSTEM PROMPT & TOKEN (10h40 – 11h20) · CP2

Mục tiêu: dùng role `system` để "đóng vai" cho model, đếm token thật bằng tiktoken, tách chi phí input/output.

### Task 2.1 — `chat_with_system_prompt`

Giống `call_openai`, chỉ khác `messages` có 2 phần tử:

```python
def chat_with_system_prompt(system_prompt, user_prompt, model=OPENAI_MODEL,
                            temperature=0.7, max_tokens=256):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = time.time() - start
    return response.choices[0].message.content, latency
```

> **Note bản chất:** `system` phải **đứng đầu** messages. Model đọc từ trên xuống;истем dòng đầu là "bối cảnh/người chỉ đạo", rồi mới tới câu hỏi. Nếu bạn đảo role → model coi "Bạn là giáo viên..." là câu hỏi người dùng và sẽ **trả lời về** câu đó thay vì **trả lời theo vai** đó. Test có case kiểm tra chính việc `system_prompt` được gửi lên đúng nội dung.

### Task 2.2 — `count_tokens`

```python
def count_tokens(text, model=OPENAI_MODEL):
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)        # ước lượng 1 token ≈ 4 ký tự
```

> **Note bản chất:**
> - `tiktoken` lần đầu chạy cần **tải bảng mã hóa từ mạng**. Trong lab không có mạng hoặc tên model lạ (vd NVIDIA Llama) → raise. Hàm tiện ích **không được crash** vì chuyện hệ thống, nên bọc try/except.
> - `max(1, ...)` thay vì `len(text)//4` cél: text rỗng sẽ ra 0, và 0 token có thể làm chia 0 ở Task 2.3 → dính ZeroDivision 🛑. Đây là phòng thủ vĩ mô: **luôn đảm bảo contract trả giá trị ">" rỗng**.
> - Trong `PRICING_PER_1K_TOKENS` không có tên model NIM → phải fallback ở Task 2.3.

### Task 2.3 — `estimate_cost`

```python
def estimate_cost(prompt, response, model=OPENAI_MODEL):
    input_tokens = count_tokens(prompt, model)
    output_tokens = count_tokens(response, model)

    pricing = PRICING_PER_1K_TOKENS.get(model, PRICING_PER_1K_TOKENS["gpt-4o"])
    input_cost = input_tokens / 1000 * pricing["input"]
    output_cost = output_tokens / 1000 * pricing["output"]

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }
```

> **Note bản chất:**
> - **Input và output có giá KHÁC NHAU** (output thường đắt hơn ~4 lần). Vì sao? Đầu vào model chỉ cần "đọc", đầu ra model phải **sinh token** từng cái một (tốn compute nhiều hơn).
> - `.get(model, default)` là mẹo "mềm" — tránh KeyError nếu model lạ. Đây chính là pattern **defensive programming**:ฐ lập trình phòng thủ,kt với "lỗi offline" ở Task 2.2 cùng triết lý.
> - `total_cost = input + output` không phải `total_cost = output` — test có thể check sự tồn tại key này riêng biệt.

### ✅ CHECKPOINT 2
```bash
pytest tests/test_part2.py -v          # mong đợi: 10 passed
```

---

## BLOCK 3 — STREAMING & ĐỘ BỀN (11h30 – 12h10) · CP3

Mục tiêu: in token dần dần (UX tức thời), duy trì hội thoại, tự retry khi lỗi.

### Task 3.1 — `streaming_chatbot`

```python
def streaming_chatbot():
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    history = []

    while True:
        user_msg = input("Bạn: ")
        if user_msg.strip().lower() in ("quit", "exit"):
            break

        messages = history + [{"role": "user", "content": user_msg}]
        stream = client.chat.completions.create(
            model=OPENAI_MODEL, messages=messages, stream=True,
        )

        reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""    # chunk cuối là None!
            print(delta, end="", flush=True)
            reply += delta

        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": reply})
        history = history[-6:]                                # giữ 3 lượt cuối
```

> **Note bản chất:**
> - `stream=True` đổi ngỏ giao tiếp: thay vì 1 response trọn vẹn, server bắn về một **iterator**. Mỗi `chunk.choices[0].delta.content` = **một mảnh** text mới sinh (delta = "số gia tăng").
> - **Quên `or ""`** là bug kinh điển: chunk cuối cùng thường có `delta.content = None` (chỉ chứa metadata đóng stream), và `print(None)` in chữ "None" rác ra màn hình / crash khi cộng chuỗi. Test cũng bắt được lỗi này.
> - `history[-6:]` vì **1 lượt = 1 user + 1 assistant = 2 message**, nên 3 lượt = 6 message. Vì sao phải cắt? Càng chat lâu, messages càng phình → mỗi lượt sau càng tốn **input token** → chi phí tuyến tính tăng theo thời gian chat. Đây là một tradeoff kinh điển: **ngắn nhớ = rẻ, dài nhớ = đắt**.
> - `print(..., flush=True)` ép Python đẩy buffer ra ngay — không thì terminal có thể "đ Savings lũy" cho đến khi full dòng mới in, mất luôn cảm giác "đang gõ".

### Task 3.2 — `retry_with_backoff`

```python
def retry_with_backoff(fn, max_retries=3, base_delay=0.1):
    for attempt in range(max_retries + 1):       # lần đầu + max_retries lần retry
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise                            # hết lượt → ném lỗi GỐC ra
            time.sleep(base_delay * (2 ** attempt))
```

> **Note bản chất:**
> - `range(max_retries + 1)` vì ta cần **thử tổng cộng `max_retries + 1` lần** (lần đầu không phải retry). Lập trình viên hay nhầm chỗ này.
> - `raise` trần (không tham số) **giữ nguyên exception gốc** với traceback nguyên vẹn → người gọi biết chính xác lỗi gì. Nếu dùng `raise Exception("...")` thì mất ngữ cảnh, khó debug.
> - **Exponential backoff** (0.1 → 0.2 → 0.4 → 0.8...) vs **delay cố định** (luôn 1s): nếu server đang quá tải, hàng nghìn client cùng chờ đúng 1s rồi retry cùng lúc → server nghẽn hơn. Backoff exponentially **"phân tán"** thời gian retry tự nhiên → giảm áp lực. Đây là pattern chuẩn trong **mọi SDK cloud** (AWS, Google, Stripe...).
> - Bọc lời gọi trong `lambda` ở Block 4 (`retry_with_backoff(lambda: client.chat...)`) là vì `retry_with_backoff` nhận **callable không tham số** — lambda đóng gói lời gọi có tham số thành hàm 0-arg. Đây là pattern **higher-order function** phổ biến.

### ✅ CHECKPOINT 3
```bash
pytest tests/test_part3.py -v          # mong đợi: 6 passed
```

---

## BLOCK 4 — MINI-PROJECT: TRỢ LÝ CLI (12h10 – 12h50) · CP4

Ghép tất cả: persona + streaming + history + retry + thống kê.

```python
def run_assistant(persona, get_input=None, max_turns=None):
    if get_input is None:
        get_input = input
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    history, num_turns, total_tokens, total_cost = [], 0, 0, 0.0

    while True:
        # Kiểm tra max_turns TRƯỚC khi đọc input — nếu không sẽ StopIteration
        if max_turns is not None and num_turns >= max_turns:
            break
        user_msg = get_input()
        if user_msg.strip().lower() in ("quit", "exit"):
            break

        # System prompt đứng đầu; persona không bị trôi khi history bị cắt
        messages = ([{"role": "system", "content": persona}]
                     + history + [{"role": "user", "content": user_msg}])

        stream = retry_with_backoff(
            lambda: client.chat.completions.create(
                model=OPENAI_MODEL, messages=messages, stream=True,
            )
        )

        reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            reply += delta

        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": reply})
        history = history[-6:]

        num_turns += 1
        total_tokens += count_tokens(user_msg) + count_tokens(reply)
        total_cost += estimate_cost(user_msg, reply)["total_cost"]

    return {
        "num_turns": num_turns,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "history": history,
    }
```

> **Note bản chất — 3 điểm khác biệt then chốt so với `streaming_chatbot`:**
>
> 1. **`get_input` là tham số (mặc định = `input`)** — đây là kỹ thuật **dependency injection**. Test sẽ thay `get_input` bằng list các câu hỏi giả lập → test "gõ phím hộ" bạn, không cần bàn phím thật. Đây là cách mọi hệ thống viết unit test cho CLI.
> 2. **System prompt ghép lại MỖI lượt** (không nằm trong history). Vì sao? Nếu persona nằm trong `history` thì tới lúc `history[-6:]` cắt đi, **persona bị mất** → model tự nhiên quên vai trò. Ghép lại ở `messages` mỗi lần gọi ⇒ persona luôn ở đầu, không bị cắt. Đây là bug rất thường gặp!
> 3. **Kiểm tra `max_turns` TRƯỚC `get_input()`** — nếu test truyền list 3 câu và `max_turns=3`, bạn đọc input lần thứ 4 → `StopIteration`. Check trước = chấm dứt đúng lúc, không "hơi" thêm 1 lần input.

> **Note bản chất — vì sao max_turns=None có nghĩa "vô hạn"?** Đây là convention Python: `None` được dùng làm **giá trị sentinel** thay cho "không giới hạn". So sánh `if max_turns is not None` chính ra đẹp hơn `if max_turns` vì tránh bị lừa bởi `max_turns=0` (0 là falsy!). Nếu viết `if max_turns and num_turns >= max_turns:` thì `max_turns=0` sẽ bỏ qua check → API không trả lời turn 0 → bug.

### ✅ CHECKPOINT 4
```bash
pytest tests/test_part4.py -v          # mong đợi: 9 passed (4 Basic + 5 Scenario)
python template.py                    # demo thật (cần key)
```

> **Note bản chất:** Nhóm test `Scenario` = "demo tự động" — nó mô phỏng hội thoại nhiều lượt, kiểm tra `num_turns`, `total_tokens`, `total_cost`, `history`, và cả streaming. Đây chính là **15 điểm demo** của bạn (đã tự động hóa, không cần trình bày).

---

## WRAP-UP & NỘP BÀI

```bash
# 1. Rà exercises.md — đủ 9 câu trả lời chưa?
# 2. Chấm điểm tự động:
python grade.py                       # nhìn bảng điểm, mục nào thấp → sửa file ấy

# 3. Đóng gói:
mkdir -p solution
cp template.py solution/solution.py
cp exercises.md solution/exercises.md
python grade.py                       # grade ưu tiên chấm solution/ → xác nhận lại
zip -r solution.zip solution/
# Đổi tên: <mã sinh viên>_lab_1.zip → upload LMS
```

> **Note bản chất:** `tests/_loader.py` có dòng `if (SOLUTION_DIR / "solution.py").exists(): ...` — nếu thấy folder `solution/` thì chấm file đó, không thì rơi về `template.py`. Nghĩa là sau khi copy vào `solution/`, chạy `grade.py` **sẽ chấm solution** chứ không phải template. Đừng quên bước verify này, nếu không điểm có thể lệch!

---

## PHỤ LỤC A — Lỗi Thường Gặp (giải thích bản chất)

| Triệu chứng | Bản chất | Cách sửa |
|---|---|---|
| Test fail dù "chạy thật OK" | Import OpenAI ở đầu file → mock không tác dụng | Chuyển `from openai import OpenAI` vào **trong hàm** |
| `AuthenticationError` khi pytest | Cùng原因 trên — mock không thao túng được import vĩ mô | Như trên |
| `KeyError: 'gpt4o_response'` | Tên key dict sai 1 chữ/symbol | So từng ký tự với docstring |
| Chunk cuối crash (`TypeError NoneType`) | Chunk cuối có `delta.content = None`, `+ None` lỗi | `delta = ...delta.content or ""` |
| History phình, chi phí tăng dần | Quên cắt history | `history = history[-6:]` sau mỗi lượt |
| `StopIteration` trong scenario test | Đọc input nhiều hơn số câu kịch bản | Check `max_turns` **trước** `get_input()` |
| tiktoken treo/lỗi | Lần đầu cần mạng tải encoding, hoặc model lạ | Fallback `max(1, len(text) // 4)` trong try/except |
| `ZeroDivisionError` ở cost | `len(text)//4` trả 0, sau `/0` | `max(1, ...)` để đảm bảo ≥ 1 token |

---

## PHỤ LỤC B — Lấy API Key NVIDIA NIM Miễn Phí

NVIDIA NIM = endpoint **tương thích chuẩn OpenAI** nhưng miễn phí. **Không sửa dòng code nào** — OpenAI SDK tự đọc `OPENAI_BASE_URL` từ `.env`.

1. Mở [build.nvidia.com](https://build.nvidia.com) → Login → Create Account
2. Vào trang model bất kỳ → **Get API Key** → Generate Key → copy dạng `nvapi-...` **lưu ngay** (chỉ hiện 1 lần)
3. Sửa `.env`:
   ```bash
   OPENAI_API_KEY=nvapi-key-cua-ban
   OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
   LAB_MODEL=meta/llama-3.3-70b-instruct
   LAB_MINI_MODEL=meta/llama-3.1-8b-instruct
   ```
4. Test:
   ```bash
   python -c "from template import call_openai; print(call_openai('Chào bằng 1 câu tiếng Việt'))"
   ```

> **Note bản chất:**
> - Lab này thay model mà code **không phải sửa** vì có 2 lớp trừu tượng: (1) OpenAI SDK đọc `base_url` từ env để chuyển server, (2) `template.py` đọc `LAB_MODEL`/`LAB_MINI_MODEL` từ env để đổi tên model. **Hướng tới dependency injection qua env** là pattern production-grade.
> - `count_tokens` với model Llama → tiktoken không có encoding → fallback `len(text)//4` (đúng như thiết kế). `estimate_cost` với model lạ → `.get(model, default=gpt-4o)` (tham chiếu học tập, vì NIM thực tế miễn phí).
> - `pytest` và `grade.py` không cần key → điểm không phụ thuộc bạn dùng OpenAI hay NIM.
> - Lỗi 429 (hết hạn mức) ⇒ là lúc `retry_with_backoff` (Task 3.2) **tỏa sáng** — bạn tự kiểm chứng được lý do bài ra đề hàm đó.

---

## TÓM TẮT BẢN CHẤT CUỐI CÙNG

Buổi lab dạy 1 vòng tròn: **Tán thông, Kiểm soát, Đo lường, Chịu lỗi**.

```
        [Tán thông]
   call_openai  →  system prompt  →  streaming
   (API cơ bản)     (đóng vai)       (UX tức thời)
        ↑                                  ↓
   [Đo lường]                       [Kiểm soát]
   count_tokens                      temperature, top_p,
   estimate_cost                     max_tokens, history[-6:]
        ↑                                  ↓
        └──────────[Chịu lỗi]──────────────┘
                  retry_with_backoff
                  (chống tạm thời lỗi)

   Tất cả ghép vào: run_assistant (mini-project)
```

Ngắn gọn: bạn đã đi từ "người dùng ChatGPT" thành "người biết lắp ráp một trợ lý AI từ những nguyên tử API". Đây là móng cho mọi K3 sau này.

Chúc bạn hài lòng với buổi lab! 🎯
```
```