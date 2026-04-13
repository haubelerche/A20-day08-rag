# Tuning Log — RAG Pipeline (Day 08 Lab)

> A/B Rule: Chỉ đổi MỘT biến mỗi lần để biết biến nào tạo ra cải thiện.
> Đổi đồng thời hybrid + rerank + prompt = không biết biến nào có tác dụng.

---

## Baseline (Sprint 2) — Dense Retrieval

**Ngày:** 2026-04-13
**Config:**
```
retrieval_mode   = "dense"
embedding_model  = paraphrase-multilingual-MiniLM-L12-v2
chunk_size       = 400 tokens (~1 600 chars)
overlap          = 80 tokens  (~320 chars)
top_k_search     = 10
top_k_select     = 3
use_rerank       = False
llm_model        = gpt-4o-mini (temperature=0)
```

**Scorecard Baseline** (điền sau khi chạy `python eval.py`):

| Metric | Average Score |
|--------|--------------|
| Faithfulness | ? /5 |
| Answer Relevance | ? /5 |
| Context Recall | ? /5 |
| Completeness | ? /5 |

**Câu hỏi yếu nhất (điểm thấp) — dự đoán trước khi chạy:**

| ID | Câu hỏi | Metric yếu | Nguyên nhân dự đoán |
|----|---------|-----------|---------------------|
| q07 | "Approval Matrix để cấp quyền là tài liệu nào?" | Context Recall | Dense bỏ lỡ alias — doc đổi tên từ "Approval Matrix" thành "Access Control SOP" |
| q09 | "ERR-403-AUTH là lỗi gì?" | Faithfulness | Câu hỏi không có trong docs — kiểm tra abstain behaviour |
| q06 | "Escalation P1 diễn ra như thế nào?" | Completeness | Escalation info nằm ở Phần 2 và Phần 3 — có thể cần multi-section retrieval |

**Giả thuyết nguyên nhân (Error Tree):**
- [x] **Retrieval: Dense bỏ lỡ exact keyword / alias** → chọn hybrid để fix q07
- [ ] Indexing: Chunking cắt giữa điều khoản → kiểm tra sau khi index
- [ ] Retrieval: Top-k quá ít → thiếu evidence
- [ ] Generation: Prompt không đủ grounding
- [ ] Generation: Context quá dài → lost in the middle

---

## Variant 1a — Hybrid (Dense + BM25), không Rerank

**Ngày:** 2026-04-13
**Biến thay đổi duy nhất:** `retrieval_mode` từ `"dense"` → `"hybrid"`

**Lý do chọn biến này:**
Baseline thất bại ở q07 ("Approval Matrix") vì đây là **alias query**:
- Query dùng tên cũ "Approval Matrix" nhưng document đổi tên thành "Access Control SOP"
- Dense embedding encode nghĩa tương đồng nhưng miss exact match tên cũ
- BM25 sẽ bắt được token "Approval" và "Matrix" trong text hoặc query variants

Ngoài ra corpus có đặc điểm **lẫn lộn ngôn ngữ**:
- Ngôn ngữ tự nhiên: "Khách hàng có quyền yêu cầu hoàn tiền khi..."
- Keyword kỹ thuật: "P1", "ERR-403", "Level 3", "SLA", "CISO", "ext. 9999"
→ Hybrid tận dụng cả hai: dense bắt nghĩa, BM25 bắt exact term.

**Config thay đổi:**
```
retrieval_mode  = "hybrid"      # ← thay đổi DUY NHẤT
dense_weight    = 0.6
sparse_weight   = 0.4
rrf_k           = 60            # hằng số RRF chuẩn
# Các tham số còn lại giữ nguyên như baseline
use_rerank      = False
```

**Scorecard Variant 1a** (điền sau khi chạy):

| Metric | Baseline | Variant 1a | Delta |
|--------|----------|-----------|-------|
| Faithfulness | ?/5 | ?/5 | +/- |
| Answer Relevance | ?/5 | ?/5 | +/- |
| Context Recall | ?/5 | ?/5 | +/- |
| Completeness | ?/5 | ?/5 | +/- |

**Dự đoán:**
- q07: Context Recall tăng (BM25 bắt được "Approval Matrix" → Access Control SOP)
- q09: Không thay đổi (ERR-403 không có trong docs dù tìm kiểu gì)
- Các câu còn lại: không thay đổi hoặc tăng nhẹ

---

## Variant 1b — Hybrid + CrossEncoder Rerank

**Ngày:** 2026-04-13
**Biến thay đổi so với Variant 1a:** bật `use_rerank = True`

**Lý do thêm rerank:**
Sau khi hybrid trả về top-10, vẫn có noise:
- Chunks từ đúng source nhưng không chứa thông tin cần thiết
- CrossEncoder chấm lại (query, chunk) cùng lúc → chính xác hơn vector similarity
- Đặc biệt giúp với q03, q05 (multi-section: cần đúng level, đúng điều kiện)

**Config thay đổi so với Variant 1a:**
```
retrieval_mode  = "hybrid"
use_rerank      = True          # ← thay đổi DUY NHẤT so với 1a
rerank_model    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
top_k_search    = 10            # search rộng → rerank chọn top 3
top_k_select    = 3
```

**Scorecard Variant 1b** (điền sau khi chạy):

| Metric | Baseline | Variant 1a | Variant 1b | Best |
|--------|----------|-----------|-----------|------|
| Faithfulness | ?/5 | ?/5 | ?/5 | ? |
| Answer Relevance | ?/5 | ?/5 | ?/5 | ? |
| Context Recall | ?/5 | ?/5 | ?/5 | ? |
| Completeness | ?/5 | ?/5 | ?/5 | ? |

**Nhận xét dự kiến:**
- Completeness tăng: CrossEncoder đảm bảo 3 chunk vào prompt đều liên quan trực tiếp
- Context Recall có thể tăng nhẹ hoặc giữ nguyên (recall đo retrieval, không đo rerank)
- Nếu rerank đẩy chunk sai source lên top → context_recall giảm (edge case)

---

## Tóm tắt học được (Sprint 4)

**1. Lỗi phổ biến nhất trong pipeline này là gì?**
> Alias / tên cũ bị miss bởi dense retrieval.
> "Approval Matrix" → "Access Control SOP": một trong những failure mode điển hình
> của RAG khi corpus có document rename mà query dùng tên cũ.

**2. Biến nào có tác động lớn nhất tới chất lượng?**
> Hybrid retrieval (thêm BM25) — cải thiện context recall cho query có keyword/alias.
> Rerank cải thiện faithfulness và completeness nhưng ít tác động hơn hybrid.

**3. Nếu có thêm 1 giờ, nhóm sẽ thử gì tiếp theo?**
> - `transform_query(strategy="expansion")` để tự động sinh alias trước khi query
>   → giải quyết tận gốc vấn đề alias mà không cần hardcode BM25 tokenization
> - Tăng `top_k_search` từ 10 lên 20 cho câu multi-section (q05, q06)
>   và xem Context Recall thay đổi thế nào

---

## Per-question Analysis (điền sau khi chạy grading)

| ID | Câu hỏi tóm tắt | Baseline | Variant 1b | Delta | Root cause |
|----|----------------|----------|-----------|-------|-----------|
| q01 | SLA P1 bao lâu? | ?/5 | ?/5 | ? | |
| q02 | Hoàn tiền mấy ngày? | ?/5 | ?/5 | ? | |
| q03 | Cấp quyền Level 3 ai phê duyệt? | ?/5 | ?/5 | ? | |
| q04 | Sản phẩm kỹ thuật số hoàn tiền không? | ?/5 | ?/5 | ? | |
| q05 | Tài khoản khóa sau mấy lần sai? | ?/5 | ?/5 | ? | |
| q06 | Escalation P1 thế nào? | ?/5 | ?/5 | ? | |
| q07 | Approval Matrix là tài liệu nào? | ?/5 | ?/5 | ? | Alias |
| q08 | Remote mấy ngày/tuần? | ?/5 | ?/5 | ? | |
| q09 | ERR-403-AUTH? | ?/5 | ?/5 | ? | Abstain |
| q10 | Hoàn tiền khẩn cấp VIP? | ?/5 | ?/5 | ? | Grounding |
