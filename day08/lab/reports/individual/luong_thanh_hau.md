# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Lương Thanh Hậu
**Vai trò trong nhóm:** Eval Owner 
**Ngày nộp:** 13/04/2026
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

Trong bài lab này, tôi chủ yếu phụ trách việc evaluation cho RAG pipeline (Sprint 4: RAG Evaluation). Đầu tiên, tôi phân tích bộ câu hỏi kiểm thử `test_questions.json`, đặc biệt là các câu hỏi gài bẫy (như các câu truy vấn sử dụng alias, hoặc câu hỏi ngoài tài liệu). Tiếp theo, tôi chạy baseline model với chế độ Dense Retrieval và tạo ra bảng điểm `scorecard_baseline.md`. Dựa trên kết quả chạy baseline, tôi xác định các điểm yếu (ví dụ context recall bị ảnh hưởng do query chứa tên cũ thay vì tên mới của tài liệu). Để giải quyết vấn đề này, tôi đã cùng đồng đội chuyển sang sử dụng Hybrid Retrieval (BM25 + Dense) kết hợp CrossEncoder rerank và đánh giá lại bằng `scorecard_variant.md`. Công việc đánh giá của tôi giúp các bạn làm retrieval và generation trong team hiểu rõ chiến lược nào đang mang lại hiệu quả tốt nhất để quyết định phương án cuối cùng.

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

Một khái niệm quan trọng tôi thấu hiểu sâu sắc hơn sau lab là **Hybrid Retrieval**. Ban đầu tôi nghĩ các model embedding (Dense Retrieval) đã rất mạnh vì nó bắt được ngữ nghĩa, nên không cần tới tìm kiếm theo từ khoá (Sparse/BM25). Nhưng thực tế thì ngược lại, điểm yếu lớn nhất của dense embedding là khi người dùng hỏi các thuật ngữ kỹ thuật, mã lỗi đặc thù (ERR-403) hay các bí danh (alias - như “Approval Matrix” thay vì “Access Control SOP”). Dense có thể bỏ qua cụm từ khóa chính xác đó. Bằng cách kết hợp dense (so sánh ý nghĩa) và sparse (BM25 - khớp chính xác từ ngữ), hybrid retrieval tận dụng điểm mạnh của cả hai.

Bên cạnh đó là concept **Evaluation Loop** (LLM-as-a-judge). Thay vì dò từng câu thủ công, việc có 1 bảng điểm tự động với các tiêu chí faithfulness, relevance, context recall, completeness giúp việc debug RAG pipeline nhanh và có căn cứ định lượng (như `scorecard_baseline` so với `scorecard_variant`).

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

Tôi thực sự ngạc nhiên khi dùng Reranker. Kết quả LLM-as-judge khi chuyển từ model dense sang hybrid mode và có rerank lại không làm điểm average tăng như kỳ vọng (thậm chí Relevance tụt từ 4.80 ở baseline xuống còn 4.50 ở variant_hybrid_rerank). Cụ thể ở dạng gài bẫy như câu `q10` - thông báo lỗi khi không có đủ context của model trong Variant 1b (Relevance điểm 1). Điều này đi ngược lại với giả thuyết của tôi ban đầu rằng việc kết hợp thêm Cross-Encoder (cho điểm chéo câu hỏi và đoạn văn bản) sẽ gạn lọc các tài liệu và cải thiện rõ rệt tất cả các chỉ số (đặc biệt là Relevance và Context Recall). Ngoài ra, câu `q09` - một câu hỏi cố tình nằm ngoài doc, điểm Relevance và Faithfulness lại bị giảm ở Variant 1b (hybrid + rerank). Thách thức hiện tại là làm sao cân bằng trọng số giữa điểm của Dense, Sparse và điểm Rerank để trả về câu trả lời ổn định nhất.

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

**Câu hỏi:** `q07`: "Approval Matrix để cấp quyền hệ thống là tài liệu nào?"

**Phân tích:**

Câu hỏi `q07` là một câu alias – tài liệu trong corpus hiện đã đổi tên từ "Approval Matrix" thành "Access Control SOP".
1. Ở baseline (dense retrieval mode), context recall cho câu `q07` bị liệt vào dạng “hỏi mẹo”. Hệ embedding truyền thống ưu tiên các vector ngữ nghĩa tương đồng (trong khi "Approval Matrix" xa lạ với "Access Control SOP" về mặt không gian vector). Do đó, điểm Completeness khá lẹt đẹt vì retrieval engine không bốc được chính xác tài liệu cũ.
2. Qua scorecard, điểm completeness của `q07` tại baseline là 3.
3. Khi đổi qua Variant 1 (hybrid), sự xuất hiện của BM25 (Sparse) khắc phục hoàn toàn vấn đề này bằng cách tra đúng token (cụm từ chuẩn "Approval Matrix" hoặc "Approval" / "Matrix") giúp context recall tăng lên, nhưng nếu để cấu hình hybrid không có rerank có thể điểm vẫn thấp. Nhưng kết quả với Variant 1b (Hybrid + Rerank) cho thấy Completeness vẫn giữ điểm 3. Có vẻ đoạn văn mô tả "Approval Matrix" trong tài liệu Access Control SOP. Giải pháp có lẽ cần đánh thêm metadata cho metadata alias hoặc dùng LLM để generate thêm alias như Expansion query thay vì trông chờ mỗi Rerank.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

Tôi muốn áp dụng query expansion/transformation. Kết quả ở câu hỏi q07 (Alias document) cho thấy dù đã chuyển sang dùng hybrid kết hợp, rerank model có vẻ vẫn không thể đưa mức completeness lên tuyệt đối. Bằng cách sử dụng LLM để sinh ra các câu query con và synonym (như "Access Control" thay thế cho "Approval Matrix" trước khi chạy vào hàm retrieve) thì độ bao phủ context và recall chắc chắn sẽ tốt hơn. Tôi cũng sẽ nâng tham số `top_k_search` lên 15 hoặc 20 trước rerank.

---

*Lưu file này với tên: `reports/individual/[ten_ban].md`*
*Ví dụ: `reports/individual/nguyen_van_a.md`*
