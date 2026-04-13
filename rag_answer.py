"""
rag_answer.py — Sprint 2 + Sprint 3: Retrieval & Grounded Answer
================================================================
Sprint 2 (60 phút): Baseline RAG
  - Dense retrieval từ ChromaDB
  - Grounded answer function với prompt ép citation
  - Trả lời được ít nhất 3 câu hỏi mẫu, output có source

Sprint 3 (60 phút): Tuning tối thiểu
  - Thêm hybrid retrieval (dense + sparse/BM25) với Reciprocal Rank Fusion
  - Thêm rerank bằng CrossEncoder
  - Query transformation (expansion / decomposition / HyDE)
  - Bảng so sánh baseline vs variant

Definition of Done Sprint 2:
  ✓ rag_answer("SLA ticket P1?") → câu trả lời có citation [1]
  ✓ rag_answer("ERR-403-AUTH") → "Không đủ dữ liệu" (abstain)

Definition of Done Sprint 3:
  ✓ Hybrid + CrossEncoder rerank chạy end-to-end
  ✓ Có bảng so sánh baseline vs variant
"""
from rank_bm25 import BM25Okapi

import os
import re
import sys
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# =============================================================================
# CẤU HÌNH
# =============================================================================

TOP_K_SEARCH = 10    # Số chunk lấy từ vector store (search rộng)
TOP_K_SELECT = 3     # Số chunk gửi vào prompt (sau rerank/select)

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Module-level caches — tránh reload ChromaDB / BM25 mỗi lần query
_chroma_collection = None
_bm25_index = None
_bm25_chunks: List[Dict[str, Any]] = []
_cross_encoder = None


# =============================================================================
# CACHE HELPERS
# =============================================================================

def _get_collection():
    """Lazy-load ChromaDB collection (singleton).
    - HttpClient  nếu CHROMA_HOST được set (Docker mode)
    - PersistentClient nếu chạy local
    """
    global _chroma_collection
    if _chroma_collection is None:
        from index import CHROMA_DB_DIR, _get_chroma_client
        client = _get_chroma_client(CHROMA_DB_DIR)
        _chroma_collection = client.get_collection("rag_lab")
    return _chroma_collection


def _get_bm25():
    """Lazy-load BM25 index (singleton). Load tất cả chunks từ ChromaDB 1 lần."""
    global _bm25_index, _bm25_chunks
    if _bm25_index is not None:
        return _bm25_index, _bm25_chunks

    collection = _get_collection()

    # Load ALL chunks (documents + metadatas)
    results = collection.get(include=["documents", "metadatas"])
    _bm25_chunks = [
        {"text": doc, "metadata": meta, "score": 0.0}
        for doc, meta in zip(results["documents"], results["metadatas"])
    ]

    # Tokenize: lowercase split (đơn giản, đủ cho corpus này)
    tokenized = [doc.lower().split() for doc in results["documents"]]
    _bm25_index = BM25Okapi(tokenized)
    return _bm25_index, _bm25_chunks


def _get_cross_encoder():
    """Lazy-load CrossEncoder model (singleton)."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        print("[CrossEncoder] Loading ms-marco-MiniLM-L-6-v2 ...")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("[CrossEncoder] Loaded.")
    return _cross_encoder


# =============================================================================
# RETRIEVAL — DENSE (Vector Search)
# =============================================================================

def retrieve_dense(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    """
    Dense retrieval: tìm kiếm theo embedding similarity trong ChromaDB.

    - Embed query bằng cùng model đã dùng khi index (get_embedding trong index.py)
    - Query ChromaDB với cosine similarity
    - Trả về kết quả kèm score (score = 1 - cosine_distance)

    Returns:
        List dicts: {"text", "metadata", "score"}
    """
    from index import get_embedding

    collection = _get_collection()
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "metadata": meta,
            "score": round(1.0 - dist, 4),  # cosine: score = 1 - distance
        })

    return chunks


# =============================================================================
# RETRIEVAL — SPARSE / BM25 (Keyword Search)
# =============================================================================

def retrieve_sparse(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    """
    Sparse retrieval: BM25 keyword search.

    Mạnh ở: exact term, mã lỗi, tên riêng (P1, ERR-403, Level 3, refund)
    Yếu ở: câu hỏi paraphrase, đồng nghĩa, ngữ nghĩa trừu tượng

    Dùng rank-bm25 (BM25Okapi) trên toàn bộ chunks đã index.
    BM25 index được build lazily và cache (rebuild nếu gọi lại sau build_index).

    Returns:
        List dicts: {"text", "metadata", "score"} — chỉ trả về docs có score > 0
    """
    bm25, all_chunks = _get_bm25()

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Sort theo score giảm dần, lấy top_k
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        {**all_chunks[i], "score": round(float(scores[i]), 4)}
        for i in top_indices
        if scores[i] > 0  # bỏ doc không liên quan gì
    ]


# =============================================================================
# RETRIEVAL — HYBRID (Dense + Sparse với Reciprocal Rank Fusion)
# =============================================================================

def retrieve_hybrid(
    query: str,
    top_k: int = TOP_K_SEARCH,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval: kết hợp dense và sparse bằng Reciprocal Rank Fusion (RRF).

    RRF formula (từ slide):
        RRF_score(doc) = dense_weight  * 1/(60 + dense_rank)
                       + sparse_weight * 1/(60 + sparse_rank)
        Hằng số 60 là giá trị chuẩn trong RRF paper (Cormack et al. 2009).

    Phù hợp khi corpus có cả:
      - Ngôn ngữ tự nhiên (policy text, FAQ)  → dense bắt được
      - Từ khoá kỹ thuật (P1, ERR-403, Level 3, SLA, refund)  → BM25 bắt được

    Args:
        dense_weight / sparse_weight: Trọng số mặc định 0.6 / 0.4
            → Điều chỉnh nếu corpus thiên về keyword (tăng sparse)
              hoặc thiên về ngữ nghĩa (tăng dense).
    """
    dense_results  = retrieve_dense(query, top_k=top_k)
    sparse_results = retrieve_sparse(query, top_k=top_k)

    # Nếu BM25 không trả về gì (chưa index / query quá mơ hồ) → fallback dense
    if not sparse_results:
        return dense_results

    RRF_K = 60
    rrf_scores: Dict[str, float] = {}
    chunk_map:  Dict[str, Dict]  = {}

    for rank, chunk in enumerate(dense_results):
        # Dùng (source + text prefix) làm key duy nhất cho mỗi chunk
        key = chunk["metadata"].get("source", "") + "|" + chunk["text"][:80]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + dense_weight / (RRF_K + rank + 1)
        chunk_map[key] = chunk

    for rank, chunk in enumerate(sparse_results):
        key = chunk["metadata"].get("source", "") + "|" + chunk["text"][:80]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + sparse_weight / (RRF_K + rank + 1)
        chunk_map[key] = chunk

    # Sort theo RRF score giảm dần
    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)[:top_k]

    result = []
    for key in sorted_keys:
        chunk = dict(chunk_map[key])
        chunk["score"] = round(rrf_scores[key], 6)
        result.append(chunk)

    return result


# =============================================================================
# RERANK — CrossEncoder
# =============================================================================

def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = TOP_K_SELECT,
) -> List[Dict[str, Any]]:
    """
    Rerank candidates bằng CrossEncoder (ms-marco-MiniLM-L-6-v2).

    Tại sao cần rerank:
      - Vector search (dense/hybrid) có thể trả về chunk liên quan về chủ đề
        nhưng không thực sự trả lời câu hỏi cụ thể.
      - CrossEncoder chấm lại "query + chunk" cùng lúc → chính xác hơn bi-encoder.

    Funnel logic (từ slide):
      Search rộng top_k_search=10 → CrossEncoder rerank → chọn top_k_select=3

    Khi nào dùng:
      - Dense/hybrid trả về nhiều noise (câu hỏi mơ hồ, corpus rộng)
      - Muốn đảm bảo 3 chunk vào prompt đều thực sự relevant

    Note: CrossEncoder ms-marco-MiniLM-L-6-v2 được train trên tiếng Anh.
    Với corpus tiếng Việt, vẫn hoạt động tốt với các term kỹ thuật (SLA, P1,
    access control, refund) nhưng có thể kém hơn với paraphrase thuần Việt.

    Fallback: nếu sentence_transformers chưa cài → trả về top_k đầu.
    """
    try:
        model = _get_cross_encoder()
        pairs = [[query, chunk["text"]] for chunk in candidates]
        scores = model.predict(pairs)

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in ranked[:top_k]]

    except ImportError:
        print("[rerank] sentence_transformers chưa cài — fallback về top_k đầu tiên")
        return candidates[:top_k]
    except Exception as e:
        print(f"[rerank] Lỗi: {e} — fallback về top_k đầu tiên")
        return candidates[:top_k]


# =============================================================================
# QUERY TRANSFORMATION
# =============================================================================

def transform_query(query: str, strategy: str = "expansion") -> List[str]:
    """
    Biến đổi query để tăng recall.

    Strategies:
      "expansion"     — Thêm 2 paraphrase / synonym để cover nhiều cách diễn đạt
                        Khi dùng: query dùng alias, tên cũ ("Approval Matrix" →
                        "Access Control SOP")
      "decomposition" — Tách query phức tạp thành 2-3 sub-queries đơn giản hơn
                        Khi dùng: câu hỏi hỏi nhiều thứ cùng lúc
      "hyde"          — Sinh câu trả lời giả (Hypothetical Document Embedding)
                        rồi embed câu trả lời đó thay cho query
                        Khi dùng: query mơ hồ, abstract; document style rất cụ thể

    Nếu không có API key → trả về [query] gốc (không transform).
    """
    prompt_map = {
        "expansion": (
            f'Cho câu hỏi: "{query}"\n'
            'Tạo 2 cách diễn đạt khác (paraphrase hoặc synonym) bằng tiếng Việt.\n'
            'Mục đích: giúp tìm kiếm tài liệu với các cách viết khác nhau.\n'
            'Output JSON array only: ["phiên bản 1", "phiên bản 2"]'
        ),
        "decomposition": (
            f'Tách câu hỏi sau thành 2-3 câu hỏi đơn giản hơn: "{query}"\n'
            'Output JSON array only: ["câu 1", "câu 2"]'
        ),
        "hyde": (
            f'Viết một đoạn văn ngắn (~3 câu) có thể là câu trả lời cho: "{query}"\n'
            'Đây là câu trả lời giả để cải thiện vector search. Output: chỉ đoạn văn.'
        ),
    }

    if strategy not in prompt_map:
        return [query]

    try:
        raw = call_llm(prompt_map[strategy])

        if strategy == "hyde":
            return [raw.strip()]

        # Parse JSON array cho expansion / decomposition
        clean = re.sub(r"```json\s*|```\s*", "", raw).strip()
        variants: List[str] = json.loads(clean)
        # Trả về query gốc + tối đa 2 variant
        return [query] + [v for v in variants if v and v != query][:2]

    except Exception:
        return [query]  # Fallback an toàn


# =============================================================================
# GENERATION — GROUNDED ANSWER FUNCTION
# =============================================================================

def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """
    Đóng gói danh sách chunks thành context block để đưa vào prompt.

    Format: structured snippets với source, section, score.
    Số [1][2][3] giúp model dễ trích dẫn trong câu trả lời.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta  = chunk.get("metadata", {})
        source  = meta.get("source", "unknown")
        section = meta.get("section", "")
        score   = chunk.get("score", 0)
        text    = chunk.get("text", "")

        header = f"[{i}] {source}"
        if section:
            header += f" | {section}"
        if score > 0:
            header += f" | score={score:.3f}"

        context_parts.append(f"{header}\n{text}")

    return "\n\n".join(context_parts)


def build_grounded_prompt(query: str, context_block: str) -> str:
    """
    Grounded prompt theo 4 quy tắc từ slide:
      1. Evidence-only : chỉ trả lời từ retrieved context
      2. Abstain       : thiếu context → nói không đủ dữ liệu
      3. Citation      : gắn [1][2] khi có thể
      4. Short & clear : ngắn, rõ, ổn định (temperature=0)
    """
    return f"""Answer only from the retrieved context below.
If the context is insufficient to answer the question, say you do not know and do not make up information.
Cite the source field (in brackets like [1]) when possible.
Keep your answer short, clear, and factual.
Respond in the same language as the question.

Question: {query}

Context:
{context_block}

Answer:"""


def call_llm(prompt: str) -> str:
    """
    Gọi LLM để sinh câu trả lời.

    Ưu tiên:
      1. OpenAI (OPENAI_API_KEY) — model từ LLM_MODEL env var (default: gpt-4o-mini)
      2. Google Gemini (GOOGLE_API_KEY) — gemini-1.5-flash

    temperature=0 để output ổn định, nhất quán cho evaluation.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GOOGLE_API_KEY")

    if openai_key:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=512,
        )
        return response.choices[0].message.content

    if gemini_key:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0, "max_output_tokens": 512},
        )
        return response.text

    raise RuntimeError(
        "Cần OPENAI_API_KEY hoặc GOOGLE_API_KEY.\n"
        "Tạo file .env từ .env.example và điền API key."
    )


def rag_answer(
    query: str,
    retrieval_mode: str = "dense",
    top_k_search: int = TOP_K_SEARCH,
    top_k_select: int = TOP_K_SELECT,
    use_rerank: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Pipeline RAG hoàn chỉnh: query → retrieve → (rerank) → generate.

    Args:
        query          : Câu hỏi
        retrieval_mode : "dense" | "sparse" | "hybrid"
        top_k_search   : Số chunk lấy từ vector store (search rộng)
        top_k_select   : Số chunk đưa vào prompt (sau rerank/select)
        use_rerank     : Có dùng CrossEncoder rerank không
        verbose        : In debug info

    Returns:
        Dict: answer, sources, chunks_used, query, config
    """
    config = {
        "retrieval_mode": retrieval_mode,
        "top_k_search": top_k_search,
        "top_k_select": top_k_select,
        "use_rerank": use_rerank,
    }

    # --- Bước 1: Retrieve ---
    if retrieval_mode == "dense":
        candidates = retrieve_dense(query, top_k=top_k_search)
    elif retrieval_mode == "sparse":
        candidates = retrieve_sparse(query, top_k=top_k_search)
    elif retrieval_mode == "hybrid":
        candidates = retrieve_hybrid(query, top_k=top_k_search)
    else:
        raise ValueError(f"retrieval_mode không hợp lệ: '{retrieval_mode}'. Chọn: dense|sparse|hybrid")

    if verbose:
        print(f"\n[RAG] Query: {query}")
        print(f"[RAG] Retrieved {len(candidates)} candidates (mode={retrieval_mode})")
        for i, c in enumerate(candidates[:5]):
            print(f"  [{i+1}] score={c.get('score', 0):.4f} | "
                  f"{c['metadata'].get('source', '?')} | {c['metadata'].get('section', '')}")

    # --- Bước 2: Rerank hoặc truncate ---
    if use_rerank and candidates:
        candidates = rerank(query, candidates, top_k=top_k_select)
    else:
        candidates = candidates[:top_k_select]

    if verbose:
        print(f"[RAG] After select: {len(candidates)} chunks")

    # --- Bước 3: Build context + prompt ---
    context_block = build_context_block(candidates)
    prompt = build_grounded_prompt(query, context_block)

    if verbose:
        print(f"\n[RAG] Context block:\n{context_block[:400]}...\n")

    # --- Bước 4: Generate ---
    answer = call_llm(prompt)

    # --- Bước 5: Extract sources ---
    sources = list({
        c["metadata"].get("source", "unknown")
        for c in candidates
    })

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "chunks_used": candidates,
        "config": config,
    }


# =============================================================================
# SPRINT 3: SO SÁNH BASELINE VS VARIANT
# =============================================================================

def compare_retrieval_strategies(query: str) -> None:
    """
    So sánh dense vs hybrid (vs sparse) với cùng một query.

    Dùng để justify tại sao chọn variant đó cho Sprint 3.
    A/B Rule: Chỉ đổi MỘT biến mỗi lần — kết quả mới có thể interpret được.
    """
    print(f"\n{'='*65}")
    print(f"Compare strategies | Query: {query}")
    print('='*65)

    configs = [
        {"mode": "dense",  "rerank": False, "label": "Baseline (dense)"},
        {"mode": "hybrid", "rerank": False, "label": "Variant A (hybrid, no rerank)"},
        {"mode": "hybrid", "rerank": True,  "label": "Variant B (hybrid + rerank)"},
    ]

    for cfg in configs:
        print(f"\n--- {cfg['label']} ---")
        try:
            result = rag_answer(
                query,
                retrieval_mode=cfg["mode"],
                use_rerank=cfg["rerank"],
                verbose=False,
            )
            print(f"Answer  : {result['answer'][:200]}")
            print(f"Sources : {result['sources']}")
            print(f"Chunks  : {len(result['chunks_used'])}")
        except NotImplementedError as e:
            print(f"Chưa implement: {e}")
        except Exception as e:
            print(f"Lỗi: {e}")


# =============================================================================
# MAIN — Demo và Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 2 + 3: RAG Answer Pipeline")
    print("=" * 60)

    test_queries = [
        ("SLA xử lý ticket P1 là bao lâu?",                     "dense"),
        ("Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày?", "dense"),
        ("Ai phải phê duyệt để cấp quyền Level 3?",             "dense"),
        ("ERR-403-AUTH là lỗi gì?",                              "dense"),   # abstain test
        ("Approval Matrix để cấp quyền là tài liệu nào?",       "hybrid"),  # alias test
    ]

    print("\n--- Sprint 2: Baseline Dense ---")
    for query, mode in test_queries[:4]:
        print(f"\nQuery: {query}")
        try:
            result = rag_answer(query, retrieval_mode="dense", verbose=True)
            print(f"Answer : {result['answer']}")
            print(f"Sources: {result['sources']}")
        except Exception as e:
            print(f"Lỗi: {e}")

    print("\n\n--- Sprint 3: So sánh strategies ---")
    compare_retrieval_strategies("Approval Matrix để cấp quyền là tài liệu nào?")
    compare_retrieval_strategies("ERR-403-AUTH")

    print("\n\nTiếp theo: python eval.py để chạy scorecard đầy đủ.")
