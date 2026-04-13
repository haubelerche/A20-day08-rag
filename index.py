"""
index.py — Sprint 1: Build RAG Index
====================================
Mục tiêu Sprint 1 (60 phút):
  - Đọc và preprocess tài liệu từ data/docs/
  - Chunk tài liệu theo cấu trúc tự nhiên (heading/section)
  - Gắn metadata: source, section, department, effective_date, access
  - Embed và lưu vào vector store (ChromaDB)

Definition of Done Sprint 1:
  ✓ Script chạy được và index đủ docs
  ✓ Có ít nhất 3 metadata fields hữu ích cho retrieval
  ✓ Có thể kiểm tra chunk bằng list_chunks()
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Fix Windows console encoding cho tiếng Việt
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()


def _get_chroma_client(db_dir: Path = None):
    """
    Trả về ChromaDB client phù hợp với môi trường:
      - HttpClient  nếu CHROMA_HOST được set  (Docker / server mode)
      - PersistentClient nếu chạy local
    """
    import chromadb
    host = os.getenv("CHROMA_HOST")
    port = int(os.getenv("CHROMA_PORT", "8000"))
    if host:
        print(f"  [ChromaDB] Kết nối HTTP server {host}:{port}")
        return chromadb.HttpClient(host=host, port=port)
    path = str(db_dir) if db_dir else str(CHROMA_DB_DIR)
    print(f"  [ChromaDB] Dùng PersistentClient (local): {path}")
    return chromadb.PersistentClient(path=path)

# =============================================================================
# CẤU HÌNH
# =============================================================================

DOCS_DIR = Path(__file__).parent / "data" / "docs"
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"

# Chunking: 400 tokens ≈ 1600 chars (4 chars/token), overlap 80 tokens ≈ 320 chars
# Quyết định: ưu tiên cắt theo section heading trước, paragraph sau.
# Lý do: corpus có cấu trúc "=== Section ===" rõ ràng — giúp chunk giữ nguyên
#         ngữ cảnh của từng điều khoản mà không bị cắt ngang.
CHUNK_SIZE = 400       # tokens
CHUNK_OVERLAP = 80     # tokens

# Module-level model cache — tránh reload SentenceTransformer mỗi lần gọi
_st_model = None


def _get_st_model():
    """Load và cache SentenceTransformer model (chỉ load 1 lần per process)."""
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        print("  [Embedding] Loading paraphrase-multilingual-MiniLM-L12-v2 ...")
        _st_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        print("  [Embedding] Model loaded.")
    return _st_model


# =============================================================================
# STEP 1: PREPROCESS
# =============================================================================

def preprocess_document(raw_text: str, filepath: str) -> Dict[str, Any]:
    """
    Preprocess một tài liệu: extract metadata từ header và làm sạch nội dung.

    Format header trong data/docs/*.txt:
        TITLE (dòng chữ hoa — bỏ qua)
        Source: policy/refund-v4.pdf
        Department: CS
        Effective Date: 2026-02-01
        Access: internal
        (blank line)
        === Section 1 ===
        content...

    Returns:
        Dict với "text" (nội dung đã clean) và "metadata" dict.
    """
    lines = raw_text.strip().split("\n")
    metadata = {
        "source": filepath,
        "section": "",
        "department": "unknown",
        "effective_date": "unknown",
        "access": "internal",
    }
    content_lines = []
    header_done = False

    for line in lines:
        if not header_done:
            if line.startswith("Source:"):
                metadata["source"] = line.replace("Source:", "").strip()
            elif line.startswith("Department:"):
                metadata["department"] = line.replace("Department:", "").strip()
            elif line.startswith("Effective Date:"):
                metadata["effective_date"] = line.replace("Effective Date:", "").strip()
            elif line.startswith("Access:"):
                metadata["access"] = line.replace("Access:", "").strip()
            elif line.startswith("==="):
                header_done = True
                content_lines.append(line)
            elif line.strip() == "" or line.isupper():
                continue  # Dòng tiêu đề (chữ hoa) hoặc dòng trống trong header
        else:
            content_lines.append(line)

    cleaned_text = "\n".join(content_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)  # max 2 dòng trống

    return {"text": cleaned_text, "metadata": metadata}


# =============================================================================
# STEP 2: CHUNK
# =============================================================================

def chunk_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Chunk một tài liệu đã preprocess.

    Chiến lược 2 lớp:
      1. Split theo section heading "=== ... ===" (primary boundary)
      2. Nếu section quá dài (> CHUNK_SIZE*4 chars), _split_by_size() theo paragraph
      3. Overlap giữa chunks: giữ paragraph cuối của chunk trước

    Mỗi chunk giữ đủ metadata: source, section, department, effective_date, access.
    """
    text = doc["text"]
    base_metadata = doc["metadata"].copy()
    chunks = []

    sections = re.split(r"(===.*?===)", text)
    current_section = "General"
    current_section_text = ""

    for part in sections:
        if re.match(r"===.*?===", part):
            if current_section_text.strip():
                chunks.extend(_split_by_size(
                    current_section_text.strip(),
                    base_metadata=base_metadata,
                    section=current_section,
                ))
            current_section = part.strip("=").strip()
            current_section_text = ""
        else:
            current_section_text += part

    if current_section_text.strip():
        chunks.extend(_split_by_size(
            current_section_text.strip(),
            base_metadata=base_metadata,
            section=current_section,
        ))

    return chunks


def _split_by_size(
    text: str,
    base_metadata: Dict,
    section: str,
    chunk_chars: int = CHUNK_SIZE * 4,
    overlap_chars: int = CHUNK_OVERLAP * 4,
) -> List[Dict[str, Any]]:
    """
    Split text theo paragraph với overlap (cải tiến từ naive character split).

    Thuật toán:
      1. Split text thành paragraphs (\\n\\n boundary)
      2. Tích luỹ paragraphs cho đến khi gần đủ chunk_chars
      3. Khi đủ: emit chunk; giữ lại paragraph cuối làm overlap cho chunk tiếp
      4. Fallback: nếu paragraph đơn > chunk_chars → hard character split

    Kết quả: chunk không bao giờ cắt ngang giữa paragraph (= điều khoản/câu hỏi).
    """
    def _make(parts: List[str]) -> Dict[str, Any]:
        return {
            "text": "\n\n".join(parts),
            "metadata": {**base_metadata, "section": section},
        }

    if len(text) <= chunk_chars:
        return [_make([text])]

    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    result: List[Dict[str, Any]] = []
    current_parts: List[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        # Paragraph đơn lẻ vượt chunk_chars → hard split (không bỏ nội dung)
        if para_len > chunk_chars:
            if current_parts:
                result.append(_make(current_parts))
                current_parts = []
                current_len = 0
            for start in range(0, para_len, chunk_chars - overlap_chars):
                end = min(start + chunk_chars, para_len)
                result.append(_make([para[start:end]]))
            continue

        sep_len = 2 if current_parts else 0  # "\n\n" separator

        if current_len + sep_len + para_len > chunk_chars and current_parts:
            # Emit chunk hiện tại
            result.append(_make(current_parts))

            # Tính overlap từ cuối chunk (giữ các paragraph gần nhất ≤ overlap_chars)
            overlap_parts: List[str] = []
            overlap_len = 0
            for p in reversed(current_parts):
                cost = len(p) + (2 if overlap_parts else 0)
                if overlap_len + cost <= overlap_chars:
                    overlap_parts.insert(0, p)
                    overlap_len += cost
                else:
                    break

            current_parts = overlap_parts + [para]
            current_len = sum(len(p) for p in current_parts) + 2 * max(0, len(current_parts) - 1)
        else:
            current_parts.append(para)
            current_len += sep_len + para_len

    if current_parts:
        result.append(_make(current_parts))

    return result if result else [_make([text[:chunk_chars]])]


# =============================================================================
# STEP 3: EMBED + STORE
# =============================================================================

def get_embedding(text: str) -> List[float]:
    """
    Tạo embedding vector cho một đoạn text.

    Ưu tiên:
      1. OpenAI text-embedding-3-small (nếu có OPENAI_API_KEY trong .env)
         Dims: 1536, multilingual, nhanh, cần API key + billing
      2. Sentence Transformers paraphrase-multilingual-MiniLM-L12-v2 (local)
         Dims: 384, multilingual (hỗ trợ tiếng Việt), chạy local không cần API

    QUAN TRỌNG: Model khi index và khi query PHẢI giống nhau.
    Nếu đổi model → xóa chroma_db/ và chạy build_index() lại.
    """
    openai_key = os.getenv("OPENAI_API_KEY")

    if openai_key:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small",
        )
        return response.data[0].embedding

    # Fallback: Sentence Transformers (local, không cần API key)
    model = _get_st_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def build_index(docs_dir: Path = DOCS_DIR, db_dir: Path = CHROMA_DB_DIR) -> None:
    """
    Pipeline hoàn chỉnh: đọc docs → preprocess → chunk → embed → upsert vào ChromaDB.

    - Collection name: "rag_lab"
    - Similarity metric: cosine
    - Upsert: chạy lại không tạo duplicate (overwrite theo chunk_id)
    """
    import chromadb
    from tqdm import tqdm

    print(f"Đang build index từ: {docs_dir}")
    if not os.getenv("CHROMA_HOST"):
        db_dir.mkdir(parents=True, exist_ok=True)

    # Khởi tạo ChromaDB client (HttpClient hoặc PersistentClient)
    client = _get_chroma_client(db_dir)
    collection = client.get_or_create_collection(
        name="rag_lab",
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0
    doc_files = list(docs_dir.glob("*.txt"))

    if not doc_files:
        print(f"Không tìm thấy file .txt trong {docs_dir}")
        return

    for filepath in doc_files:
        print(f"\n  Processing: {filepath.name}")
        raw_text = filepath.read_text(encoding="utf-8")

        doc = preprocess_document(raw_text, str(filepath))
        chunks = chunk_document(doc)

        for i, chunk in tqdm(
            enumerate(chunks),
            total=len(chunks),
            desc=f"    Embedding {filepath.stem}",
            leave=False,
        ):
            chunk_id = f"{filepath.stem}_{i:03d}"
            embedding = get_embedding(chunk["text"])
            collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk["text"]],
                metadatas=[chunk["metadata"]],
            )

        total_chunks += len(chunks)
        print(f"    ✓ {len(chunks)} chunks | source={doc['metadata']['source']}")

    print(f"\n{'='*50}")
    print(f"Hoàn thành! Tổng chunks: {total_chunks}")
    print(f"ChromaDB: {db_dir} | Collection: rag_lab | Similarity: cosine")


# =============================================================================
# STEP 4: INSPECT / KIỂM TRA
# =============================================================================

def list_chunks(db_dir: Path = CHROMA_DB_DIR, n: int = 5) -> None:
    """In ra n chunk đầu tiên để kiểm tra chất lượng index."""
    try:
        client = _get_chroma_client(db_dir)
        collection = client.get_collection("rag_lab")
        results = collection.get(limit=n, include=["documents", "metadatas"])

        total = collection.count()
        print(f"\n=== Top {n}/{total} chunks trong index ===\n")
        for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
            print(f"[Chunk {i+1}]")
            print(f"  Source    : {meta.get('source', 'N/A')}")
            print(f"  Section   : {meta.get('section', 'N/A')}")
            print(f"  Eff. Date : {meta.get('effective_date', 'N/A')}")
            print(f"  Dept      : {meta.get('department', 'N/A')}")
            print(f"  Text      : {doc[:120]}...")
            print()
    except Exception as e:
        print(f"Lỗi: {e}. Hãy chạy build_index() trước.")


def inspect_metadata_coverage(db_dir: Path = CHROMA_DB_DIR) -> None:
    """Kiểm tra phân phối metadata trong toàn bộ index."""
    try:
        client = _get_chroma_client(db_dir)
        collection = client.get_collection("rag_lab")
        results = collection.get(include=["metadatas"])

        print(f"\nTổng chunks: {len(results['metadatas'])}")

        departments: Dict[str, int] = {}
        sources: Dict[str, int] = {}
        missing_date = 0

        for meta in results["metadatas"]:
            dept = meta.get("department", "unknown")
            departments[dept] = departments.get(dept, 0) + 1
            src = meta.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
            if meta.get("effective_date") in ("unknown", "", None):
                missing_date += 1

        print("\nPhân bố theo department:")
        for dept, count in sorted(departments.items()):
            print(f"  {dept}: {count} chunks")

        print("\nPhân bố theo source:")
        for src, count in sorted(sources.items()):
            print(f"  {src}: {count} chunks")

        print(f"\nChunks thiếu effective_date: {missing_date}/{len(results['metadatas'])}")

    except Exception as e:
        print(f"Lỗi: {e}. Hãy chạy build_index() trước.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 1: Build RAG Index")
    print("=" * 60)

    doc_files = list(DOCS_DIR.glob("*.txt"))
    print(f"\nTìm thấy {len(doc_files)} tài liệu:")
    for f in doc_files:
        print(f"  - {f.name}")

    # Test preprocess + chunking (không cần API key)
    print("\n--- Test preprocess + chunking ---")
    for filepath in doc_files[:2]:
        raw = filepath.read_text(encoding="utf-8")
        doc = preprocess_document(raw, str(filepath))
        chunks = chunk_document(doc)
        print(f"\nFile: {filepath.name}")
        print(f"  source={doc['metadata']['source']} | "
              f"dept={doc['metadata']['department']} | "
              f"date={doc['metadata']['effective_date']}")
        print(f"  Số chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks[:2]):
            print(f"\n  [Chunk {i+1}] Section: {chunk['metadata']['section']}")
            print(f"  {len(chunk['text'])} chars | {chunk['text'][:150]}...")

    # Build full index
    print("\n--- Build Full Index ---")
    mode = "OpenAI text-embedding-3-small" if os.getenv("OPENAI_API_KEY") \
        else "Sentence Transformers paraphrase-multilingual-MiniLM-L12-v2 (local)"
    print(f"Embedding: {mode}")

    build_index()

    print("\n--- Kiểm tra Index ---")
    list_chunks(n=3)
    inspect_metadata_coverage()

    print("\nSprint 1 hoàn thành! Tiếp theo: python rag_answer.py")
