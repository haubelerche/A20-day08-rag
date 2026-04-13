import os
import re
import chromadb
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Tải các biến môi trường từ file .env
load_dotenv()

# CẤU HÌNH ĐƯỜNG DẪN VÀ THÔNG SỐ CHUNK
DOCS_DIR = Path(__file__).parent / "data" / "docs"       # Thư mục chứa file tài liệu đầu vào
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"     # Thư mục lưu trữ cơ sở dữ liệu vector

CHUNK_SIZE = 400        # Kích thước tối đa của mỗi chunk (đơn vị ước lượng token)
CHUNK_OVERLAP = 80      # Số lượng ký tự/token lặp lại giữa các chunk kế tiếp để giữ ngữ cảnh

# Khởi tạo model Local - Chạy trực tiếp trên máy, không mất phí và bảo mật dữ liệu
print("⏳ Đang khởi tạo model all-MiniLM-L6-v2...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# STEP 1: PREPROCESS - Trích xuất Metadata & Làm sạch văn bản
def preprocess_document(raw_text: str, filepath: str) -> Dict[str, Any]:
    """Tách phần Header (Metadata) và phần Body (Nội dung) của tài liệu."""
    lines = raw_text.strip().split("\n")
    # Khởi tạo giá trị mặc định cho Metadata
    metadata = {
        "source": Path(filepath).name,
        "section": "General",
        "department": "unknown",
        "effective_date": "unknown",
        "access": "internal",
    }
    
    content_lines = []
    header_done = False

    for line in lines:
        if not header_done:
            # Sử dụng Regex để tìm các thông tin dạng "Key: Value" ở đầu file
            match = re.match(r"^(Source|Department|Effective Date|Access):\s*(.*)$", line, re.IGNORECASE)
            if match:
                # Chuyển Key về dạng chuẩn (ví dụ: 'Effective Date' -> 'effective_date')
                key = match.group(1).lower().replace(" ", "_")
                value = match.group(2).strip()
                metadata[key] = value
            elif line.startswith("==="):
                # Khi gặp tiêu đề bắt đầu bằng ===, coi như kết thúc phần Header
                header_done = True
                content_lines.append(line)
        else:
            # Thu thập các dòng nội dung sau phần Header
            content_lines.append(line)

    # Chuyển danh sách dòng thành văn bản và chuẩn hóa khoảng trắng (tối đa 2 dòng trống liên tiếp)
    cleaned_text = "\n".join(content_lines).strip()
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text) 

    return {"text": cleaned_text, "metadata": metadata}

# STEP 2: CHUNK - Chia nhỏ văn bản để tối ưu hóa việc tìm kiếm
def _split_by_size(text: str, base_metadata: Dict, section: str) -> List[Dict[str, Any]]:
    """Hàm bổ trợ: Cắt nhỏ nội dung trong một Section dựa trên dung lượng tối đa."""
    chunk_chars = CHUNK_SIZE * 4      # Quy đổi ước lượng: 1 token ~ 4 ký tự
    overlap_chars = CHUNK_OVERLAP * 4
    
    # Chia nhỏ theo đoạn văn (\n\n) để tránh cắt ngang câu/ý nghĩa
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_parts = []
    current_len = 0
    overlap_tail = "" # Lưu phần cuối của chunk trước để đưa vào chunk sau

    for para in paragraphs:
        para_len = len(para)
        # Nếu thêm đoạn mới mà vượt giới hạn -> Đóng gói chunk hiện tại
        if current_len + para_len > chunk_chars and current_parts:
            chunk_text = overlap_tail + "\n\n".join(current_parts)
            chunks.append({
                "text": chunk_text.strip(),
                "metadata": {**base_metadata, "section": section}
            })
            # Tạo phần gối đầu (overlap) cho chunk tiếp theo
            overlap_tail = chunk_text[-overlap_chars:] + "\n\n" if len(chunk_text) > overlap_chars else ""
            current_parts = []
            current_len = 0

        current_parts.append(para)
        current_len += para_len + 2 # +2 cho ký tự xuống dòng

    # Lưu lại phần nội dung còn sót lại cuối cùng
    if current_parts:
        chunks.append({
            "text": (overlap_tail + "\n\n".join(current_parts)).strip(),
            "metadata": {**base_metadata, "section": section}
        })
    return chunks

def chunk_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Phân rã tài liệu dựa trên tiêu đề Section (=== ... ===)."""
    text = doc["text"]
    base_metadata = doc["metadata"].copy()
    chunks = []

    # Tách văn bản dựa trên các mốc tiêu đề Section
    parts = re.split(r"(===.*?===)", text)
    current_section = "General"
    
    for part in parts:
        part = part.strip()
        if not part: continue
        # Nếu là tiêu đề Section, cập nhật tên section hiện tại
        if re.match(r"===.*?===", part):
            current_section = part.replace("=", "").strip()
        else:
            # Nếu là nội dung, tiến hành chia nhỏ theo kích thước
            section_chunks = _split_by_size(part, base_metadata, current_section)
            chunks.extend(section_chunks)
    return chunks

# STEP 3: EMBED + STORE - Chuyển văn bản thành vector và lưu vào Database
def get_embedding(text: str) -> List[float]:
    """Chuyển đổi một đoạn văn bản thành danh sách số (vector embedding)."""
    from sentence_transformers import SentenceTransformer
    # Kỹ thuật Singleton: Chỉ khởi tạo model một lần duy nhất để tiết kiệm bộ nhớ
    if not hasattr(get_embedding, "_model"):
        get_embedding._model = SentenceTransformer("all-MiniLM-L6-v2")
    return get_embedding._model.encode(text, normalize_embeddings=True).tolist()


def build_index(docs_dir: Path = DOCS_DIR, db_dir: Path = CHROMA_DB_DIR) -> None:
    """Quy trình tổng thể: Đọc file -> Xử lý -> Cắt nhỏ -> Vector hóa -> Lưu trữ."""
    print(f"🚀 Bắt đầu Indexing tài liệu vào: {db_dir}")
    db_dir.mkdir(parents=True, exist_ok=True)
    
    # Kết nối tới cơ sở dữ liệu ChromaDB (Lưu trữ bền vững trên ổ đĩa)
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection(
        name="rag_lab",
        metadata={"hnsw:space": "cosine"} # Sử dụng công thức Cosine Similarity để so khớp
    )

    doc_files = list(docs_dir.glob("*.txt"))
    total_chunks = 0

    for filepath in doc_files:
        print(f"📄 Đang đọc: {filepath.name}")
        raw_text = filepath.read_text(encoding="utf-8")
        
        # Tiền xử lý và chia nhỏ tài liệu
        doc = preprocess_document(raw_text, str(filepath))
        chunks = chunk_document(doc)
        
        # Chuẩn bị dữ liệu theo mẻ (Batch) để đẩy vào DB tối ưu hơn
        ids, embs, docs, metas = [], [], [], []
        for i, ck in enumerate(chunks):
            ids.append(f"{filepath.stem}_{i}")
            embs.append(get_embedding(ck["text"]))
            docs.append(ck["text"])
            metas.append(ck["metadata"])
            
        # Lưu dữ liệu vào ChromaDB (Sử dụng upsert để tránh trùng lặp)
        if ids:
            collection.upsert(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
            total_chunks += len(ids)

    print(f"\n✅ Thành công! Đã lưu {total_chunks} chunks.")

# STEP 4: INSPECT - Kiểm tra và thống kê dữ liệu sau khi Index
def list_chunks(db_dir: Path = CHROMA_DB_DIR, n: int = 3):
    """In ra n mẫu dữ liệu đầu tiên để kiểm tra trực quan."""
    print("\n" + "="*50 + "\nKIỂM TRA CHUNK MẪU\n" + "="*50)
    try:
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection("rag_lab")
        res = collection.get(limit=n, include=["documents", "metadatas"])
        for i, (d, m) in enumerate(zip(res["documents"], res["metadatas"])):
            print(f"Chunk {i+1} | Nguồn: {m['source']} | Mục: {m['section']}")
            print(f"Ngày hiệu lực: {m.get('effective_date')}")
            print(f"Nội dung: {d[:150]}...\n")
    except Exception as e:
        print(f"Lỗi đọc DB: {e}")

def inspect_metadata_coverage(db_dir: Path = CHROMA_DB_DIR):
    """Thống kê sự phân bổ của các chunk theo phòng ban (department)."""
    print("--- THỐNG KÊ METADATA ---")
    try:
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection("rag_lab")
        res = collection.get(include=["metadatas"])
        depts = {}
        for m in res["metadatas"]:
            d = m.get("department", "unknown")
            depts[d] = depts.get(d, 0) + 1
        for d, count in depts.items():
            print(f"- {d}: {count} chunks")
    except Exception as e:
        print(f"Lỗi: {e}")

# Khởi chạy quy trình chính khi thực thi file này
if __name__ == "__main__":
    build_index()
    list_chunks()
    inspect_metadata_coverage()