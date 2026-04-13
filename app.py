"""
app.py — Chainlit Frontend cho RAG Pipeline
============================================
Chạy: chainlit run app.py -w

Tính năng:
  - Chat UI để truy vấn RAG pipeline
  - Hiện top k search candidates (trước rerank) tại terminal
  - Hiện top k select chunks (sau rerank/select) tại terminal
  - Cài đặt: retrieval mode, rerank, top_k_search, top_k_select
"""

import sys
import os

# Fix Windows console encoding cho tiếng Việt
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

# Thêm thư mục lab vào sys.path để import rag_answer, index
sys.path.insert(0, os.path.dirname(__file__))

import chainlit as cl
from rag_answer import (
    retrieve_dense,
    retrieve_sparse,
    retrieve_hybrid,
    rerank,
    build_context_block,
    build_grounded_prompt,
    call_llm,
    TOP_K_SEARCH,
    TOP_K_SELECT,
)


# =============================================================================
# HELPERS — In kết quả retrieval ra terminal
# =============================================================================

def print_section(title: str, char: str = "=", width: int = 70) -> None:
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_top_k_search(candidates: list, top_k: int, mode: str) -> None:
    """In danh sách candidates từ search (trước rerank) ra terminal."""
    print_section(
        f"TOP K SEARCH  —  top_k={top_k}  |  mode={mode}",
        char="=",
    )
    if not candidates:
        print("  (Không có kết quả)")
        return
    for i, c in enumerate(candidates, 1):
        meta = c.get("metadata", {})
        source = meta.get("source", "?")
        section = meta.get("section", "")
        score = c.get("score", 0.0)
        text_preview = c.get("text", "")[:120].replace("\n", " ")
        print(
            f"  [{i:02d}] score={score:.4f} | {source}"
            + (f" | {section}" if section else "")
        )
        print(f"       {text_preview}...")
    print()


def print_top_k_select(chunks: list, top_k: int, use_rerank: bool) -> None:
    """In danh sách chunks đã chọn (sau rerank/truncate) ra terminal."""
    label = "CrossEncoder rerank" if use_rerank else "top-K truncate"
    print_section(
        f"TOP K SELECT  —  top_k={top_k}  |  {label}",
        char="-",
    )
    if not chunks:
        print("  (Không có chunk nào được chọn)")
        return
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        source = meta.get("source", "?")
        section = meta.get("section", "")
        score = c.get("score", 0.0)
        text_preview = c.get("text", "")[:160].replace("\n", " ")
        print(
            f"  [{i}] score={score:.4f} | {source}"
            + (f" | {section}" if section else "")
        )
        print(f"      {text_preview}...")
    print()


# =============================================================================
# CHAINLIT — Chat Settings
# =============================================================================

@cl.on_chat_start
async def on_chat_start():
    """Khởi tạo cài đặt mặc định khi bắt đầu chat session."""
    # Lưu config vào user session
    cl.user_session.set("retrieval_mode", "hybrid")
    cl.user_session.set("use_rerank", True)
    cl.user_session.set("top_k_search", TOP_K_SEARCH)
    cl.user_session.set("top_k_select", TOP_K_SELECT)

    settings_text = (
        f"**Cài đặt mặc định:**\n"
        f"- Retrieval mode: `hybrid`\n"
        f"- Rerank (CrossEncoder): `bật`\n"
        f"- Top K Search: `{TOP_K_SEARCH}` chunks\n"
        f"- Top K Select: `{TOP_K_SELECT}` chunks\n\n"
        f"Gõ câu hỏi để bắt đầu.\n"
        f"Dùng lệnh `/config` để xem cài đặt hiện tại.\n"
        f"Dùng `/set mode=dense|sparse|hybrid rerank=on|off k_search=N k_select=N` để thay đổi."
    )
    await cl.Message(content=settings_text).send()


# =============================================================================
# CHAINLIT — Xử lý tin nhắn
# =============================================================================

@cl.on_message
async def on_message(message: cl.Message):
    query = message.content.strip()

    # --- Lệnh /config ---
    if query.lower() == "/config":
        mode = cl.user_session.get("retrieval_mode")
        rerank_on = cl.user_session.get("use_rerank")
        k_search = cl.user_session.get("top_k_search")
        k_select = cl.user_session.get("top_k_select")
        await cl.Message(content=(
            f"**Cài đặt hiện tại:**\n"
            f"- Retrieval mode: `{mode}`\n"
            f"- Rerank: `{'bật' if rerank_on else 'tắt'}`\n"
            f"- Top K Search: `{k_search}`\n"
            f"- Top K Select: `{k_select}`"
        )).send()
        return

    # --- Lệnh /set ---
    if query.lower().startswith("/set"):
        await _handle_set_command(query)
        return

    # --- Truy vấn RAG ---
    await _run_rag(query)


async def _handle_set_command(raw: str) -> None:
    """Xử lý lệnh /set để thay đổi cài đặt pipeline."""
    parts = raw.split()[1:]  # bỏ "/set"
    changes = []

    for part in parts:
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.lower().strip()
        val = val.strip()

        if key == "mode":
            if val in ("dense", "sparse", "hybrid"):
                cl.user_session.set("retrieval_mode", val)
                changes.append(f"retrieval_mode = `{val}`")
            else:
                changes.append(f"mode không hợp lệ: `{val}` (chọn: dense|sparse|hybrid)")

        elif key == "rerank":
            on = val.lower() in ("on", "true", "1", "yes")
            cl.user_session.set("use_rerank", on)
            changes.append(f"rerank = `{'bật' if on else 'tắt'}`")

        elif key == "k_search":
            try:
                n = int(val)
                if 1 <= n <= 50:
                    cl.user_session.set("top_k_search", n)
                    changes.append(f"top_k_search = `{n}`")
                else:
                    changes.append(f"k_search phải trong khoảng 1-50")
            except ValueError:
                changes.append(f"k_search không hợp lệ: `{val}`")

        elif key == "k_select":
            try:
                n = int(val)
                if 1 <= n <= 20:
                    cl.user_session.set("top_k_select", n)
                    changes.append(f"top_k_select = `{n}`")
                else:
                    changes.append(f"k_select phải trong khoảng 1-20")
            except ValueError:
                changes.append(f"k_select không hợp lệ: `{val}`")

    if changes:
        await cl.Message(content="**Đã cập nhật:**\n" + "\n".join(f"- {c}" for c in changes)).send()
    else:
        await cl.Message(content=(
            "Cú pháp: `/set mode=hybrid rerank=on k_search=10 k_select=3`"
        )).send()


async def _run_rag(query: str) -> None:
    """Chạy RAG pipeline, in terminal, gửi kết quả lên chat."""
    mode = cl.user_session.get("retrieval_mode")
    use_rerank = cl.user_session.get("use_rerank")
    top_k_search = cl.user_session.get("top_k_search")
    top_k_select = cl.user_session.get("top_k_select")

    # Thông báo đang xử lý
    thinking_msg = cl.Message(content="Đang tìm kiếm và tổng hợp câu trả lời...")
    await thinking_msg.send()

    try:
        # ── Bước 1: Retrieve (search rộng) ──────────────────────────────
        print_section(f'QUERY: "{query}"', char="#", width=70)

        if mode == "dense":
            candidates = retrieve_dense(query, top_k=top_k_search)
        elif mode == "sparse":
            candidates = retrieve_sparse(query, top_k=top_k_search)
        else:  # hybrid
            candidates = retrieve_hybrid(query, top_k=top_k_search)

        # In top k search ra terminal
        print_top_k_search(candidates, top_k=top_k_search, mode=mode)

        # ── Bước 2: Rerank hoặc truncate (select) ───────────────────────
        if use_rerank and candidates:
            selected = rerank(query, candidates, top_k=top_k_select)
        else:
            selected = candidates[:top_k_select]

        # In top k select ra terminal
        print_top_k_select(selected, top_k=top_k_select, use_rerank=use_rerank)

        # ── Bước 3: Build context + prompt ──────────────────────────────
        context_block = build_context_block(selected)
        prompt = build_grounded_prompt(query, context_block)

        # ── Bước 4: Generate ────────────────────────────────────────────
        answer = call_llm(prompt)

        # ── Bước 5: Extract sources ─────────────────────────────────────
        sources = sorted({
            c["metadata"].get("source", "unknown")
            for c in selected
        })

        # ── Gửi kết quả lên chat ────────────────────────────────────────
        source_lines = "\n".join(f"- `{s}`" for s in sources) if sources else "- (không có)"

        chunks_detail = []
        for i, c in enumerate(selected, 1):
            meta = c.get("metadata", {})
            src = meta.get("source", "?")
            sec = meta.get("section", "")
            sc = c.get("score", 0.0)
            label = f"[{i}] {src}" + (f" | {sec}" if sec else "") + f" | score={sc:.4f}"
            chunks_detail.append(label)

        chunks_text = "\n".join(f"- {l}" for l in chunks_detail)

        reply = (
            f"{answer}\n\n"
            f"---\n"
            f"**Nguồn ({len(sources)}):**\n{source_lines}\n\n"
            f"**Chunks đưa vào prompt ({len(selected)}/{top_k_search}):**\n{chunks_text}\n\n"
            f"*Config: mode=`{mode}` | rerank=`{'on' if use_rerank else 'off'}` "
            f"| k_search=`{top_k_search}` | k_select=`{top_k_select}`*"
        )

        # Cập nhật tin nhắn "đang xử lý"
        thinking_msg.content = reply
        await thinking_msg.update()

        print(f"[ANSWER] {answer[:200]}{'...' if len(answer) > 200 else ''}\n")

    except Exception as e:
        error_msg = f"Lỗi: {e}"
        thinking_msg.content = error_msg
        await thinking_msg.update()
        print(f"[ERROR] {e}")
