from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3

import streamlit as st

from embedding import get_embedding
from llm import get_llm
from pdf_loader import load_pdf
from rag_pipeline import ask_corag, ask_rag
from text_splitter import split_docs
from vector_store import get_retriever, load_or_create_vector_store, sha256_bytes


st.set_page_config(page_title="SmartDoc AI", page_icon="📄", layout="centered")
st.title("📄 SmartDoc AI")
st.caption("Upload PDF → hỏi đáp theo nội dung tài liệu (offline/local)")
mode = st.radio(
    "Chế độ",
    ["RAG", "Co-RAG", "So sánh"]
)

DATA_DIR = Path(__file__).resolve().parent / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
FAISS_DIR = DATA_DIR / "faiss"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FAISS_DIR.mkdir(parents=True, exist_ok=True)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _db_path() -> Path:
    return DATA_DIR / "chat_history.db"


def _db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_db() -> None:
    with _db_connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_hash TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_conversations_doc_hash_updated_at
            ON conversations(doc_hash, updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_messages_conversation_id_id
            ON messages(conversation_id, id);
            """
        )


def _list_conversations(doc_hash: str, limit: int = 50) -> list[sqlite3.Row]:
    with _db_connect() as conn:
        cur = conn.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            WHERE doc_hash = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (doc_hash, int(limit)),
        )
        return list(cur.fetchall())


def _create_conversation(doc_hash: str, title: str) -> int:
    now = _utcnow_iso()
    with _db_connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO conversations(doc_hash, title, created_at, updated_at)
            VALUES(?, ?, ?, ?)
            """,
            (doc_hash, title, now, now),
        )
        return int(cur.lastrowid)


def _touch_conversation(conversation_id: int) -> None:
    with _db_connect() as conn:
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (_utcnow_iso(), int(conversation_id)),
        )


def _set_conversation_title(conversation_id: int, title: str) -> None:
    with _db_connect() as conn:
        conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (title, int(conversation_id)),
        )


def _load_messages(conversation_id: int) -> list[sqlite3.Row]:
    with _db_connect() as conn:
        cur = conn.execute(
            """
            SELECT id, role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (int(conversation_id),),
        )
        return list(cur.fetchall())


def _add_message(conversation_id: int, role: str, content: str) -> None:
    now = _utcnow_iso()
    with _db_connect() as conn:
        conn.execute(
            """
            INSERT INTO messages(conversation_id, role, content, created_at)
            VALUES(?, ?, ?, ?)
            """,
            (int(conversation_id), role, content, now),
        )
    _touch_conversation(conversation_id)


def _format_chat_history(messages: list[sqlite3.Row], max_chars: int = 4000) -> str:
    lines: list[str] = []
    for m in messages:
        role = "User" if m["role"] == "user" else "Assistant"
        text = (m["content"] or "").strip()
        if text:
            lines.append(f"{role}: {text}")
    history = "\n".join(lines).strip()
    if len(history) > max_chars:
        history = history[-max_chars:]
    return history


def _build_retrieval_query(query: str, previous_messages: list[sqlite3.Row]) -> str:
    q = (query or "").strip()
    if not q:
        return q

    prev_user = [
        m["content"]
        for m in previous_messages
        if m["role"] == "user" and (m["content"] or "").strip()
    ]
    if not prev_user:
        return q

    if len(q) < 80:
        tail = "\n".join([t.strip() for t in prev_user[-2:]])
        return f"{tail}\n{q}".strip()

    return q


uploaded_file = st.file_uploader(
    "Upload tài liệu PDF (kéo & thả được)",
    type=["pdf"],
    accept_multiple_files=False,
)

if "doc_hash" not in st.session_state:
    st.session_state.doc_hash = None
    st.session_state.retriever = None

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "conversation_doc_hash" not in st.session_state:
    st.session_state.conversation_doc_hash = None


def _build_index_for_upload(file_bytes: bytes, file_name: str):
    doc_hash = sha256_bytes(file_bytes)
    safe_name = Path(file_name).name
    pdf_path = UPLOAD_DIR / f"{doc_hash}_{safe_name}"
    pdf_path.write_bytes(file_bytes)

    embedding = get_embedding()
    persist_dir = FAISS_DIR / doc_hash

    docs = load_pdf(str(pdf_path))
    if not docs:
        raise ValueError("Không đọc được nội dung PDF (tài liệu rỗng hoặc bị lỗi).")

    chunks = split_docs(docs)
    if not chunks:
        raise ValueError("Không tách được văn bản từ PDF (chunking trả về rỗng).")

    vector_db = load_or_create_vector_store(chunks, embedding, str(persist_dir))

    # Two retrieval strategies: fast similarity + diversity-biased MMR.
    # These can be executed in parallel inside rag_pipeline.ask_question().
    retrievers = [
        get_retriever(vector_db, k=3, search_type="similarity"),
        get_retriever(vector_db, k=5, search_type="mmr", fetch_k=20, lambda_mult=0.5),
    ]
    return doc_hash, retrievers


if uploaded_file is None:
    st.info("Hãy upload 1 file PDF để bắt đầu.")
    st.stop()

try:
    file_bytes = uploaded_file.getvalue()
    if not file_bytes:
        st.error("File upload rỗng.")
        st.stop()

    new_hash = sha256_bytes(file_bytes)
    if st.session_state.doc_hash != new_hash or st.session_state.retriever is None:
        with st.spinner("Đang xử lý PDF và tạo chỉ mục (FAISS)…"):
            doc_hash, retriever = _build_index_for_upload(file_bytes, uploaded_file.name)
        st.session_state.doc_hash = doc_hash
        st.session_state.retriever = retriever

    st.success("Tài liệu đã sẵn sàng. Bạn có thể chat bên dưới.")

except Exception as exc:
    st.error(f"Lỗi xử lý tài liệu: {exc}")
    st.stop()


_init_db()

# Reset conversation selection when switching PDFs.
if st.session_state.conversation_doc_hash != st.session_state.doc_hash:
    st.session_state.conversation_doc_hash = st.session_state.doc_hash
    st.session_state.conversation_id = None

st.sidebar.subheader("Lịch sử trò chuyện")
conversations = _list_conversations(st.session_state.doc_hash)

if st.sidebar.button("Tạo hội thoại mới"):
    st.session_state.conversation_id = _create_conversation(st.session_state.doc_hash, "Chat mới")
    conversations = _list_conversations(st.session_state.doc_hash)

conversation_id = st.session_state.conversation_id

if conversations:
    conv_ids = [int(c["id"]) for c in conversations]
    title_by_id = {int(c["id"]): str(c["title"]) for c in conversations}
    label_by_id = {
        int(c["id"]): f"{c['updated_at'].replace('T', ' ')[:19]} - {c['title']}" for c in conversations
    }

    if conversation_id is None:
        conversation_id = conv_ids[0]
        st.session_state.conversation_id = conversation_id

    try:
        current_index = conv_ids.index(int(conversation_id))
    except ValueError:
        current_index = 0

    selected_id = st.sidebar.selectbox(
        "Chọn hội thoại",
        options=conv_ids,
        index=current_index,
        format_func=lambda cid: label_by_id.get(int(cid), str(cid)),
    )
    st.session_state.conversation_id = int(selected_id)
    conversation_id = st.session_state.conversation_id
else:
    title_by_id = {}
    st.sidebar.info("Chưa có lịch sử cho tài liệu này.")

messages: list[sqlite3.Row] = _load_messages(conversation_id) if conversation_id else []

for m in messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_query = st.chat_input("Nhập câu hỏi…")

if user_query:
    try:
        previous_messages = list(messages)

        if not conversation_id:
            title = (user_query or "").strip()[:60] or "Chat mới"
            conversation_id = _create_conversation(st.session_state.doc_hash, title)
            st.session_state.conversation_id = conversation_id
        else:
            if title_by_id.get(int(conversation_id)) == "Chat mới" and not previous_messages:
                _set_conversation_title(
                    int(conversation_id),
                    (user_query or "").strip()[:60] or "Chat mới",
                )

        with st.chat_message("user"):
            st.markdown(user_query)
        _add_message(int(conversation_id), "user", user_query)

        chat_history = _format_chat_history(previous_messages[-8:])
        retrieval_query = _build_retrieval_query(user_query, previous_messages)

        with st.chat_message("assistant"):
            with st.spinner("Đang suy luận với Ollama…"):
                llm = get_llm()

                if mode == "RAG":
                    answer = ask_rag(
                        user_query,
                        st.session_state.retriever,
                        llm,
                        chat_history=chat_history,
                        retrieval_query=retrieval_query,
                    )
                    st.markdown(answer)

                elif mode == "Co-RAG":
                    answer = ask_corag(
                        user_query,
                        st.session_state.retriever,
                        llm,
                        chat_history=chat_history,
                        retrieval_query=retrieval_query,
                    )
                    st.markdown(answer)

                else:
                    rag_answer = ask_rag(
                        user_query,
                        st.session_state.retriever,
                        llm,
                        chat_history=chat_history,
                        retrieval_query=retrieval_query,
                    )
                    corag_answer = ask_corag(
                        user_query,
                        st.session_state.retriever,
                        llm,
                        chat_history=chat_history,
                        retrieval_query=retrieval_query,
                    )
                    answer = f"### RAG\n{rag_answer}\n\n### Co-RAG\n{corag_answer}"
                    st.markdown(answer)

        _add_message(int(conversation_id), "assistant", answer)

    except Exception as exc:
        st.error(f"Lỗi: {exc}")