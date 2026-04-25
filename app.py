from __future__ import annotations

from pathlib import Path

import streamlit as st
from rag_pipeline import ask_rag, ask_corag
from embedding import get_embedding
from llm import get_llm
from pdf_loader import load_pdf
from rag_pipeline import ask_question
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

import sqlite3
import uuid
from datetime import datetime

# DB setup
DB_PATH = DATA_DIR / "chats.db"
DB_PATH.parent.mkdir(exist_ok=True)
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS messages (
    session_id TEXT,
    role TEXT,
    content TEXT,
    timestamp TEXT,
    PRIMARY KEY (session_id, timestamp)
)
""")
conn.commit()

import pandas as pd

# Sidebar for history
with st.sidebar:
    st.header("📜 Lịch sử tin nhắn")
    if "session_id" in st.session_state:
        session_id = st.session_state.session_id
        df = pd.read_sql_query("SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp", conn, params=(session_id,))
        if not df.empty:
            for _, row in df.iterrows():
                with st.container():
                    role_badge = {"user": "👤", "rag": "📄", "corag": "🔄", "comparison": "⚖️"}.get(row['role'], "❓")
                    st.write(f"{role_badge} **{row['role'].upper()}**")
                    st.write(row['content'][:200] + "..." if len(row['content']) > 200 else row['content'])
            if st.button("🗑️ Xóa lịch sử"):
                cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                conn.commit()
                st.rerun()
            st.success("Đã tải lịch sử")
        else:
            st.info("Chưa có tin nhắn")
    else:
        st.info("Upload PDF để bắt đầu chat")

def store_message(role: str, content: str):
    if "session_id" not in st.session_state:
        return
    session_id = st.session_state.session_id
    timestamp = datetime.now().isoformat()
    cursor.execute(
        "INSERT OR REPLACE INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, role, content, timestamp)
    )
    conn.commit()


uploaded_file = st.file_uploader(
    "Upload tài liệu PDF (kéo & thả được)",
    type=["pdf"],
    accept_multiple_files=False,
)

if "doc_hash" not in st.session_state:
    st.session_state.doc_hash = None

if "rag_retriever" not in st.session_state:
    st.session_state.rag_retriever = None

if "corag_retrievers" not in st.session_state:
    st.session_state.corag_retrievers = None
    


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
    # RAG: baseline yếu hơn (để thấy sự khác biệt)
    rag_retriever = get_retriever(
        vector_db,
        k=2,
        search_type="similarity"
    )

    # Co-RAG: mạnh hơn rõ rệt
    corag_retrievers = [
        get_retriever(vector_db, k=6, search_type="similarity"),
        get_retriever(vector_db, k=10, search_type="mmr", fetch_k=40, lambda_mult=0.2),
    ]

    return doc_hash, rag_retriever, corag_retrievers


if uploaded_file is None:
    st.info("Hãy upload 1 file PDF để bắt đầu.")
    st.stop()

try:
    file_bytes = uploaded_file.getvalue()
    if not file_bytes:
        st.error("File upload rỗng.")
        st.stop()

    new_hash = sha256_bytes(file_bytes)
    if (
        st.session_state.doc_hash != new_hash
        or st.session_state.rag_retriever is None
    ):
        with st.spinner("Đang xử lý PDF và tạo chỉ mục (FAISS)…"):
            doc_hash, rag_ret, corag_rets = _build_index_for_upload(file_bytes, uploaded_file.name)

        st.session_state.doc_hash = doc_hash
        st.session_state.rag_retriever = rag_ret
        st.session_state.corag_retrievers = corag_rets

    st.success("Tài liệu đã sẵn sàng. Bạn có thể đặt câu hỏi.")
    if st.session_state.rag_retriever is None:
        st.error("Retriever chưa sẵn sàng.")
        st.stop()
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

except Exception as exc:
    st.error(f"Lỗi xử lý tài liệu: {exc}")
    st.stop()


query = st.text_input("Câu hỏi (Vietnamese / English / 50+ languages)")

if query:
    store_message("user", query)
    try:
        with st.spinner("Đang suy luận với Ollama (Qwen2.5)…"):
            llm = get_llm()

            if mode == "RAG":
                answer = ask_rag(query, st.session_state.rag_retriever, llm)
                st.subheader("RAG Answer")
                st.write(answer)
                store_message("rag", answer)

            elif mode == "Co-RAG":
                answer = ask_corag(query, st.session_state.corag_retrievers, llm)
                st.subheader("Co-RAG Answer")
                st.write(answer)
                store_message("corag", answer)

            else:
                # chạy trước → đảm bảo cùng điều kiện
                rag_answer = ask_rag(query, st.session_state.rag_retriever, llm)
                corag_answer = ask_corag(query, st.session_state.corag_retrievers, llm)

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("RAG")
                    st.write(rag_answer)
                    store_message("rag", rag_answer)

                with col2:
                    st.subheader("Co-RAG")
                    st.write(corag_answer)
                    store_message("corag", corag_answer)

    except Exception as exc:
        st.error(f"Lỗi: {exc}")