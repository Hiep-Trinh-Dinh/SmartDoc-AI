from __future__ import annotations

from pathlib import Path

import streamlit as st
from rag_pipeline import ask_rag, ask_corag, ask_self_rag
from embedding import get_embedding
from llm import get_llm
from document_loader import load_document
from rag_pipeline import ask_question
from text_splitter import split_docs
from vector_store import get_retriever, get_hybrid_retriever, load_or_create_vector_store, sha256_bytes


st.set_page_config(page_title="SmartDoc AI", page_icon="📄", layout="wide")

st.title("📄 SmartDoc AI")
st.caption("Upload PDF/DOCX → hỏi đáp theo nội dung tài liệu (với Hybrid Search & Re-ranking)")
mode = st.radio(
    "Chế độ",
    ["RAG", "Co-RAG", "Hybrid RAG", "Self-RAG", "So sánh"]
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

# Sidebar for settings and history
with st.sidebar:
    st.header("⚙️ Cấu hình RAG")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", 50, 200, 100, step=50)
    
    st.divider()
    
    st.header("📜 Lịch sử tin nhắn")
    if "session_id" in st.session_state:
        session_id = st.session_state.session_id
        df = pd.read_sql_query("SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp", conn, params=(session_id,))
        if not df.empty:
            for _, row in df.iterrows():
                with st.expander(f"{row['role'].upper()} - {row['timestamp'][:19]}"):
                    st.write(row['content'])
            
            if st.button("🗑️ Xóa lịch sử"):
                if st.checkbox("Xác nhận xóa lịch sử?"):
                    cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                    conn.commit()
                    st.rerun()
        else:
            st.info("Chưa có tin nhắn")
    
    st.divider()
    
    st.header("🧹 Dọn dẹp")
    if st.button("🗑️ Xóa Vector Store"):
        if st.checkbox("Xác nhận xóa toàn bộ tài liệu đã upload?"):
            import shutil
            if FAISS_DIR.exists():
                shutil.rmtree(FAISS_DIR)
                FAISS_DIR.mkdir(parents=True, exist_ok=True)
            if UPLOAD_DIR.exists():
                shutil.rmtree(UPLOAD_DIR)
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            st.session_state.doc_hash = None
            st.session_state.rag_retriever = None
            st.session_state.corag_retrievers = None
            st.success("Đã xóa toàn bộ dữ liệu.")
            st.rerun()

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


uploaded_files = st.file_uploader(
    "Upload tài liệu (PDF, DOCX)",
    type=["pdf", "docx"],
    accept_multiple_files=True,
)

if "doc_hash" not in st.session_state:
    st.session_state.doc_hash = None

if "rag_retriever" not in st.session_state:
    st.session_state.rag_retriever = None

if "corag_retrievers" not in st.session_state:
    st.session_state.corag_retrievers = None
    


def _build_index_for_uploads(files, chunk_size, chunk_overlap):
    all_docs = []
    combined_hash = ""
    
    for uploaded_file in files:
        file_bytes = uploaded_file.getvalue()
        doc_hash = sha256_bytes(file_bytes)
        combined_hash += doc_hash
        
        safe_name = Path(uploaded_file.name).name
        file_path = UPLOAD_DIR / f"{doc_hash}_{safe_name}"
        if not file_path.exists():
            file_path.write_bytes(file_bytes)

        docs = load_document(str(file_path))
        if docs:
            # Add metadata for filtering (8.2.8)
            for doc in docs:
                doc.metadata["file_name"] = safe_name
                doc.metadata["upload_date"] = datetime.now().strftime("%Y-%m-%d")
            all_docs.extend(docs)

    if not all_docs:
        raise ValueError("Không đọc được nội dung từ các file đã upload.")

    final_hash = sha256_bytes(combined_hash.encode())
    embedding = get_embedding()
    persist_dir = FAISS_DIR / final_hash

    chunks = split_docs(all_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError("Không tách được văn bản (chunking trả về rỗng).")

    vector_db = load_or_create_vector_store(chunks, embedding, str(persist_dir))

    rag_retriever = get_retriever(
        vector_db,
        k=5,
        search_type="similarity"
    )

    corag_retrievers = [
        get_retriever(vector_db, k=8, search_type="similarity"),
        get_retriever(vector_db, k=12, search_type="mmr", fetch_k=40, lambda_mult=0.2),
    ]
    
    hybrid_retriever = get_hybrid_retriever(vector_db, chunks, k=8)

    return final_hash, rag_retriever, corag_retrievers, hybrid_retriever


if not uploaded_files:
    st.info("Hãy upload ít nhất 1 file PDF hoặc DOCX để bắt đầu.")
    st.stop()

if "hybrid_retriever" not in st.session_state:
    st.session_state.hybrid_retriever = None

try:
    # Calculate a simple hash based on file names and sizes to detect changes
    current_files_info = "".join([f"{f.name}{f.size}{chunk_size}{chunk_overlap}" for f in uploaded_files])
    current_hash = sha256_bytes(current_files_info.encode())

    if (
        st.session_state.doc_hash != current_hash
        or st.session_state.rag_retriever is None
    ):
        with st.spinner("Đang xử lý tài liệu và tạo chỉ mục…"):
            doc_hash, rag_ret, corag_rets, hybrid_ret = _build_index_for_uploads(uploaded_files, chunk_size, chunk_overlap)

        st.session_state.doc_hash = current_hash
        st.session_state.rag_retriever = rag_ret
        st.session_state.corag_retrievers = corag_rets
        st.session_state.hybrid_retriever = hybrid_ret

    st.success("Tài liệu đã sẵn sàng. Bạn có thể đặt câu hỏi.")
    if st.session_state.rag_retriever is None:
        st.error("Retriever chưa sẵn sàng.")
        st.stop()
    if "session_id" not in st.session_state:
        # Tạo session_id ổn định cho lần khởi tạo app/tab này.
        # Không tạo lại ở các lần rerun.
        st.session_state.session_id = str(uuid.uuid4())

except Exception as exc:
    st.error(f"Lỗi xử lý tài liệu: {exc}")
    st.stop()

# Control new session explicitly (tạo session mới theo nút, không reset ngoài ý muốn)
if "new_chat_requested" not in st.session_state:
    st.session_state.new_chat_requested = False

with st.sidebar:
    if st.button("➕ New chat"):
        st.session_state.new_chat_requested = True
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()



def display_answer_with_sources(answer, docs):
    st.markdown(answer)
    with st.expander("📌 Nguồn trích dẫn"):
        for i, doc in enumerate(docs):
            st.markdown(f"**Nguồn {i+1}:** {doc.metadata.get('file_name', 'Unknown')} (Trang {doc.metadata.get('page', '?')})")
            st.info(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)

query = st.text_input("Câu hỏi (Vietnamese / English / 50+ languages)")

if query:
    store_message("user", query)
    
    # Get history from DB
    df_history = pd.read_sql_query("SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp", conn, params=(st.session_state.session_id,))
    history = df_history.to_dict('records')

    try:
        with st.spinner("Đang suy luận với Ollama (Qwen2.5)…"):
            llm = get_llm()

            if mode == "RAG":
                answer, docs = ask_rag(query, st.session_state.rag_retriever, llm, history=history)
                st.subheader("RAG Answer")
                display_answer_with_sources(answer, docs)
                store_message("rag", answer)

            elif mode == "Co-RAG":
                answer, docs = ask_corag(query, st.session_state.corag_retrievers, llm, history=history)
                st.subheader("Co-RAG Answer")
                display_answer_with_sources(answer, docs)
                store_message("corag", answer)
            
            elif mode == "Hybrid RAG":
                answer, docs = ask_rag(query, st.session_state.hybrid_retriever, llm, history=history)
                st.subheader("Hybrid RAG Answer")
                display_answer_with_sources(answer, docs)
                store_message("hybrid", answer)

            elif mode == "Self-RAG":
                answer, docs, evaluation, optimized_query = ask_self_rag(query, st.session_state.corag_retrievers, llm, history=history)
                st.subheader("Self-RAG Answer")
                st.caption(f"Optimized Query: {optimized_query}")
                display_answer_with_sources(answer, docs)
                
                col_s, col_e = st.columns([1, 4])
                with col_s:
                    st.metric("Confidence Score", f"{evaluation['score']}%")
                with col_e:
                    st.info(f"**Evaluation:** {evaluation['evaluation']}")
                
                store_message("self-rag", f"{answer}\n\n[Score: {evaluation['score']}%]")

            else:
                # Compare
                rag_answer, rag_docs = ask_rag(query, st.session_state.rag_retriever, llm, history=history)
                corag_answer, corag_docs = ask_corag(query, st.session_state.corag_retrievers, llm, history=history)

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Standard RAG")
                    display_answer_with_sources(rag_answer, rag_docs)
                    store_message("rag", rag_answer)

                with col2:
                    st.subheader("Co-RAG")
                    display_answer_with_sources(corag_answer, corag_docs)
                    store_message("corag", corag_answer)

    except Exception as exc:
        st.error(f"Lỗi: {exc}")