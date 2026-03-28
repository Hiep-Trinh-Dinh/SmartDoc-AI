from __future__ import annotations

from pathlib import Path

import streamlit as st

from embedding import get_embedding
from llm import get_llm
from pdf_loader import load_pdf
from rag_pipeline import ask_question
from text_splitter import split_docs
from vector_store import get_retriever, load_or_create_vector_store, sha256_bytes


st.set_page_config(page_title="SmartDoc AI", page_icon="📄", layout="centered")
st.title("📄 SmartDoc AI")
st.caption("Upload PDF → hỏi đáp theo nội dung tài liệu (offline/local)")


DATA_DIR = Path(__file__).resolve().parent / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
FAISS_DIR = DATA_DIR / "faiss"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FAISS_DIR.mkdir(parents=True, exist_ok=True)


uploaded_file = st.file_uploader(
    "Upload tài liệu PDF (kéo & thả được)",
    type=["pdf"],
    accept_multiple_files=False,
)

if "doc_hash" not in st.session_state:
    st.session_state.doc_hash = None
    st.session_state.retriever = None


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
    retriever = get_retriever(vector_db, k=3)
    return doc_hash, retriever


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

    st.success("Tài liệu đã sẵn sàng. Bạn có thể đặt câu hỏi.")

except Exception as exc:
    st.error(f"Lỗi xử lý tài liệu: {exc}")
    st.stop()


query = st.text_input("Câu hỏi (Vietnamese / English / 50+ languages)")

if query:
    try:
        with st.spinner("Đang suy luận với Ollama (Qwen2.5)…"):
            llm = get_llm()
            answer = ask_question(query, st.session_state.retriever, llm)
        st.subheader("Trả lời")
        st.write(answer)
    except Exception as exc:
        st.error(
            "Lỗi khi gọi model. Hãy chắc rằng Ollama đang chạy và model đã được pull.\n\n"
            f"Chi tiết: {exc}"
        )