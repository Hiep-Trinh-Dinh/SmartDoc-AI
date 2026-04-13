from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_community.vectorstores import FAISS


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def create_vector_store(docs, embedding) -> FAISS:
    return FAISS.from_documents(docs, embedding)


def save_vector_store(vector_db: FAISS, persist_dir: str) -> None:
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    vector_db.save_local(persist_dir)


def load_vector_store(persist_dir: str, embedding) -> FAISS:
    # FAISS docstore uses pickle under the hood.
    # This is safe in our intended offline/local scenario.
    return FAISS.load_local(persist_dir, embedding, allow_dangerous_deserialization=True)


def load_or_create_vector_store(docs, embedding, persist_dir: str) -> FAISS:
    persist_path = Path(persist_dir)
    if persist_path.exists() and any(persist_path.iterdir()):
        return load_vector_store(str(persist_path), embedding)
    vector_db = create_vector_store(docs, embedding)
    save_vector_store(vector_db, str(persist_path))
    return vector_db


def get_retriever(
    vector_db: FAISS,
    k: int = 3,
    *,
    search_type: str = "similarity",
    fetch_k: int | None = None,
    lambda_mult: float | None = None,
):
    search_kwargs: dict = {"k": k}
    if fetch_k is not None:
        search_kwargs["fetch_k"] = int(fetch_k)
    if lambda_mult is not None:
        search_kwargs["lambda_mult"] = float(lambda_mult)

    # LangChain supports different search types depending on vector store.
    # FAISS retriever commonly supports: "similarity" and "mmr".
    return vector_db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)