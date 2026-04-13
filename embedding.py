from __future__ import annotations

import os
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embedding(
    model_name: str | None = None,
    device: str | None = None,
    cache_dir: str | None = None,
):
    """Return a multilingual embedding model (offline after first download)."""
    model_name = model_name or os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )

    device = device or os.getenv("EMBEDDING_DEVICE")
    cache_dir = cache_dir or os.getenv("HF_HOME")
    if not cache_dir:
        cache_dir = str(Path(__file__).resolve().parent / "data" / "hf_cache")

    model_kwargs = {"device": device} if device else {}
    encode_kwargs = {"normalize_embeddings": True}

    return HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=cache_dir,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )