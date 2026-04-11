from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

try:
    from langdetect import detect as _detect
except Exception:  # langdetect missing or misinstalled
    _detect = None


def _safe_detect_language(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "vi"
    try:
        if _detect is None:
            return "vi"
        return _detect(text)
    except Exception:
        return "vi"


def _get_relevant_docs(retriever: Any, query: str):
    # LangChain retriever API differs by version.
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    return retriever.get_relevant_documents(query)


def _dedupe_docs(docs: list[Any]) -> list[Any]:
    """Best-effort doc de-duplication while preserving order."""
    seen: set[tuple[str, str]] = set()
    out: list[Any] = []
    for d in docs:
        page_content = (getattr(d, "page_content", None) or "").strip()
        metadata = getattr(d, "metadata", None) or {}
        meta_key = ""
        try:
            # Stable-ish key: common fields in LangChain Document metadata
            meta_key = str(
                (
                    metadata.get("source"),
                    metadata.get("file_path"),
                    metadata.get("page"),
                    metadata.get("loc"),
                )
            )
        except Exception:
            meta_key = ""

        key = (meta_key, page_content)
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _get_relevant_docs_multi(
    retrievers: list[Any],
    query: str,
    *,
    parallel: bool = True,
    max_workers: int | None = None,
) -> list[Any]:
    if not retrievers:
        return []

    if not parallel or len(retrievers) == 1:
        merged: list[Any] = []
        for r in retrievers:
            try:
                merged.extend(_get_relevant_docs(r, query) or [])
            except Exception:
                continue
        return _dedupe_docs(merged)

    merged: list[Any] = []
    try:
        with ThreadPoolExecutor(max_workers=max_workers or min(8, len(retrievers))) as ex:
            futures = [ex.submit(_get_relevant_docs, r, query) for r in retrievers]
            for fut in as_completed(futures):
                try:
                    merged.extend(fut.result() or [])
                except Exception:
                    continue
    except Exception:
        # Safety fallback: sequential
        for r in retrievers:
            try:
                merged.extend(_get_relevant_docs(r, query) or [])
            except Exception:
                continue

    return _dedupe_docs(merged)


def _build_prompt(context: str, query: str, lang: str) -> str:
    if lang == "vi":
        return (
            "Bạn là trợ lý hỏi-đáp tài liệu. Chỉ dùng THÔNG TIN trong Context để trả lời. "
            "Nếu Context không đủ thông tin, hãy nói rõ bạn không tìm thấy trong tài liệu.\n\n"
            f"Context:\n{context}\n\n"
            f"Câu hỏi: {query}\n"
            "Trả lời (ngắn gọn, đúng trọng tâm):"
        )
    # default to English
    return (
        "You are a document QA assistant. Answer ONLY using the information in the Context. "
        "If the Context is insufficient, say you couldn't find it in the document.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer (concise, factual):"
    )


def ask_question(
    query: str,
    retriever,
    llm,
    max_context_chars: int = 12000,
    parallel_retrieval: bool | None = None,
    max_retrieval_workers: int | None = None,
):
    lang = _safe_detect_language(query)

    # Allow passing either a single retriever or a list/tuple of retrievers.
    retrievers = list(retriever) if isinstance(retriever, (list, tuple)) else [retriever]

    if parallel_retrieval is None:
        parallel_retrieval = os.getenv("RAG_PARALLEL_RETRIEVAL", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
    if max_retrieval_workers is None:
        try:
            max_retrieval_workers = int(os.getenv("RAG_MAX_RETRIEVAL_WORKERS", "0")) or None
        except Exception:
            max_retrieval_workers = None

    docs = _get_relevant_docs_multi(
        retrievers,
        query,
        parallel=bool(parallel_retrieval),
        max_workers=max_retrieval_workers,
    )
    context = "\n\n".join([(d.page_content or "").strip() for d in docs if (d.page_content or "").strip()])
    if len(context) > max_context_chars:
        context = context[:max_context_chars]

    prompt = _build_prompt(context=context, query=query, lang=lang)
    return llm.invoke(prompt)