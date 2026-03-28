from __future__ import annotations

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
):
    lang = _safe_detect_language(query)
    docs = _get_relevant_docs(retriever, query)
    context = "\n\n".join([(d.page_content or "").strip() for d in docs if (d.page_content or "").strip()])
    if len(context) > max_context_chars:
        context = context[:max_context_chars]

    prompt = _build_prompt(context=context, query=query, lang=lang)
    return llm.invoke(prompt)