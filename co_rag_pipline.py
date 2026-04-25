from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Any


# =========================
# 1. Sub-query generation (đa dạng thật)
# =========================
def generate_sub_queries(query: str) -> List[str]:
    return list(set([
        query,
        f"What is {query}?",
        f"Key concepts of {query}",
        f"Technical details of {query}",
        f"Limitations of {query}",
        f"Examples and use cases of {query}",
    ]))


# =========================
# 2. Hash nhẹ (không mất diversity)
# =========================
def _hash(text: str):
    return hashlib.md5(text.strip().encode()).hexdigest()


# =========================
# 3. Retriever wrapper (support nhiều loại)
# =========================
def _retrieve(r, query):
    try:
        if hasattr(r, "invoke"):
            return r.invoke(query)
        return r.get_relevant_documents(query)
    except Exception:
        return []


# =========================
# 4. Multi-retriever + parallel
# =========================
def retrieve_all(
    retrievers: List[Any],
    queries: List[str],
    max_workers: int = 8
):
    results = []

    def _task(r, q):
        docs = _retrieve(r, q)
        return docs or []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_task, r, q)
            for r in retrievers
            for q in queries
        ]

        for fut in as_completed(futures):
            try:
                results.extend(fut.result())
            except Exception:
                continue

    return results


# =========================
# 5. Normalize doc
# =========================
def _extract(doc):
    text = getattr(doc, "page_content", None) or str(doc)
    meta = getattr(doc, "metadata", {}) or {}
    score = meta.get("score", 0.5)
    return text.strip(), score


# =========================
# 6. Deduplicate (nhẹ)
# =========================
def dedupe_docs(docs):
    seen = set()
    out = []

    for d in docs:
        text, score = _extract(d)
        h = _hash(text)

        if h in seen:
            continue
        seen.add(h)

        out.append((text, score))

    return out


# =========================
# 7. MMR (diversity-aware)
# =========================
def mmr_select(docs, k=8, lambda_param=0.7):
    selected = []
    candidates = docs.copy()

    while candidates and len(selected) < k:
        if not selected:
            selected.append(candidates.pop(0))
            continue

        best_doc = None
        best_score = -1e9

        for doc in candidates:
            relevance = doc[1]

            diversity = max([
                _similarity(doc[0], s[0])
                for s in selected
            ] or [0])

            mmr = lambda_param * relevance - (1 - lambda_param) * diversity

            if mmr > best_score:
                best_score = mmr
                best_doc = doc

        selected.append(best_doc)
        candidates.remove(best_doc)

    return selected


# =========================
# 8. Similarity (simple fallback)
# =========================
def _similarity(a: str, b: str):
    # đơn giản để tránh tốn compute
    return len(set(a.split()) & set(b.split())) / (len(set(a.split())) + 1)


# =========================
# 9. Rerank (LLM-based nhẹ)
# =========================
def rerank_with_llm(llm, query, docs, top_k=6):
    scored = []

    for text, _ in docs:
        prompt = f"""
Score relevance from 0 to 1.

Query: {query}
Document: {text[:1000]}

Score:
"""
        try:
            score = float(llm.invoke(prompt).strip())
        except:
            score = 0.5

        scored.append((text, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# =========================
# 10. Build context (token-aware)
# =========================
def build_context(docs, max_chars=12000):
    context = []

    total = 0
    for text, _ in docs:
        if total + len(text) > max_chars:
            break
        context.append(text)
        total += len(text)

    return "\n\n".join(context)


# =========================
# 11. Prompt builder
# =========================
def build_prompt(query, context, lang="en"):
    if lang == "vi":
        return f"""
Bạn là trợ lý QA tài liệu.
Chỉ dùng thông tin trong Context.

Context:
{context}

Câu hỏi: {query}
Trả lời:
"""
    return f"""
You are a document QA assistant.
Use ONLY the context.

Context:
{context}

Question: {query}
Answer:
"""


# =========================
# 12. FINAL Co-RAG
# =========================
def co_rag(
    query: str,
    retrievers,
    llm,
    max_context_chars=12000,
):
    retrievers = list(retrievers) if isinstance(retrievers, (list, tuple)) else [retrievers]

    # Step 1: sub queries
    sub_queries = generate_sub_queries(query)

    # Step 2: retrieve
    raw_docs = retrieve_all(retrievers, sub_queries)

    # Step 3: dedupe
    docs = dedupe_docs(raw_docs)

    # Step 4: sort initial
    docs.sort(key=lambda x: x[1], reverse=True)

    # Step 5: MMR
    docs = mmr_select(docs, k=10)

    # Step 6: rerank
    docs = rerank_with_llm(llm, query, docs, top_k=6)

    # Step 7: build context
    context = build_context(docs, max_context_chars)

    # Step 8: answer
    prompt = build_prompt(query, context)
    return llm.invoke(prompt)