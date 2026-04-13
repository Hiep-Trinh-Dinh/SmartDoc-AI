from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# ---------------------------
# 1. Generate sub-queries tốt hơn
# ---------------------------
def generate_sub_queries(query: str):
    return [
        query,
        f"Explain in detail: {query}",
        f"Technical explanation of {query}",
        f"Context and background of {query}",
    ]


# ---------------------------
# 2. Hash để dedupe mạnh hơn
# ---------------------------
def _hash_text(text: str) -> str:
    return hashlib.md5(text.strip().encode()).hexdigest()


# ---------------------------
# 3. Retrieve song song + giữ score
# ---------------------------
def retrieve_multi_context(sub_queries, top_k=3, max_workers=4):
    all_docs = []

    def _retrieve(q):
        emb = get_embedding(q)
        docs = search_vector_store(emb, top_k=top_k)
        return docs

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_retrieve, q) for q in sub_queries]

        for fut in as_completed(futures):
            try:
                docs = fut.result()
                all_docs.extend(docs)
            except Exception:
                continue

    return all_docs


# ---------------------------
# 4. Merge + dedupe + ranking
# ---------------------------
def merge_rank_contexts(contexts, max_docs=6):
    seen = set()
    scored_docs = []

    for c in contexts:
        # nếu vector store trả về tuple (text, score)
        if isinstance(c, tuple):
            text, score = c
        else:
            text, score = c, 0.5  # fallback

        h = _hash_text(text)

        if h in seen:
            continue
        seen.add(h)

        scored_docs.append((text, score))

    # sort theo score giảm dần
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [t for t, _ in scored_docs[:max_docs]]


# ---------------------------
# 5. Build context thông minh
# ---------------------------
def build_context(docs, max_chars=12000):
    context = ""
    for d in docs:
        if len(context) + len(d) > max_chars:
            break
        context += d.strip() + "\n\n"
    return context.strip()


# ---------------------------
# 6. Final Co-RAG pipeline
# ---------------------------
def run_co_rag(query):
    sub_queries = generate_sub_queries(query)

    contexts = retrieve_multi_context(sub_queries)

    contexts = merge_rank_contexts(contexts)

    final_context = build_context(contexts)

    prompt = f"""
You are a document QA assistant.
Answer ONLY using the provided context.
If insufficient, say you don't know.

Context:
{final_context}

Question: {query}

Answer:
"""

    answer = call_llm(prompt)

    return answer, contexts