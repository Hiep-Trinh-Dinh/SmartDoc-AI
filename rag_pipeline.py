from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Dict
from sentence_transformers import CrossEncoder

try:
    from langdetect import detect as _detect
except Exception:
    _detect = None

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _reranker

def _generate_sub_queries(query: str, llm=None):
    if llm:
        lang = _safe_detect_language(query)
        # Hướng dẫn LLM sinh query đa dạng để quét cả tổng thể lẫn chi tiết
        prompt = (
            f"Tạo 3 câu hỏi tìm kiếm khác nhau cho: '{query}'. "
            f"Trong đó có 1 câu hỏi tập trung vào từ khóa chính, 1 câu hỏi tập trung vào cấu trúc/tổng thể tài liệu. "
            "Chỉ trả về danh sách câu hỏi."
        )
        try:
            res = llm.invoke(prompt).strip().split("\n")
            queries = [q.strip("- ").strip() for q in res if q.strip()]
            return list(set([query] + queries))
        except:
            pass
    return [query, f"Cấu trúc và mục lục của {query}", f"Tóm tắt {query}"]

def rerank_docs(query: str, docs: List[Any], top_k: int = 10, llm=None):
    if not docs:
        return []
    
    # 1. Fast Rerank with CrossEncoder (Local, very fast)
    try:
        reranker = get_reranker()
        doc_texts = [d.page_content for d in docs]
        pairs = [[query, text] for text in doc_texts]
        scores = reranker.predict(pairs)
        for i, doc in enumerate(docs):
            doc.metadata["relevance_score"] = float(scores[i])
        docs = sorted(docs, key=lambda x: x.metadata["relevance_score"], reverse=True)
    except Exception:
        pass

    # 2. Refine with LLM (Only for top candidates, in ONE batch call)
    lang = _safe_detect_language(query)
    if lang == "vi" and llm and len(docs) > 1:
        candidates = docs[:10] # Only refine top 10
        list_text = ""
        for i, d in enumerate(candidates):
            list_text += f"DOC_{i}: {d.page_content[:300]}\n"
        
        prompt = (
            f"Câu hỏi: {query}\n\n"
            f"Danh sách đoạn văn:\n{list_text}\n"
            "Đánh giá độ liên quan của từng đoạn (0-10). "
            "Trả về định dạng: DOC_0: điểm, DOC_1: điểm... Chỉ trả về kết quả này."
        )
        try:
            res = llm.invoke(prompt).strip()
            # Parse scores (simple regex)
            import re
            for i in range(len(candidates)):
                match = re.search(f"DOC_{i}:\s*(\d+\.?\d*)", res)
                if match:
                    candidates[i].metadata["relevance_score"] = float(match.group(1))
            
            return sorted(candidates, key=lambda x: x.metadata["relevance_score"], reverse=True)[:top_k]
        except:
            return docs[:top_k]
            
    return docs[:top_k]

def ask_corag(query: str, retriever, llm, history: List[Dict] = None, max_context_chars: int = 15000):
    retrievers = list(retriever) if isinstance(retriever, (list, tuple)) else [retriever]
    sub_queries = _generate_sub_queries(query, llm)

    all_docs = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_get_relevant_docs_multi, retrievers, q) for q in sub_queries]
        for fut in as_completed(futures):
            all_docs.extend(fut.result() or [])

    docs = _dedupe_docs(all_docs)
    
    # Nếu câu hỏi có từ khóa "bao nhiêu", "tổng số", "liệt kê" -> Lấy nhiều context hơn
    is_global_query = any(k in query.lower() for k in ["bao nhiêu", "tổng", "liệt kê", "danh sách", "tất cả"])
    top_k = 15 if is_global_query else 8
    
    docs = rerank_docs(query, docs, top_k=top_k, llm=llm)

    context_parts = []
    for d in docs:
        meta = f"[Nguồn: {d.metadata.get('file_name','?')}, Trang: {d.metadata.get('page','?')}]"
        context_parts.append(f"{meta}\n{d.page_content.strip()}")
    
    context = "\n\n---\n\n".join(context_parts)[:max_context_chars]
    lang = _safe_detect_language(query)
    prompt = _build_prompt(context=context, query=query, lang=lang, history=history)
    answer = llm.invoke(prompt)
    
    return answer, docs

def _build_prompt(context: str, query: str, lang: str, history: List[Dict] = None) -> str:
    history_str = ""
    if history:
        for msg in history[-5:]:
            role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
            history_str += f"{role}: {msg['content']}\n"

    if lang == "vi":
        return (
            "Bạn là trợ lý phân tích tài liệu chuyên nghiệp. Hãy trả lời dựa trên các quy tắc sau:\n"
            "1. CĂN CỨ: Chỉ sử dụng thông tin trong phần 'Ngữ cảnh' được cung cấp.\n"
            "2. ĐỘ CHÍNH XÁC: Đối với các câu hỏi về số lượng hoặc danh sách, hãy rà soát kỹ các tiêu đề, số thứ tự và định dạng danh sách (1, 2, 3... hoặc a, b, c...).\n"
            "3. PHÂN CẤP: Phân biệt rõ mục lớn (Chương/Điều/Phần) và mục con (Khoản/Điểm). Không lấy số lượng mục con để trả lời cho tổng số mục lớn.\n"
            "4. TRUNG THỰC: Nếu tài liệu bị thiếu hoặc không đủ thông tin để khẳng định một con số, hãy nêu rõ những gì bạn thấy thay vì phỏng đoán.\n"
            "5. TRÌNH BÀY: Trình bày súc tích, chuyên nghiệp, có dẫn chứng rõ ràng.\n\n"
            f"Lịch sử:\n{history_str}\n"
            f"Ngữ cảnh:\n{context}\n\n"
            f"Câu hỏi: {query}\n"
            "Câu trả lời:"
        )
    return (
        "You are a professional document analysis assistant. Rules:\n"
        "1. Context-based: Use only the provided context.\n"
        "2. Precision: For quantities or lists, check headers and numbering carefully.\n"
        "3. Hierarchy: Distinguish between main sections and sub-items.\n"
        "4. Honesty: If information is incomplete, state what is available instead of guessing.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

def ask_rag(query: str, retriever, llm, history: List[Dict] = None, max_context_chars: int = 12000):
    docs = _get_relevant_docs_multi(
        [retriever[0]] if isinstance(retriever, (list, tuple)) else [retriever],
        query,
        parallel=False
    )
    docs = _dedupe_docs(docs)
    docs = rerank_docs(query, docs, top_k=8, llm=llm)
    context = "\n\n".join([f"[Nguồn: {d.metadata.get('file_name','?')}] {d.page_content.strip()}" for d in docs])[:max_context_chars]
    prompt = _build_prompt(context=context, query=query, lang=_safe_detect_language(query), history=history)
    return llm.invoke(prompt), docs

def _safe_detect_language(text: str) -> str:
    text = (text or "").strip()
    if not text: return "vi"
    try:
        return _detect(text) if _detect else "vi"
    except: return "vi"

def _get_relevant_docs_multi(retrievers, query, parallel=True):
    merged = []
    for r in retrievers:
        try:
            res = r.invoke(query) if hasattr(r, "invoke") else r.get_relevant_documents(query)
            merged.extend(res or [])
        except: continue
    return merged

def _dedupe_docs(docs):
    seen = set()
    out = []
    for d in docs:
        content = d.page_content.strip()
        key = (d.metadata.get("file_name"), d.metadata.get("page"), content[:100])
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out

def rewrite_query(query: str, llm, history: List[Dict] = None) -> str:
    lang = _safe_detect_language(query)
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in (history[-3:] if history else [])])
    prompt = f"Dựa trên lịch sử: {history_str}\nViết lại câu hỏi để tìm kiếm tài liệu tốt hơn: {query}"
    try: return llm.invoke(prompt).strip() or query
    except: return query

def evaluate_answer(query: str, context: str, answer: str, llm) -> Dict[str, Any]:
    prompt = f"Câu hỏi: {query}\nNgữ cảnh: {context[:1500]}\nCâu trả lời: {answer}\nĐánh giá độ chính xác (0-100) và lý do (Format: Score: X | Reason: Y):"
    try:
        res = llm.invoke(prompt)
        import re
        score = int(re.search(r"Score:\s*(\d+)", res).group(1))
        return {"score": score, "evaluation": res}
    except: return {"score": 70, "evaluation": "N/A"}

def ask_self_rag(query, retriever, llm, history=None, max_context_chars=15000):
    opt_query = rewrite_query(query, llm, history)
    ans, docs = ask_corag(opt_query, retriever, llm, history, max_context_chars)
    eval_res = evaluate_answer(opt_query, "\n".join([d.page_content for d in docs]), ans, llm)
    return ans, docs, eval_res, opt_query

def ask_question(query, retriever, llm, history=None, max_context_chars=12000):
    return ask_corag(query, retriever, llm, history, max_context_chars)[0]
