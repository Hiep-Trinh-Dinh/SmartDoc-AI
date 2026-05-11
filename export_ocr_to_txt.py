from __future__ import annotations

import os
from typing import List, Any


def export_documents_to_txt(docs: List[Any], output_txt_path: str) -> str:
    """Export LangChain `Document` list to a single TXT file.

    Format per page/chunk:
    [file_name] - page X
    <page_content>

    Returns the written path.
    """
    os.makedirs(os.path.dirname(output_txt_path) or ".", exist_ok=True)

    lines: List[str] = []
    for idx, d in enumerate(docs):
        meta = getattr(d, "metadata", None) or {}
        page = meta.get("page", meta.get("page_number", "?"))
        file_name = meta.get("file_name", meta.get("source", "?"))
        header = f"[{file_name}] - page {page} (chunk {idx})"
        content = getattr(d, "page_content", "") or ""
        lines.append(header)
        lines.append(content.strip())
        lines.append("")

    content_out = "\n".join(lines).strip() + "\n"

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(content_out)

    return output_txt_path

