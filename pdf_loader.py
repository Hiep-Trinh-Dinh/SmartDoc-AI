from __future__ import annotations
import pytesseract

import os
import re
import unicodedata

from langchain_core.documents import Document
from langchain_community.document_loaders import PDFPlumberLoader, PyPDFLoader


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

try:
    import pypdfium2 as pdfium
except Exception:  # pragma: no cover
    pdfium = None

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\x0c", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_ocr_line(line: str) -> str:
    line = _normalize_text(line)
    if not line:
        return ""
    return line.strip()


def _clean_page_text(text: str) -> str:
    cleaned_lines = []
    for raw_line in _normalize_text(text).splitlines():
        cleaned = _clean_ocr_line(raw_line)
        if cleaned:
            cleaned_lines.append(cleaned)
    return "\n".join(cleaned_lines).strip()


def _load_pdf_text(file_path: str):
    try:
        return PDFPlumberLoader(file_path).load()
    except Exception:
        return PyPDFLoader(file_path).load()


def _should_ocr_page(text: str, min_chars: int = 40) -> bool:
    return len(_clean_page_text(text)) < min_chars


def _configure_tesseract() -> None:
    if pytesseract is None:
        return
    tesseract_cmd = os.getenv("TESSERACT_CMD")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


def _ocr_page_text(image, lang: str) -> str:
    if pytesseract is None:
        raise RuntimeError(
            "Thiếu pytesseract. Hãy cài dependencies OCR và Tesseract OCR để đọc PDF dạng ảnh."
        )
    _configure_tesseract()
    return pytesseract.image_to_string(image, lang=lang)


def _ocr_pdf_pages(file_path: str, page_numbers: set[int], lang: str = "vie+eng") -> dict[int, str]:
    if not page_numbers:
        return {}
    if pdfium is None:
        raise RuntimeError(
            "Thiếu pypdfium2. Hãy cài dependencies OCR để render PDF dạng ảnh trước khi quét chữ."
        )

    pdf = pdfium.PdfDocument(file_path)
    extracted: dict[int, str] = {}
    try:
        for page_number in sorted(page_numbers):
            page = pdf[page_number]
            bitmap = page.render(scale=2.0)
            pil_image = bitmap.to_pil()
            extracted[page_number] = _clean_page_text(_ocr_page_text(pil_image, lang=lang))
    finally:
        pdf.close()
    return extracted


def _merge_text_with_ocr(docs, file_path: str):
    page_numbers_to_ocr: set[int] = set()

    for index, doc in enumerate(docs):
        page_number = (doc.metadata or {}).get("page", index)
        if _should_ocr_page(doc.page_content or ""):
            page_numbers_to_ocr.add(int(page_number))

    if not page_numbers_to_ocr:
        return docs

    ocr_lang = os.getenv("OCR_LANG", "vie+eng")
    ocr_results = _ocr_pdf_pages(file_path, page_numbers_to_ocr, lang=ocr_lang)

    merged_docs = []
    for index, doc in enumerate(docs):
        page_number = int((doc.metadata or {}).get("page", index))
        existing_text = _clean_page_text(doc.page_content or "")
        ocr_text = ocr_results.get(page_number, "")
        final_text = existing_text if not _should_ocr_page(existing_text) else ocr_text or existing_text
        merged_docs.append(Document(page_content=final_text, metadata=dict(doc.metadata or {})))

    return merged_docs

def load_pdf(file_path: str):
    """Load a PDF into LangChain Documents.

    Tries PDFPlumber first (better layout), then falls back to PyPDF.
    If a page has little/no extractable text, OCR is used as a fallback.
    """
    docs = _load_pdf_text(file_path)
    return _merge_text_with_ocr(docs, file_path)
