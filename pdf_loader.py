from __future__ import annotations

from langchain_community.document_loaders import PDFPlumberLoader, PyPDFLoader


def load_pdf(file_path: str):
    """Load a PDF into LangChain Documents.

    Tries PDFPlumber first (better layout), then falls back to PyPDF.
    """
    try:
        return PDFPlumberLoader(file_path).load()
    except Exception:
        return PyPDFLoader(file_path).load()