from document_loader import load_docx
import sys

file_path = r"data\uploads\50de7b5310d44ce2431e3326855a03678007d6641ce3fd391d53d7350a5780fa_xinchao.docx"
try:
    docs = load_docx(file_path)
    print(f"Loaded {len(docs)} documents.")
    for i, doc in enumerate(docs):
        print(f"--- Document {i+1} ---")
        print(f"Metadata: {doc.metadata}")
        print(f"Content (first 200 chars): {doc.page_content[:200]}")
except Exception as e:
    print(f"Error: {e}")
