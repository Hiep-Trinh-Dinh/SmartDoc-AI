# SmartDoc AI - Intelligent Document Q&A System

Hệ thống hỏi-đáp tài liệu PDF chạy **offline/local** theo mô hình **RAG** (Retrieval-Augmented Generation).

## Tính năng
- Upload PDF (kéo & thả)
- Tự đọc PDF → chia nhỏ (chunking) → embedding → lưu FAISS
- Đặt câu hỏi tự nhiên (đa ngôn ngữ) và trả lời dựa trên nội dung tài liệu
- Có loading khi xử lý và thông báo lỗi rõ ràng

## Yêu cầu
- Windows / macOS / Linux
- Python 3.10+ (workspace này đang dùng `py` launcher)
- Ollama cài sẵn (để chạy LLM local)

## Cài dependencies
```bash
py -m pip install -r requirements.txt
```

## Cài và chạy Ollama + model
1. Cài Ollama: https://ollama.com
2. Pull model:
```bash
ollama pull qwen2.5:7b
```

## Chạy ứng dụng
```bash
py -m streamlit run app.py
```

Mở trình duyệt tại: http://localhost:8501

## Ghi chú lưu trữ
- File PDF upload được lưu trong `data/uploads/`
- FAISS index được cache theo hash trong `data/faiss/` để lần sau mở lại nhanh hơn

## Cấu hình (tuỳ chọn)
- `OLLAMA_MODEL` (mặc định: `qwen2.5:7b`)
- `OLLAMA_BASE_URL` (nếu Ollama chạy remote)
- `EMBEDDING_MODEL` (mặc định: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`)
- `EMBEDDING_DEVICE` (ví dụ: `cpu` hoặc `cuda`)

## Nếu terminal báo lỗi thiếu `torchvision`
Một số bản `transformers` có module xử lý ảnh phụ thuộc `torchvision`. Streamlit file-watcher đôi khi scan và kích hoạt import các module này gây spam log.

Dự án đã cấu hình tắt file watcher tại `/.streamlit/config.toml` (fileWatcherType = "none").
Nếu bạn vẫn thấy log, hãy dừng và chạy lại Streamlit.
