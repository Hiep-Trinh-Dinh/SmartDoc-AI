from __future__ import annotations

import os

try:
    # New, non-deprecated integration package
    from langchain_ollama import OllamaLLM
except Exception:  # pragma: no cover
    OllamaLLM = None
    from langchain_community.llms import Ollama


def get_llm(
    model: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
):
    model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    base_url = base_url or os.getenv("OLLAMA_BASE_URL")
    temperature = 0.0 if temperature is None else temperature

    kwargs = {"model": model, "temperature": float(temperature)}
    if base_url:
        kwargs["base_url"] = base_url

    if OllamaLLM is not None:
        return OllamaLLM(**kwargs)

    # Fallback for environments without langchain-ollama
    return Ollama(**kwargs)