from __future__ import annotations

from typing import List

from langchain_openai import OpenAIEmbeddings

from app.config import get_settings


def get_embedding_model() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
    )


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    model = get_embedding_model()
    return model.embed_documents(texts)


def embed_query(text: str) -> List[float]:
    if not text:
        return []
    model = get_embedding_model()
    return model.embed_query(text)
