from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_model_name: str = Field("gpt-4o", alias="OPENAI_MODEL_NAME")
    openai_embedding_model: str = Field(
        "text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )

    default_token_budget: int = Field(10000, alias="DEFAULT_TOKEN_BUDGET")
    default_max_steps: int = Field(5, alias="DEFAULT_MAX_STEPS")

    chroma_persist_dir: Path = Field(Path("./chroma_db"), alias="CHROMA_PERSIST_DIR")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
