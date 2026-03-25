from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_api_key: str = Field(..., alias="LLM_API_KEY")
    llm_base_url: str = Field(
        default="https://api.deepseek.com",
        alias="LLM_BASE_URL",
    )
    llm_model: str = Field(default="deepseek-chat", alias="LLM_MODEL")

    llm_fast_api_key: str = Field(default="", alias="LLM_FAST_API_KEY")
    llm_fast_base_url: str = Field(default="", alias="LLM_FAST_BASE_URL")
    llm_fast_model: str = Field(default="", alias="LLM_FAST_MODEL")

    token_budget: int = Field(default=100_000, alias="TOKEN_BUDGET")
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_path: str = Field(default="./data/qdrant_db", alias="QDRANT_PATH")
    knowledge_update_interval: int = Field(default=10800, alias="KNOWLEDGE_UPDATE_INTERVAL")

    # Embedding / Reranker model names (loaded locally via sentence-transformers)
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Context compaction triggers at this fraction of the model context window
    compaction_threshold: float = 0.70
    # Cosine similarity above which a new chunk is considered duplicate
    dedup_similarity_threshold: float = 0.92
    # Topic coverage ratio that triggers "suggest end" for resume-driven topics
    coverage_suggest_end: float = 0.85

    @property
    def qdrant_abs_path(self) -> Path:
        p = Path(self.qdrant_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


# ────────────────── Dynamic Topic Taxonomy ──────────────────

_DEFAULT_TOPICS = [
    "Transformer架构", "大模型训练", "大模型推理与部署",
    "RAG检索增强生成", "提示工程", "Agent智能体",
    "向量数据库", "微调技术", "模型评估",
    "推荐系统", "NLP基础", "多模态模型",
    "分布式训练", "数据工程与处理", "机器学习基础",
]

_TAXONOMY_PATH = PROJECT_ROOT / "data" / "topic_taxonomy.json"

_topic_cache: list[str] | None = None


def load_topic_taxonomy(*, force_reload: bool = False) -> list[str]:
    """Load the topic taxonomy from the JSON file.

    Creates the file with defaults if it doesn't exist.
    Results are cached in memory; pass force_reload=True to re-read from disk.
    """
    global _topic_cache
    if _topic_cache is not None and not force_reload:
        return list(_topic_cache)

    if not _TAXONOMY_PATH.exists():
        _TAXONOMY_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_topic_taxonomy(_DEFAULT_TOPICS)
        _topic_cache = list(_DEFAULT_TOPICS)
        return list(_topic_cache)

    try:
        data = json.loads(_TAXONOMY_PATH.read_text(encoding="utf-8"))
        topics = data.get("topics", _DEFAULT_TOPICS)
        _topic_cache = list(topics)
        return list(_topic_cache)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Failed to load topic taxonomy from %s: %s", _TAXONOMY_PATH, e)
        _topic_cache = list(_DEFAULT_TOPICS)
        return list(_topic_cache)


def save_topic_taxonomy(topics: list[str]) -> None:
    """Persist the topic taxonomy to the JSON file and refresh the cache."""
    global _topic_cache
    unique = list(dict.fromkeys(topics))
    data = {
        "version": 1,
        "topics": unique,
    }
    _TAXONOMY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TAXONOMY_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _topic_cache = list(unique)
    logger.info("Topic taxonomy saved (%d topics) → %s", len(unique), _TAXONOMY_PATH)
