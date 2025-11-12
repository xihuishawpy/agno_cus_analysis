from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import os

import dotenv


DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass
class ModelSettings:
    model_id: str = field(default_factory=lambda: os.getenv("DASHSCOPE_MODEL", "qwen-plus"))
    temperature: float = field(default_factory=lambda: _env_float("DASHSCOPE_TEMPERATURE", 0.3))
    enable_thinking: bool = field(default_factory=lambda: _env_bool("DASHSCOPE_ENABLE_THINKING", False))
    thinking_budget: Optional[int] = field(
        default_factory=lambda: os.getenv("DASHSCOPE_THINKING_BUDGET")
    )


@dataclass
class KnowledgeSettings:
    excel_paths: List[Path]
    table_name: str = field(default_factory=lambda: os.getenv("KNOWLEDGE_TABLE", "tic_excel_documents"))
    vector_dir: Path = field(
        default_factory=lambda: Path(os.getenv("KNOWLEDGE_CACHE_DIR", ".kb_cache/lancedb"))
    )
    embedder_id: str = field(
        default_factory=lambda: os.getenv("KNOWLEDGE_EMBEDDER", "text-embedding-v3")
    )
    embedder_backend: str = field(
        default_factory=lambda: os.getenv("KNOWLEDGE_EMBEDDER_BACKEND", "dashscope")
    )
    embedder_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("KNOWLEDGE_EMBEDDER_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("QWEN_API_KEY")
    )
    embedder_base_url: Optional[str] = field(
        default_factory=lambda: os.getenv("KNOWLEDGE_EMBEDDER_BASE_URL")
        or os.getenv("DASHSCOPE_BASE_URL")
        or DEFAULT_DASHSCOPE_BASE_URL
    )
    rebuild_on_start: bool = field(default_factory=lambda: _env_bool("KNOWLEDGE_REBUILD", False))
    ingest_concurrency: int = field(default_factory=lambda: _env_int("KNOWLEDGE_INGEST_CONCURRENCY", 4))


@dataclass
class WorkflowSettings:
    initial_queries: int = field(default_factory=lambda: _env_int("INITIAL_QUERY_COUNT", 3))
    max_research_loops: int = field(default_factory=lambda: _env_int("MAX_RESEARCH_LOOPS", 2))
    min_research_chars: int = field(default_factory=lambda: _env_int("MIN_RESEARCH_CHAR_COUNT", 600))
    industry_hit_threshold: int = field(default_factory=lambda: _env_int("INDUSTRY_KB_HITS", 2))
    reflection_depth: int = field(default_factory=lambda: _env_int("REFLECTION_DEPTH", 1))


@dataclass
class AppConfig:
    model: ModelSettings
    knowledge: KnowledgeSettings
    workflow: WorkflowSettings
    dashscope_base_url: str


def load_config(env_path: Optional[Path] = None) -> AppConfig:
    dotenv.load_dotenv(env_path)

    excel_paths_env = os.getenv(
        "KNOWLEDGE_BASE_PATHS",
        "data/eastmoney_concept_constituents.xlsx,data/sw_third_industry_constituents.xlsx",
    )
    excel_paths: List[Path] = [Path(p.strip()) for p in excel_paths_env.split(",") if p.strip()]

    model_settings = ModelSettings()
    knowledge_settings = KnowledgeSettings(excel_paths=excel_paths)
    workflow_settings = WorkflowSettings()

    thinking_budget = model_settings.thinking_budget
    if isinstance(thinking_budget, str) and thinking_budget.strip():
        try:
            model_settings.thinking_budget = int(thinking_budget)
        except ValueError:
            model_settings.thinking_budget = None
    else:
        model_settings.thinking_budget = None

    base_url = os.getenv("DASHSCOPE_BASE_URL", DEFAULT_DASHSCOPE_BASE_URL)

    knowledge_settings.vector_dir.mkdir(parents=True, exist_ok=True)

    return AppConfig(
        model=model_settings,
        knowledge=knowledge_settings,
        workflow=workflow_settings,
        dashscope_base_url=base_url,
    )
