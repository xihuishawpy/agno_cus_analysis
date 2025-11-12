from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import asyncio
import json
import math
import os

import pandas as pd

from agno.knowledge import Knowledge
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.knowledge.embedder.base import Embedder
from agno.vectordb.lancedb import LanceDb

from .config import AppConfig, KnowledgeSettings


@dataclass
class KnowledgeDocument:
    name: str
    text: str
    metadata: Dict[str, str]


def build_excel_knowledge(config: AppConfig) -> Knowledge:
    settings = config.knowledge
    _maybe_drop_existing_table(settings)

    embedder = _create_embedder(config)
    vector_db = LanceDb(
        table_name=settings.table_name,
        uri=str(settings.vector_dir),
        embedder=embedder,
    )

    knowledge = Knowledge(
        name="TIC Excel Knowledge",
        description="构建自 Excel 的概念与行业 constituents",
        vector_db=vector_db,
    )

    docs = list(_read_documents(settings.excel_paths))
    if not docs:
        return knowledge

    payload = [
        {
            "name": doc.name,
            "text_content": doc.text,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]

    skip_if_exists = not settings.rebuild_on_start
    concurrency = max(1, settings.ingest_concurrency)
    asyncio.run(_bulk_add_contents(knowledge, payload, concurrency, skip_if_exists))
    return knowledge


async def _bulk_add_contents(
    knowledge: Knowledge,
    payload: List[Dict[str, Any]],
    concurrency: int,
    skip_if_exists: bool,
) -> None:
    semaphore = asyncio.Semaphore(concurrency)

    async def _worker(item: Dict[str, str]) -> None:
        async with semaphore:
            await knowledge.add_content_async(
                name=item["name"],
                text_content=item["text_content"],
                metadata=item.get("metadata"),
                skip_if_exists=skip_if_exists,
            )

    await asyncio.gather(*(_worker(item) for item in payload))


def _create_embedder(config: AppConfig) -> Embedder:
    settings = config.knowledge
    backend = (settings.embedder_backend or "dashscope").lower()
    if backend == "dashscope":
        api_key = settings.embedder_api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY 未配置，无法使用 Qwen 向量 API")
        base_url = settings.embedder_base_url or config.dashscope_base_url
        return OpenAIEmbedder(
            id=settings.embedder_id,
            api_key=api_key,
            base_url=base_url,
        )

    # fallback to local sentence-transformers
    return SentenceTransformerEmbedder(id=settings.embedder_id)


def _maybe_drop_existing_table(settings: KnowledgeSettings) -> None:
    if not settings.rebuild_on_start:
        return

    settings.vector_dir.mkdir(parents=True, exist_ok=True)

    try:
        import lancedb  # type: ignore

        connection = lancedb.connect(uri=str(settings.vector_dir))
        if settings.table_name in connection.table_names():
            connection.drop_table(settings.table_name)
    except Exception:
        pass


def _read_documents(paths: Iterable[Path]) -> Iterable[KnowledgeDocument]:
    for path in paths:
        if not path.exists():
            continue

        df = pd.read_excel(path)
        records = df.to_dict(orient="records")

        if "概念名称" in df.columns:
            yield from _build_concept_docs(path.name, records)
        else:
            yield from _build_industry_docs(path.name, records)


def _build_concept_docs(source: str, rows: List[Dict]) -> Iterable[KnowledgeDocument]:
    for row in rows:
        concept = _clean(row.get("概念名称"))
        ticker = _clean(row.get("股票代码"))
        company = _clean(row.get("股票简称"))
        if not (concept and ticker):
            continue

        text = _row_to_text("概念成分", source, row)
        metadata = {
            "source": "eastmoney",
            "concept_name": concept,
            "ticker": ticker,
            "company": company or "",
        }
        name = f"concept::{concept}::{ticker}"
        yield KnowledgeDocument(name=name, text=text, metadata=metadata)


def _build_industry_docs(source: str, rows: List[Dict]) -> Iterable[KnowledgeDocument]:
    for row in rows:
        industry_l2 = _clean(row.get("所属二级行业"))
        industry_l3 = _clean(row.get("所属三级行业名称"))
        ticker = _clean(row.get("股票代码"))
        company = _clean(row.get("股票简称"))
        if not (industry_l3 and ticker):
            continue

        text = _row_to_text("申万行业", source, row)
        metadata = {
            "source": "shenwan",
            "industry_level2": industry_l2 or "",
            "industry_level3": industry_l3,
            "ticker": ticker,
            "company": company or "",
        }
        name = f"industry::{industry_l3}::{ticker}"
        yield KnowledgeDocument(name=name, text=text, metadata=metadata)


def _row_to_text(label: str, source: str, row: Dict) -> str:
    items = []
    for key, value in row.items():
        cleaned = _clean(value)
        if cleaned is None or cleaned == "":
            continue
        items.append(f"{key}:{cleaned}")

    joined = " | ".join(items)
    metadata_block = json.dumps(
        {
            "source_file": source,
            "columns": list(row.keys()),
        },
        ensure_ascii=False,
    )
    return f"[{label}] {joined}\nMETA: {metadata_block}"


def _clean(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text if text else None
