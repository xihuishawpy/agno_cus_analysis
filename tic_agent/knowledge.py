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
from agno.vectordb.chroma import ChromaDb

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

    # Create ChromaDB vector database
    vector_db = _create_vector_db(settings, embedder)
    print(f"✓ Using agno ChromaDB vector database: {type(vector_db).__name__}")
    print(f"  Collection: {settings.chroma_collection}")
    print(f"  Path: {settings.chroma_path}")
    print(f"  Persistent: {settings.chroma_persistent}")

    knowledge = Knowledge(
        name="TIC Excel Knowledge",
        description="构建自 Excel 的概念与行业 constituents",
        vector_db=vector_db,
    )

    docs = list(_read_documents(settings.excel_paths))
    if not docs:
        print("⚠️ No documents found")
        return knowledge

    payload = [
        {
            "name": doc.name,
            "text_content": doc.text,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]

    print(f"📊 Processing {len(payload)} documents...")
    skip_if_exists = not settings.rebuild_on_start
    # Use lower concurrency for Milvus stability
    concurrency = 1  # Sequential processing for stability

    try:
        asyncio.run(_bulk_add_contents(knowledge, payload, concurrency, skip_if_exists))
        print(f"✅ Successfully loaded {len(payload)} documents into ChromaDB")
    except Exception as e:
        print(f"❌ Error loading documents into ChromaDB: {e}")
        raise

    return knowledge


async def _bulk_add_contents(
    knowledge: Knowledge,
    payload: List[Dict[str, Any]],
    concurrency: int,
    skip_if_exists: bool,
) -> None:
    """Add contents to knowledge base sequentially for Milvus stability."""
    for i, item in enumerate(payload):
        try:
            print(f"  Loading {i+1}/{len(payload)}: {item['name']}")
            await knowledge.add_content_async(
                name=item["name"],
                text_content=item["text_content"],
                metadata=item.get("metadata"),
                skip_if_exists=False,  # Disable skip_if_exists to avoid query issues
            )
        except Exception as e:
            print(f"  ⚠️ Error loading {item['name']}: {e}")
            continue


def _create_vector_db(settings: KnowledgeSettings, embedder: Embedder):
    """Create ChromaDB vector database."""
    return ChromaDb(
        collection=settings.chroma_collection,
        embedder=embedder,
        path=settings.chroma_path,
        persistent_client=settings.chroma_persistent,
    )


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
    """Drop existing ChromaDB collection if rebuild is requested."""
    if not settings.rebuild_on_start:
        return

    try:
        from chromadb import PersistentClient

        print(f"🗑️  Dropping existing ChromaDB collection: {settings.chroma_collection}")
        client = PersistentClient(path=settings.chroma_path)
        try:
            client.delete_collection(name=settings.chroma_collection)
            print(f"✅ Dropped collection: {settings.chroma_collection}")
        except Exception:
            print(f"⚠️ Collection may not exist: {settings.chroma_collection}")
    except Exception as e:
        print(f"⚠️ Could not drop collection: {e}")


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
