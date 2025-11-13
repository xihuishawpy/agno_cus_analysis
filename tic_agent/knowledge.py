from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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

    file_signatures = _collect_file_signatures(settings.excel_paths)
    manifest = _load_manifest(settings)
    rebuild_reason = _determine_rebuild_reason(settings, manifest, file_signatures)

    if not rebuild_reason:
        print("↩️ Excel 源文件无变化，复用已有向量数据库。")
        return knowledge

    print(f"🔄 触发向量库重建：{rebuild_reason}")
    _maybe_drop_existing_table(settings)

    docs = list(_read_documents(settings.excel_paths))
    if not docs:
        print("⚠️ No documents found")
        _save_manifest(settings, file_signatures)
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
    concurrency = max(1, settings.ingest_concurrency)
    print(f"  Concurrency: {concurrency}")

    try:
        asyncio.run(_bulk_add_contents(knowledge, payload, concurrency, skip_if_exists=False))
        print(f"✅ Successfully loaded {len(payload)} documents into ChromaDB")
        _save_manifest(settings, file_signatures)
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
    """Add contents to knowledge base with configurable concurrency."""

    semaphore = asyncio.Semaphore(concurrency)
    total = len(payload)

    async def _process(idx: int, item: Dict[str, Any]):
        async with semaphore:
            try:
                print(f"  Loading {idx + 1}/{total}: {item['name']}")
                await knowledge.add_content_async(
                    name=item["name"],
                    text_content=item["text_content"],
                    metadata=item.get("metadata"),
                    skip_if_exists=skip_if_exists,
                )
            except Exception as exc:
                print(f"  ⚠️ Error loading {item['name']}: {exc}")

    await asyncio.gather(*(_process(i, item) for i, item in enumerate(payload)))


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


def _collect_file_signatures(paths: Iterable[Path]) -> Dict[str, Dict[str, int]]:
    signatures: Dict[str, Dict[str, int]] = {}
    for path in paths:
        if not path.exists():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        signatures[str(path.resolve())] = {
            "mtime_ns": int(stat.st_mtime_ns),
            "size": int(stat.st_size),
        }
    return signatures


def _determine_rebuild_reason(
    settings: KnowledgeSettings,
    manifest: Optional[Dict[str, Any]],
    current_signatures: Dict[str, Dict[str, int]],
) -> Optional[str]:
    if settings.rebuild_on_start:
        return "配置开启强制重建 (KNOWLEDGE_REBUILD=1)"
    if manifest is None:
        return "首次构建向量库"
    previous_files = manifest.get("files") or {}
    if previous_files != current_signatures:
        return "检测到 Excel 源文件变更"
    return None


def _manifest_path(settings: KnowledgeSettings) -> Path:
    return settings.vector_dir / "excel_knowledge_manifest.json"


def _load_manifest(settings: KnowledgeSettings) -> Optional[Dict[str, Any]]:
    path = _manifest_path(settings)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        print(f"⚠️ 无法读取知识库 manifest: {exc}")
        return None


def _save_manifest(settings: KnowledgeSettings, files: Dict[str, Dict[str, int]]) -> None:
    path = _manifest_path(settings)
    data = {
        "files": files,
        "generated_at": datetime.utcnow().isoformat(),
    }
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"⚠️ 无法写入知识库 manifest: {exc}")
