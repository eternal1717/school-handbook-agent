"""知识库服务：把文件解析 → 切块 → 去重 → 向量化 → 入库，并记录文档元数据。

关于「增量入库」：新文档里与知识库已有内容重复的块会被跳过，只补真正新增的部分。
两级去重：
  1. 内容指纹（精确）——文字一样、只是排版/换行不同，也算重复；
  2. 语义相似度（近似）——换了说法、改了几个字的同一段内容，也算重复。
"""
import asyncio
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import KnowledgeDocument
from app.services import doc_parser, embedding as embedding_service, hybrid, rulebook, vector_store
from app.services.utils import content_hash, run_sync

# 入库串行化：去重是「先查全库指纹、再写入」的两步操作，两个请求并发上传同一份文档时
# 可能都查到"不重复"，结果各写一遍。加锁让入库排队，单机场景完全够用。
_ingest_lock = asyncio.Lock()


async def list_documents(db: AsyncSession) -> list[KnowledgeDocument]:
    result = await db.execute(select(KnowledgeDocument).order_by(KnowledgeDocument.created_at.desc()))
    return list(result.scalars().all())


async def _remove_same_filename(db: AsyncSession, filename: str) -> int:
    """同一个文件重复上传时，先把旧版本清掉，避免出现重复知识块。"""
    result = await db.execute(select(KnowledgeDocument).where(KnowledgeDocument.filename == filename))
    removed = 0
    for old in result.scalars().all():
        await run_sync(vector_store.delete_document, old.id)
        await db.delete(old)
        removed += 1
    if removed:
        await db.commit()
    return removed


async def _split_new_chunks(filename: str, path: Path) -> tuple[list[dict], list[list[float]], int]:
    """解析 → 切块 → 两级去重 → 向量化。

    返回 (新增块, 对应的向量, 被跳过的块数)。
    """
    blocks = await run_sync(doc_parser.parse_file, path)
    if not blocks:
        raise ValueError(f"文件里没有解析出任何文本：{filename}")

    chunks = doc_parser.split_blocks(blocks)
    if not chunks:
        raise ValueError(f"切块结果为空：{filename}")

    for chunk in chunks:
        chunk["content_hash"] = content_hash(chunk["text"])

    skipped = 0

    # ---- 第一级：内容指纹精确去重（本批内部也去重）----
    if settings.dedup_exact:
        # 老数据可能没有指纹，先补上，否则去重会漏
        await run_sync(vector_store.ensure_content_hashes)
        seen = await run_sync(vector_store.iter_existing_hashes)
        kept: list[dict] = []
        for chunk in chunks:
            if chunk["content_hash"] in seen:
                skipped += 1
                continue
            seen.add(chunk["content_hash"])
            kept.append(chunk)
        chunks = kept

    if not chunks:
        return [], [], skipped

    # ---- 向量化（只有活下来的块才需要算向量，省时间）----
    embed_texts = doc_parser.build_embed_texts(chunks)
    embeddings = await run_sync(embedding_service.encode_documents, embed_texts)

    # ---- 第二级：语义近似去重 ----
    if settings.dedup_semantic:
        similarities = await run_sync(vector_store.find_similar_hits, embeddings)
        kept_chunks: list[dict] = []
        kept_embeddings: list[list[float]] = []
        for chunk, embedding, similarity in zip(chunks, embeddings, similarities):
            if similarity >= settings.dedup_similarity:
                skipped += 1
                continue
            kept_chunks.append(chunk)
            kept_embeddings.append(embedding)
        chunks, embeddings = kept_chunks, kept_embeddings

    return chunks, embeddings, skipped


async def ingest_file(db: AsyncSession, path: Path, original_name: str | None = None) -> dict:
    """把一个文件灌进知识库，返回统计信息。重复内容不会重复入库。

    整段逻辑用锁串行化，避免并发上传时去重判断失效（详见文件顶部说明）。
    """
    async with _ingest_lock:
        return await _ingest_file_unlocked(db, path, original_name)


async def _ingest_file_unlocked(db: AsyncSession, path: Path, original_name: str | None = None) -> dict:
    filename = original_name or path.name

    # 同名文件先清掉旧版本，避免「旧版 + 新版」同时留在库里
    await _remove_same_filename(db, filename)

    chunks, embeddings, skipped = await _split_new_chunks(filename, path)

    vector_total = await run_sync(vector_store.count)

    # 整份文件都是已存在的内容 → 不建文档记录，避免列表里出现空文档
    if not chunks:
        return {
            "id": "",
            "filename": filename,
            "file_type": path.suffix.lower().lstrip("."),
            "chunk_count": 0,
            "skipped_chunk_count": skipped,
            "char_count": 0,
            "size_bytes": path.stat().st_size,
            "embedding_dim": 0,
            "vector_total": vector_total,
            "duplicated": True,
            "message": (
                f"《{filename}》的内容知识库里已经全部有了，没有新增知识块"
                + (f"（跳过重复 {skipped} 块）" if skipped else "")
            ),
        }

    doc_id = uuid.uuid4().hex
    added = await run_sync(vector_store.add_chunks, doc_id, filename, chunks, embeddings)
    vector_total = await run_sync(vector_store.count)

    # 向量库变了，BM25 索引和规则库都过期了。这里只置空标记，不立即重建——
    # 下一次用到时会发现条数对不上，自动重建。
    # 好处是连续上传多份文件时只重建一次，而不是每份都重建。
    # 规则库这一行不能省：新传进来的规章必须能被抽成规则和办事流程，
    # 否则「规则诊断」「办事流程」两个页面会一直显示旧文档的内容。
    hybrid.invalidate_bm25()
    rulebook.invalidate()
    document = KnowledgeDocument(
        id=doc_id,
        filename=filename,
        file_type=path.suffix.lower().lstrip("."),
        size_bytes=path.stat().st_size,
        char_count=sum(len(chunk["text"]) for chunk in chunks),
        chunk_count=added,
        skipped_chunk_count=skipped,
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)

    return {
        "id": document.id,
        "filename": document.filename,
        "file_type": document.file_type,
        "chunk_count": document.chunk_count,
        "skipped_chunk_count": document.skipped_chunk_count,
        "char_count": document.char_count,
        "size_bytes": document.size_bytes,
        "embedding_dim": len(embeddings[0]) if embeddings else 0,
        "vector_total": vector_total,
        "duplicated": False,
        "message": (
            f"新增 {added} 个知识块"
            + (f"，跳过重复内容 {skipped} 块" if skipped else "")
            + f"，向量库共 {vector_total} 条"
        ),
    }


async def get_document_chunks(db: AsyncSession, doc_id: str) -> list[dict] | None:
    """查看某文档切成了哪些知识块。文档不存在返回 None。"""
    document = await db.get(KnowledgeDocument, doc_id)
    if document is None:
        return None
    return await run_sync(vector_store.get_document_chunks, doc_id)


async def delete_document(db: AsyncSession, doc_id: str) -> bool:
    """删除文档及其向量。"""
    document = await db.get(KnowledgeDocument, doc_id)
    if document is None:
        return False

    await run_sync(vector_store.delete_document, doc_id)
    await db.delete(document)
    await db.commit()
    # 和上传一样置空派生缓存。虽然条数校验也能自愈，但显式失效更直白，
    # 也省掉下一次检索时那次「发现对不上」的探测。
    hybrid.invalidate_bm25()
    rulebook.invalidate()
    return True

