"""向量库封装（ChromaDB 本地持久化）。

为什么用 ChromaDB 而不是 FAISS：FAISS 本质是算法库，没有持久化、没有 metadata 过滤、
删一条要重建索引；ChromaDB 自带持久化和 where 过滤，更适合知识库场景。
"""
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.services.utils import content_hash

_client = None
_collection = None

# 分批拉取/查询的批大小：一次要太多会把内存顶起来，太小又慢
_BATCH = 512


def get_collection():
    """懒加载 + 复用集合。"""
    global _client, _collection
    if _collection is None:
        settings.ensure_dirs()
        _client = chromadb.PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},  # 用余弦距离，配合归一化向量
        )
    return _collection


def add_chunks(doc_id: str, filename: str, chunks: list[dict], embeddings: list[list[float]]) -> int:
    """把知识块写入向量库。先删旧的同 doc_id 数据，保证重复上传不会产生重复块。"""
    if not chunks:
        return 0

    collection = get_collection()
    delete_document(doc_id)

    ids = [f"{doc_id}_{chunk['index']}" for chunk in chunks]
    metadatas = []
    for chunk in chunks:
        meta = {
            "doc_id": doc_id,
            "filename": filename,
            "section": chunk.get("section") or "",
            "chunk_index": int(chunk["index"]),
            # 内容指纹：下次上传新文档时用它做增量去重
            "content_hash": chunk.get("content_hash") or content_hash(chunk["text"]),
        }
        if chunk.get("page") is not None:
            meta["page"] = int(chunk["page"])  # ChromaDB 不接受 None，所以按需添加
        metadatas.append(meta)

    collection.add(
        ids=ids,
        documents=[chunk["text"] for chunk in chunks],
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(ids)


def iter_existing_hashes() -> set[str]:
    """取出库里所有知识块的内容指纹（分批拉，数据量大时也不会撑爆内存）。"""
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return set()

    hashes: set[str] = set()
    for offset in range(0, total, _BATCH):
        data = collection.get(include=["metadatas"], limit=_BATCH, offset=offset)
        for meta in data.get("metadatas") or []:
            value = (meta or {}).get("content_hash")
            if value:
                hashes.add(value)
    return hashes


def ensure_content_hashes() -> int:
    """给「加去重功能之前」入库的老数据回填 content_hash，返回回填条数。

    幂等：已经有指纹的块会被跳过，所以放在每次入库前调用也不会重复干活。
    """
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return 0

    filled = 0
    for offset in range(0, total, _BATCH):
        data = collection.get(include=["documents", "metadatas"], limit=_BATCH, offset=offset)
        ids = data.get("ids") or []
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []

        upd_ids: list[str] = []
        upd_metas: list[dict] = []
        for chunk_id, text, meta in zip(ids, documents, metadatas):
            merged = dict(meta or {})
            if merged.get("content_hash"):
                continue
            merged["content_hash"] = content_hash(text or "")
            upd_ids.append(chunk_id)
            upd_metas.append(merged)

        if upd_ids:
            collection.update(ids=upd_ids, metadatas=upd_metas)
            filled += len(upd_ids)

    return filled


def find_similar_hits(embeddings: list[list[float]]) -> list[float]:
    """对每个待入库向量，返回它与库中「最相似的一条」的相似度（库里为空时返回 0）。"""
    if not embeddings:
        return []

    collection = get_collection()
    if collection.count() == 0:
        return [0.0] * len(embeddings)

    best: list[float] = [0.0] * len(embeddings)
    for start in range(0, len(embeddings), _BATCH):
        batch = embeddings[start : start + _BATCH]
        result = collection.query(
            query_embeddings=batch,
            n_results=1,
            include=["distances"],
        )
        for i, distances in enumerate(result.get("distances") or []):
            if not distances:
                continue
            best[start + i] = max(0.0, min(1.0, 1.0 - float(distances[0])))
    return best



def query(embedding: list[float], top_k: int) -> list[dict]:
    """检索最相关的知识块，返回带相似度分数的结果。"""
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []

    result = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, total),
        include=["documents", "metadatas", "distances"],
    )

    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    hits: list[dict] = []
    for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
        # cosine 距离 → 相似度，范围裁剪到 [0, 1]
        similarity = max(0.0, min(1.0, 1.0 - float(distance)))
        hits.append(
            {
                "id": chunk_id,
                "doc_id": (meta or {}).get("doc_id", ""),
                "chunk_index": (meta or {}).get("chunk_index"),
                "text": text,
                "filename": (meta or {}).get("filename", ""),
                "section": (meta or {}).get("section", ""),
                "page": (meta or {}).get("page"),
                "similarity": round(similarity, 4),
            }
        )
    return hits


def delete_document(doc_id: str) -> None:
    """删除某文档的全部向量。"""
    collection = get_collection()
    collection.delete(where={"doc_id": doc_id})


def get_document_chunks(doc_id: str) -> list[dict]:
    """取出某文档的全部知识块，按序号排好，用于在页面上查看分块结果。"""
    collection = get_collection()
    data = collection.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])

    ids = data.get("ids") or []
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    chunks: list[dict] = []
    for chunk_id, text, meta in zip(ids, documents, metadatas):
        meta = meta or {}
        text = text or ""
        chunks.append(
            {
                "id": chunk_id,
                "index": int(meta.get("chunk_index", 0)),
                "section": meta.get("section", ""),
                "page": meta.get("page"),
                "chars": len(text),
                "text": text,
                "content_hash": meta.get("content_hash", ""),
            }
        )
    chunks.sort(key=lambda c: c["index"])  # ChromaDB 不保证顺序，按块序号排
    return chunks


def count() -> int:
    return get_collection().count()


def iter_all_chunks() -> list[dict]:
    """取出全库知识块（分批），供 BM25 建索引使用。

    BM25 需要「全部文本 + 元数据」，而 ChromaDB 是内容的事实来源，
    所以索引重建时从这里拉一次全量。
    """
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []

    chunks: list[dict] = []
    for offset in range(0, total, _BATCH):
        data = collection.get(include=["documents", "metadatas"], limit=_BATCH, offset=offset)
        ids = data.get("ids") or []
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        for chunk_id, text, meta in zip(ids, documents, metadatas):
            meta = meta or {}
            chunks.append(
                {
                    "id": chunk_id,
                    "text": text or "",
                    "filename": meta.get("filename", ""),
                    "section": meta.get("section", ""),
                    "page": meta.get("page"),
                    "chunk_index": int(meta.get("chunk_index", 0)),
                    "doc_id": meta.get("doc_id", ""),
                }
            )
    return chunks
