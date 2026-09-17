"""混合检索：向量 + BM25，用 RRF 融合（零依赖）。

## 为什么必须混合

两路检索的「失败模式」是互补的：

| 问题类型 | 纯向量 | 纯 BM25 |
| --- | --- | --- |
| 「请假要办什么手续」（口语、意图模糊） | 好 | 差 |
| 「第七十七条写了什么」（精确条款号） | 差 | 好 |
| 「学号丢了怎么办」（含精确标识符） | 一般 | 好 |
| 「挂科了会怎样」（同义表达，原文写「不及格」） | 好 | 差 |

只用一路，就等于把另一路能答的问题直接丢掉。学生手册里条款号、
部门名、表格编号遍地都是，所以这一路 BM25 的收益非常直接。

## 融合为什么用 RRF 而不是加权求和

余弦相似度在 [0,1]，BM25 分数是**无上界**的（取决于语料和词频）。
想把两者加权求和，得先做归一化——而 BM25 分数的分布随查询变化很大，
归一化本身就是个坑（max-min 归一化会被单个异常高分带偏）。

RRF（Reciprocal Rank Fusion）绕开了这件事：它**只看排名，不看分数**。

    score(d) = Σ_r  1 / (k + rank_r(d))

k 取 60 是行业惯例（原论文推荐值），作用是压平头部差异，
避免某一路的第 1 名独断专行：排名 1 和 2 的分差（1/61 vs 1/62）
远小于排名 1 和 20 的分差。同一段被两路同时召回时会自然叠加得分，
这正好是「两个独立信号都认为它相关」的合理表达。
"""
from app.config import settings
from app.services import embedding as embedding_service, vector_store
from app.services.bm25 import BM25Index
from app.services.utils import run_sync

_bm25: BM25Index | None = None


# ---------------- BM25 索引生命周期 ----------------

def rebuild_bm25() -> BM25Index:
    """从 ChromaDB 全量重建 BM25 索引并落盘。

    为什么整体重建而不是增量维护：手册只有两百多个块，重建不到一秒；
    增量更新要在删除/覆盖文档时逆向调整 df、avgdl、倒排表，
    出错概率远高于重建的收益。派生缓存就该用最笨但最可靠的方式维护。
    """
    global _bm25
    chunks = vector_store.iter_all_chunks()
    index = BM25Index(k1=settings.bm25_k1, b=settings.bm25_b)
    index.build(chunks)
    try:
        index.save(settings.bm25_path)
    except OSError as exc:  # 落盘失败不该影响检索，内存里这份仍然可用
        print(f"[hybrid] BM25 索引落盘失败（不影响本次检索）：{exc}")
    _bm25 = index
    return index


def get_bm25() -> BM25Index:
    """拿到与 ChromaDB 同步的 BM25 索引（懒加载 + 条数校验）。"""
    global _bm25

    expected = vector_store.count()

    if _bm25 is not None and len(_bm25) == expected:
        return _bm25

    cached = BM25Index.load(settings.bm25_path)
    if cached is not None and len(cached) == expected:
        _bm25 = cached
        return _bm25

    return rebuild_bm25()


def invalidate_bm25() -> None:
    """入库/删文档后调用，让下次检索时重建。"""
    global _bm25
    _bm25 = None


# ---------------- RRF 融合 ----------------

def rrf_fuse(rankings: list[list[str]], k: int | None = None) -> dict[str, float]:
    """倒数排名融合。输入多组「按相关性降序的 id 列表」，输出 id → 融合分。"""
    k = k or settings.rrf_k
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + position)
    return scores


# ---------------- 对外检索 ----------------

async def retrieve_multi(
    queries: list[str],
    top_n: int | None = None,
    fused_limit: int | None = None,
) -> dict:
    """多查询混合检索。

    参数 queries 通常只有一条（改写后的查询）；复杂问题会有多条子查询，
    这时每条都跑一遍双路检索，再把所有排名一起丢进 RRF——这是
    「查询拆分 + 融合召回」的标准做法。

    返回 {hits, stats}，hits 已按融合分降序，含两路各自的原始分数与排名。
    """
    top_n = top_n or settings.rerank_top_n
    fused_limit = fused_limit or settings.fused_candidates
    queries = [q.strip() for q in queries if q and q.strip()]
    if not queries:
        return {"hits": [], "stats": {"queries": [], "vector_hits": 0, "bm25_hits": 0}}

    bm25_index = await run_sync(get_bm25) if settings.hybrid_enabled else None

    pool: dict[str, dict] = {}       # id → 命中的块（合并元数据）
    rankings: list[list[str]] = []   # 每一路、每一条查询的排名列表
    stats = {"queries": queries, "vector_hits": 0, "bm25_hits": 0, "ranks": []}

    for query in queries:
        # ---- 第一路：向量检索 ----
        embedding = await run_sync(embedding_service.encode_query, query)
        vector_hits = await run_sync(vector_store.query, embedding, settings.vector_candidates)
        vector_ranking: list[str] = []
        for rank, hit in enumerate(vector_hits, start=1):
            chunk_id = hit["id"]
            merged = pool.setdefault(chunk_id, {**hit, "vector_rank": None, "bm25_rank": None,
                                                "vector_similarity": None, "bm25_score": None,
                                                "matched_queries": []})
            merged["vector_rank"] = rank if merged["vector_rank"] is None else min(merged["vector_rank"], rank)
            merged["vector_similarity"] = hit.get("similarity")
            merged["matched_queries"].append({"query": query, "channel": "vector", "rank": rank})
            vector_ranking.append(chunk_id)
        rankings.append(vector_ranking)
        stats["vector_hits"] += len(vector_hits)

        # ---- 第二路：BM25 ----
        if bm25_index is not None:
            bm25_hits = await run_sync(bm25_index.search, query, settings.bm25_candidates)
            bm25_ranking: list[str] = []
            for rank, hit in enumerate(bm25_hits, start=1):
                chunk_id = hit["id"]
                merged = pool.setdefault(chunk_id, {**hit, "vector_rank": None, "bm25_rank": None,
                                                    "vector_similarity": None, "bm25_score": None,
                                                    "matched_queries": []})
                merged["bm25_rank"] = rank if merged["bm25_rank"] is None else min(merged["bm25_rank"], rank)
                merged["bm25_score"] = hit.get("bm25_score")
                merged["matched_queries"].append({"query": query, "channel": "bm25", "rank": rank})
                bm25_ranking.append(chunk_id)
            rankings.append(bm25_ranking)
            stats["bm25_hits"] += len(bm25_hits)

    # ---- RRF 融合 ----
    fused = rrf_fuse([ranking for ranking in rankings if ranking])
    for chunk_id, score in fused.items():
        pool[chunk_id]["rrf_score"] = round(score, 6)

    ordered = sorted(pool.values(), key=lambda item: item.get("rrf_score", 0.0), reverse=True)
    candidates = ordered[:fused_limit]

    stats["pool_size"] = len(pool)
    stats["fused"] = len(candidates)
    stats["vector_only"] = sum(1 for c in candidates if c.get("bm25_rank") is None)
    stats["bm25_only"] = sum(1 for c in candidates if c.get("vector_rank") is None)
    stats["both"] = sum(1 for c in candidates
                        if c.get("bm25_rank") is not None and c.get("vector_rank") is not None)

    # 这里返回的是「融合后的候选池」（默认 20 条），不是最终结果——
    # 下一步重排要从这 20 条里挑出 top_n 条，所以候选必须给足。
    return {"hits": candidates, "stats": stats}
