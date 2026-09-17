"""手写 BM25 检索（零依赖，带持久化）。

## 它补的是哪个短板

原来只有向量检索。向量检索擅长「意思相近」，但对**精确标识符**天然偏弱：
「第七十七条」「学号」「教务处」这类字符串，在 embedding 里会被语义稀释，
学生明明打对了条款号，反而召不回来。

BM25 走的是词频（TF）× 逆文档频率（IDF）这条统计路线，
对「这个词在这段出现过几次、在整个库里有多罕见」非常敏感，
正好和向量检索形成互补。两者用 RRF 融合，就是业界标准的混合检索。

## 为什么手写

1. 项目约定不新增依赖；
2. BM25 本身只有几十行数学，比引入 rank_bm25 更容易按中文场景调参
   （尤其是配合 tokenizer 里的二元组分词）；
3. 索引需要跟 ChromaDB 的内容保持同步，自己控制生命周期更简单。

## 索引怎么维护

ChromaDB 是内容的事实来源，BM25 索引是它的派生缓存。
每次入库完成后**整体重建**——手册只有两百多个块，重建耗时不到一秒，
远比维护增量更新（删块、改块时要逆向调整 df/avgdl）可靠。
重建后写入 JSON 落盘，启动时若缓存条数与 Chroma 一致就直接复用。
"""
import json
import math
from collections import Counter
from pathlib import Path

from app.services.tokenizer import tokenize, tokenize_query

INDEX_VERSION = 1


class BM25Index:
    """标准的 Okapi BM25，默认参数取业界常用值。

    k1 控制词频饱和度（一个词出现 10 次不该比 1 次重要 10 倍），
    b 控制文档长度归一化（长文档天然容易命中更多词，需要惩罚）。
    中文二元组场景下 k1=1.5 / b=0.75 是稳妥的默认值，
    若要针对具体语料调参，改 .env 的 BM25_K1 / BM25_B 即可。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: list[dict] = []          # 块元数据：id / text / filename / section / page
        self.tokens: list[list[str]] = []   # 每块的 token 序列
        self.term_freqs: list[Counter] = []  # 每块的词频表
        self.doc_freq: dict[str, int] = {}   # 词 → 出现在多少个块里
        self.idf: dict[str, float] = {}      # 词 → 逆文档频率（预计算）
        self.inverted: dict[str, list[int]] = {}  # 词 → 块下标列表（倒排表，用于快速取候选）
        self.doc_len: list[int] = []
        self.avgdl: float = 0.0

    # ---------------- 构建 ----------------

    def build(self, docs: list[dict]) -> None:
        """用全部知识块重建索引。docs 每项需含 id / text / filename / section / page。"""
        self.docs = list(docs)
        self.tokens = []
        self.term_freqs = []
        self.doc_freq = {}
        self.inverted = {}
        self.doc_len = []

        for doc in self.docs:
            tokens = tokenize(doc.get("text") or "")
            self.tokens.append(tokens)
            freq = Counter(tokens)
            self.term_freqs.append(freq)
            self.doc_len.append(len(tokens))

        total = len(self.docs)
        self.avgdl = (sum(self.doc_len) / total) if total else 0.0

        # 统计 df 与倒排表
        for index, freq in enumerate(self.term_freqs):
            for term in freq:
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1
                self.inverted.setdefault(term, []).append(index)

        # 预计算 IDF。用平滑公式，避免词出现在所有文档里时 IDF 变负数。
        self.idf = {
            term: math.log(1 + (total - df + 0.5) / (df + 0.5))
            for term, df in self.doc_freq.items()
        }

    def __len__(self) -> int:
        return len(self.docs)

    # ---------------- 检索 ----------------

    def search(self, query: str, top_k: int = 30) -> list[dict]:
        """检索最相关的块，返回 [{...doc, score}]，按分数降序。"""
        if not self.docs or top_k <= 0:
            return []

        query_tokens = tokenize_query(query)
        if not query_tokens:
            return []

        # 只在「至少命中一个查询词」的块上打分，避免遍历全库
        candidates: set[int] = set()
        for token in query_tokens:
            candidates.update(self.inverted.get(token, ()))
        if not candidates:
            return []

        scored: list[tuple[int, float]] = []
        for index in candidates:
            score = self._score_one(index, query_tokens)
            if score > 0:
                scored.append((index, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        results: list[dict] = []
        for index, score in scored[:top_k]:
            item = dict(self.docs[index])
            item["bm25_score"] = round(score, 4)
            item["similarity"] = None  # BM25 分数不是余弦相似度，这里明确置空避免误用
            results.append(item)
        return results

    def _score_one(self, index: int, query_tokens: list[str]) -> float:
        """单块的 BM25 得分。"""
        freq = self.term_freqs[index]
        length = self.doc_len[index] or 1
        score = 0.0
        for token in query_tokens:
            term_count = freq.get(token)
            if not term_count:
                continue
            idf = self.idf.get(token, 0.0)
            denominator = term_count + self.k1 * (
                1 - self.b + self.b * length / (self.avgdl or 1.0)
            )
            score += idf * term_count * (self.k1 + 1) / denominator
        return score

    # ---------------- 持久化 ----------------

    def to_dict(self) -> dict:
        return {
            "version": INDEX_VERSION,
            "k1": self.k1,
            "b": self.b,
            "avgdl": self.avgdl,
            "doc_len": self.doc_len,
            "doc_freq": self.doc_freq,
            "inverted": self.inverted,
            "idf": self.idf,
            "docs": self.docs,
            "tokens": self.tokens,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "BM25Index | None":
        """读缓存。文件不存在、格式不符、版本不匹配时返回 None（调用方会重建）。"""
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if payload.get("version") != INDEX_VERSION:
            return None

        index = cls(k1=payload.get("k1", 1.5), b=payload.get("b", 0.75))
        index.docs = payload.get("docs") or []
        index.tokens = payload.get("tokens") or []
        index.doc_len = payload.get("doc_len") or []
        index.doc_freq = payload.get("doc_freq") or {}
        index.idf = payload.get("idf") or {}
        index.inverted = {
            term: list(posting) for term, posting in (payload.get("inverted") or {}).items()
        }
        index.avgdl = payload.get("avgdl", 0.0)
        index.term_freqs = [Counter(tokens) for tokens in index.tokens]

        # 自检：条数对不上说明缓存坏了，让调用方重建
        if len(index.docs) != len(index.term_freqs):
            return None
        return index
