"""Embedding 服务：加载本地 BGE 中文模型，提供向量化能力。

关键点：
1. 模型从本地目录加载，全程离线，不会联网下载（避免写满 C 盘缓存）。
2. 进程内只加载一次（单例 + 线程锁），否则每次请求都重新加载会慢到不可用。
3. 查询侧加指令前缀、文档侧不加，这是 BGE 官方推荐的用法。
"""
import os
import threading

# 必须在导入 sentence_transformers 之前设置，强制离线
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pathlib import Path  # noqa: E402

from app.config import settings  # noqa: E402

_model = None
_lock = threading.Lock()


def get_model():
    """懒加载模型（第一次调用时加载，之后复用）。"""
    global _model
    if _model is not None:
        return _model

    with _lock:
        if _model is not None:  # 双重检查，避免并发重复加载
            return _model

        model_path = Path(settings.embedding_model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"找不到 Embedding 模型：{model_path}\n"
                "请确认 .env 里的 EMBEDDING_MODEL_PATH 指向正确的本地模型目录。"
            )

        from sentence_transformers import SentenceTransformer

        print(f"[embedding] 正在加载本地模型：{model_path}")
        _model = SentenceTransformer(str(model_path))
        print("[embedding] 模型加载完成")
        return _model


def encode_documents(texts: list[str]) -> list[list[float]]:
    """文档侧向量化：不加指令前缀（BGE 官方用法）。"""
    if not texts:
        return []
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,  # 归一化后，余弦相似度 = 点积，数值更稳定
        show_progress_bar=False,
    )
    return [v.tolist() for v in vectors]


def encode_query(text: str) -> list[float]:
    """查询侧向量化：加指令前缀，能明显提升中文检索效果。"""
    model = get_model()
    vector = model.encode(
        [settings.query_instruction + text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector[0].tolist()
