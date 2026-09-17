"""把同步的耗时函数放进线程池执行，避免阻塞 FastAPI 的事件循环。

Embedding 计算和 ChromaDB 读写都是同步阻塞调用，直接在 async 函数里调用会卡住整个
事件循环（其他请求全部排队），所以统一用这个工具包一层。
"""
import hashlib
import re
from collections.abc import Callable
from functools import partial
from typing import Any, TypeVar

from anyio import to_thread

T = TypeVar("T")

_WHITESPACE = re.compile(r"\s+")


async def run_sync(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    return await to_thread.run_sync(partial(fn, *args, **kwargs))


def normalize_text(text: str) -> str:
    """归一化文本，用于生成内容指纹。

    只去掉空白（含全角空格、换行、缩进），不做别的处理：
    这样「排版变了但内容没变」的段落会被认为相同，符合增量去重的意图。
    """
    return _WHITESPACE.sub("", text or "")


def content_hash(text: str) -> str:
    """内容指纹：归一化文本的 sha256 前 16 位。

    为什么不用「原文直接哈希」：同一段话从 docx 复制到 pdf 常带不同的换行/空格，
    直接哈希会认成两份不同内容，去重就失效了。
    """
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()[:16]
