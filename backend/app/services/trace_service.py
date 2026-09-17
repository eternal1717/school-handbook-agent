"""链路追踪服务：收集一次问答的全过程，最后统一落库。

## 用法

在 answer_stream 的开头建一个 TraceRecorder，中间随手记，
结束时 save。它只在内存里累积，落库是一次写操作——
避免每个阶段都往数据库写一次，把一次问答变成十几次 IO。

## 为什么要收集阶段耗时

真正调优时最常问的三个问题是：
1. 慢在哪？——看 stages 里的 ms 分布。改写 3 秒还是生成 8 秒，处理方式完全不同。
2. 检索到没到？——看 retrieval 里的 vector_hits / bm25_hits / fused。
3. 是改写把问题带偏了吗？——对比 question 和 rewritten。
这三个问题，trace 里都能直接读出答案。
"""
import json
import time
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Trace


class TraceRecorder:
    """一次问答的追踪记录器。所有 add_* 方法都很轻，随手调用即可。"""

    def __init__(self, question: str, user_id: str, conversation_id: str) -> None:
        # trace_id 提前生成：消息表要存它，而消息先于 trace 落库。
        # 先定 id 再写两侧，就不用为了拿 id 而多写一次库。
        self.trace_id = uuid.uuid4().hex
        self.question = question
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.started = time.perf_counter()
        self.stages: list[dict] = []
        self.plan: dict = {}
        self.retrieval: dict = {}
        self.rerank: dict = {}
        self.tool_calls: list[dict] = []
        self.guard_risks: list[dict] = []
        self.usage: dict = {}
        self.rewritten = ""
        self.hops = 0
        self.source_count = 0
        self.answer_chars = 0
        self.refused = False

    # ---------------- 记录 ----------------

    def stage(self, name: str, ms: int, **detail) -> None:
        """记一个阶段的耗时与细节。ms 传实际测得毫秒数。"""
        self.stages.append({"stage": name, "ms": int(ms), **detail})

    def measure(self, name: str, started: float, **detail) -> int:
        """给定起始时刻，自动算出毫秒、记下来、并返回耗时。"""
        ms = int((time.perf_counter() - started) * 1000)
        self.stage(name, ms, **detail)
        return ms

    def set_plan(self, plan: dict) -> None:
        self.plan = {
            "intent": plan.get("intent"),
            "complexity": plan.get("complexity"),
            "needs_retrieval": plan.get("needs_retrieval"),
            "sub_queries": plan.get("sub_queries") or [],
            "reason": plan.get("reason"),
            "fallback": plan.get("_fallback", False),
        }
        self.rewritten = plan.get("rewritten") or self.question

    def set_retrieval(self, stats: dict) -> None:
        self.retrieval = stats

    def set_rerank(self, stats: dict, before: list[dict], after: list[dict]) -> None:
        """记录重排前后的顺序对比——这是判断重排有没有帮上忙的直接依据。"""
        def label(hit: dict) -> dict:
            return {
                "id": hit.get("id"),
                "section": (hit.get("section") or "")[:40],
                "vector_rank": hit.get("vector_rank"),
                "bm25_rank": hit.get("bm25_rank"),
                "rrf_score": hit.get("rrf_score"),
                "snippet": (hit.get("text") or "")[:60],
            }

        self.rerank = {
            **stats,
            "before": [label(hit) for hit in before[:settings.rerank_top_n * 2]],
            "after": [label(hit) for hit in after],
        }

    def add_tool_calls(self, calls: list[dict]) -> None:
        self.tool_calls.extend(calls)

    def add_guard_risks(self, risks: list[dict]) -> None:
        self.guard_risks.extend(risks)

    def set_usage(self, usage: dict) -> None:
        self.usage = usage

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.started) * 1000)

    def to_payload(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "question": self.question,
            "rewritten": self.rewritten,
            "plan": json.dumps(self.plan, ensure_ascii=False),
            "stages": json.dumps(self.stages, ensure_ascii=False),
            "retrieval": json.dumps(self.retrieval, ensure_ascii=False),
            "rerank": json.dumps(self.rerank, ensure_ascii=False),
            "tool_calls": json.dumps(self.tool_calls, ensure_ascii=False),
            "guard_risks": json.dumps(self.guard_risks, ensure_ascii=False),
            "usage": json.dumps(self.usage, ensure_ascii=False),
            "hops": self.hops,
            "total_ms": self.elapsed_ms(),
            "source_count": self.source_count,
            "answer_chars": self.answer_chars,
            "refused": self.refused,
            "model": settings.llm_model if not settings.use_mock_llm else "mock",
        }


async def save(db: AsyncSession, recorder: TraceRecorder, message_id: int | None = None) -> str | None:
    """把 trace 落库。追踪本身不应影响主流程，任何异常都吞掉并返回 None。"""
    if not settings.trace_enabled:
        return None
    try:
        trace = Trace(id=recorder.trace_id, message_id=message_id, **recorder.to_payload())
        db.add(trace)
        await db.commit()
        await db.refresh(trace)
        return trace.id
    except Exception as exc:  # noqa: BLE001  可观测是增强项，不能拖垮问答
        print(f"[trace] 落库失败：{exc}")
        await db.rollback()
        return None
