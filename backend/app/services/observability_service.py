"""可观测服务：反馈读写 + trace 查询。

## 为什么把「导出 badcase」做成接口

反馈的价值不在于统计有多少个赞，而在于**把点踩的问题变成回归测试集**。
所以这里直接提供 `export_badcases()`，一键把点踩记录导出成
tests/eval_dataset.json 能吃的格式——评测集不该靠人凭空想问题，
而应该从真实失败里长出来。
"""
import json

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Feedback, Message, Trace


# ---------------- 反馈 ----------------

async def upsert_feedback(
    db: AsyncSession,
    message_id: int,
    user_id: str,
    rating: str,
    conversation_id: str = "",
    trace_id: str | None = None,
    question: str = "",
    comment: str = "",
) -> Feedback | None:
    """写反馈。同一条消息同一用户重复评价时覆盖，不产生多条记录。

    message_id 指向的消息不存在时返回 None（调用方回 404）。
    message_id 是「这条回答」的外键语义，不校验就写会留下孤儿反馈：
    满意度统计被它们拉偏，badcase 导出里还混进无法复核的问题。
    """
    if await db.get(Message, message_id) is None:
        return None

    result = await db.execute(
        select(Feedback).where(
            Feedback.message_id == message_id, Feedback.user_id == user_id
        )
    )
    existing = result.scalar_one_or_none()

    if existing is not None:
        existing.rating = rating
        # 截断口径必须和新建分支一致。早先这里直接赋原值，
        # 于是「先写一条短评、再改成超长文本」能绕过 500 字上限。
        existing.comment = comment[:500]
        if trace_id:
            existing.trace_id = trace_id
        if question:
            existing.question = question[:500]
    else:
        existing = Feedback(
            message_id=message_id,
            conversation_id=conversation_id,
            trace_id=trace_id,
            user_id=user_id,
            rating=rating,
            comment=comment[:500],
            question=question[:500],
        )
        db.add(existing)

    await db.commit()
    await db.refresh(existing)
    return existing


async def delete_feedback(db: AsyncSession, message_id: int, user_id: str) -> bool:
    result = await db.execute(
        delete(Feedback).where(
            Feedback.message_id == message_id, Feedback.user_id == user_id
        )
    )
    await db.commit()
    return bool(result.rowcount)


async def list_feedback_for_conversation(db: AsyncSession, conversation_id: str, user_id: str) -> dict[int, str]:
    """拿到某个会话里「当前用户」对每条消息的评价，供历史回看时还原按钮状态。"""
    result = await db.execute(
        select(Feedback.message_id, Feedback.rating).where(
            Feedback.conversation_id == conversation_id, Feedback.user_id == user_id
        )
    )
    return {message_id: rating for message_id, rating in result.all()}


async def feedback_summary(db: AsyncSession) -> dict:
    """总体满意度概览。"""
    result = await db.execute(
        select(Feedback.rating, func.count(Feedback.id)).group_by(Feedback.rating)
    )
    counts = {rating: count for rating, count in result.all()}
    up = int(counts.get("up", 0))
    down = int(counts.get("down", 0))
    total = up + down
    return {
        "up": up,
        "down": down,
        "total": total,
        "satisfaction": round(up / total, 4) if total else None,
    }


async def list_badcases(db: AsyncSession, limit: int = 50) -> list[dict]:
    """列出所有点踩记录，按时间倒序。"""
    result = await db.execute(
        select(Feedback).where(Feedback.rating == "down").order_by(Feedback.created_at.desc()).limit(limit)
    )
    return [
        {
            "message_id": item.message_id,
            "trace_id": item.trace_id,
            "question": item.question,
            "comment": item.comment,
            "created_at": item.created_at.strftime("%Y-%m-%d %H:%M"),
        }
        for item in result.scalars().all()
    ]


async def export_badcases(db: AsyncSession, limit: int = 50) -> dict:
    """导出成评测集格式，直接可以粘进 tests/eval_dataset.json 的 cases 数组。

    评测集条目里只填「问题」和「应该命中的关键词」——
    后者由人工补，因为「答得对不对」这件事机器判不准，
    但「有没有检索到正确条款」可以靠关键词覆盖来自动打分。
    """
    result = await db.execute(
        select(Feedback).where(Feedback.rating == "down").order_by(Feedback.created_at.desc()).limit(limit)
    )
    cases = []
    for item in result.scalars().all():
        if not item.question:
            continue
        cases.append({
            "question": item.question,
            "expected_keywords": [],
            "note": (item.comment or "由用户点踩导出，待补充期望关键词")[:120],
            "source": "feedback",
            "message_id": item.message_id,
        })
    return {
        "count": len(cases),
        "hint": "把 cases 数组里的条目补上 expected_keywords 后，粘贴进 tests/eval_dataset.json",
        "cases": cases,
    }


# ---------------- Trace ----------------

def _decode_trace(trace: Trace) -> dict:
    """把 trace 行里存的 JSON 字符串解回来。坏数据不该让接口 500。"""
    def load(raw: str, fallback):
        try:
            return json.loads(raw) if raw else fallback
        except (json.JSONDecodeError, TypeError):
            return fallback

    return {
        "id": trace.id,
        "conversation_id": trace.conversation_id,
        "message_id": trace.message_id,
        "user_id": trace.user_id,
        "question": trace.question,
        "rewritten": trace.rewritten,
        "plan": load(trace.plan, {}),
        "stages": load(trace.stages, []),
        "retrieval": load(trace.retrieval, {}),
        "rerank": load(trace.rerank, {}),
        "tool_calls": load(trace.tool_calls, []),
        "guard_risks": load(trace.guard_risks, []),
        "usage": load(trace.usage, {}),
        "hops": trace.hops,
        "total_ms": trace.total_ms,
        "source_count": trace.source_count,
        "answer_chars": trace.answer_chars,
        "refused": trace.refused,
        "model": trace.model,
        "created_at": trace.created_at,
    }


async def get_trace(db: AsyncSession, trace_id: str) -> dict | None:
    trace = await db.get(Trace, trace_id)
    return _decode_trace(trace) if trace else None


async def list_traces(db: AsyncSession, limit: int = 30, only_refused: bool = False) -> list[dict]:
    """最近的 trace 列表（概览用，不带大字段）。"""
    statement = select(Trace).order_by(Trace.created_at.desc()).limit(limit)
    if only_refused:
        statement = select(Trace).where(Trace.refused.is_(True)).order_by(Trace.created_at.desc()).limit(limit)

    result = await db.execute(statement)
    items = []
    for trace in result.scalars().all():
        plan = _decode_trace(trace)["plan"]
        items.append({
            "id": trace.id,
            "question": trace.question[:80],
            "rewritten": trace.rewritten[:80],
            "intent": plan.get("intent"),
            "complexity": plan.get("complexity"),
            "hops": trace.hops,
            "total_ms": trace.total_ms,
            "source_count": trace.source_count,
            "refused": trace.refused,
            "created_at": trace.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        })
    return items


async def trace_overview(db: AsyncSession) -> dict:
    """trace 的聚合指标——面试时「你的系统平均耗时多少、拒答率多少」就靠它。

    一条 SQL 出不来（SQLite 没有 percentile_cont），所以拉最近若干条在内存里算。
    规模上限是本地演示库，够用。
    """
    result = await db.execute(select(Trace).order_by(Trace.created_at.desc()).limit(500))
    traces = list(result.scalars().all())
    if not traces:
        return {"total": 0}

    durations = sorted(trace.total_ms for trace in traces)
    refused = sum(1 for trace in traces if trace.refused)
    hops = [trace.hops for trace in traces]

    def percentile(values: list[int], ratio: float) -> int:
        if not values:
            return 0
        index = min(len(values) - 1, int(len(values) * ratio))
        return values[index]

    stage_totals: dict[str, list[int]] = {}
    for trace in traces:
        for stage in _decode_trace(trace)["stages"]:
            stage_totals.setdefault(stage.get("stage", "?"), []).append(stage.get("ms", 0))

    return {
        "total": len(traces),
        "refused": refused,
        "refusal_rate": round(refused / len(traces), 4),
        "avg_ms": int(sum(durations) / len(durations)),
        "p50_ms": percentile(durations, 0.5),
        "p95_ms": percentile(durations, 0.95),
        "max_ms": durations[-1],
        "avg_hops": round(sum(hops) / len(hops), 2),
        "stage_avg_ms": {
            name: int(sum(values) / len(values)) for name, values in sorted(stage_totals.items())
        },
    }
