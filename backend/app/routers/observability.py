"""可观测接口：用户反馈 + 链路追踪查询。"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas import FeedbackRequest
from app.services import observability_service

router = APIRouter(prefix="/api", tags=["observability"])


# ---------------- 反馈 ----------------

@router.post("/feedback", response_model=dict)
async def submit_feedback(payload: FeedbackRequest, db: AsyncSession = Depends(get_db)) -> dict:
    """提交点赞 / 点踩。同一条回答重复提交会覆盖之前的选择。"""
    if payload.rating not in {"up", "down"}:
        raise HTTPException(status_code=400, detail="rating 只能是 up 或 down")

    item = await observability_service.upsert_feedback(
        db,
        message_id=payload.message_id,
        user_id=payload.user_id,
        rating=payload.rating,
        conversation_id=payload.conversation_id,
        trace_id=payload.trace_id,
        question=payload.question,
        comment=payload.comment,
    )
    if item is None:
        raise HTTPException(status_code=404, detail="消息不存在，无法提交反馈")
    return {"ok": True, "message_id": item.message_id, "rating": item.rating}


@router.delete("/feedback/{message_id}", response_model=dict)
async def cancel_feedback(
    message_id: int,
    user_id: str = Query("default_user"),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """取消评价（再次点击同一个按钮时用）。"""
    removed = await observability_service.delete_feedback(db, message_id, user_id)
    return {"deleted": removed, "message_id": message_id}


@router.get("/feedback/summary", response_model=dict)
async def feedback_summary(db: AsyncSession = Depends(get_db)) -> dict:
    """满意度概览：赞 / 踩 / 满意率。"""
    return await observability_service.feedback_summary(db)


@router.get("/feedback/badcases", response_model=dict)
async def badcases(limit: int = Query(50, ge=1, le=200), db: AsyncSession = Depends(get_db)) -> dict:
    """点踩记录列表。"""
    items = await observability_service.list_badcases(db, limit)
    return {"count": len(items), "items": items}


@router.get("/feedback/export", response_model=dict)
async def export_badcases(limit: int = Query(50, ge=1, le=200), db: AsyncSession = Depends(get_db)) -> dict:
    """把点踩记录导出成评测集条目——评测集应该从真实失败里长出来，而不是凭空想。"""
    return await observability_service.export_badcases(db, limit)


# ---------------- 链路追踪 ----------------

@router.get("/trace/overview", response_model=dict)
async def trace_overview(db: AsyncSession = Depends(get_db)) -> dict:
    """聚合指标：平均/P50/P95 耗时、拒答率、平均跳数、各阶段平均耗时。"""
    return await observability_service.trace_overview(db)


@router.get("/trace", response_model=list[dict])
async def list_traces(
    limit: int = Query(30, ge=1, le=200),
    only_refused: bool = Query(False),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """最近的问答链路列表。"""
    return await observability_service.list_traces(db, limit, only_refused)


@router.get("/trace/{trace_id}", response_model=dict)
async def get_trace(trace_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """单条链路的完整详情：改写前后、各阶段耗时、两路召回、重排顺序、工具调用。"""
    item = await observability_service.get_trace(db, trace_id)
    if item is None:
        raise HTTPException(status_code=404, detail="trace 不存在")
    return item
