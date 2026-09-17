"""历史会话接口 + 长期记忆查看接口。"""
import json

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas import ConversationOut, MemoryOut, MessageOut, SourceItem
from app.services import memory_service, observability_service

router = APIRouter(prefix="/api", tags=["history"])


@router.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(
    user_id: str = Query("default_user"), db: AsyncSession = Depends(get_db)
) -> list[ConversationOut]:
    items = await memory_service.list_conversations(db, user_id)
    return [ConversationOut(**item) for item in items]


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageOut])
async def list_messages(
    conversation_id: str,
    user_id: str = Query("default_user"),
    db: AsyncSession = Depends(get_db),
) -> list[MessageOut]:
    """读会话历史消息。

    带上 user_id 是为了把「当前用户对这些回答的评价」一起返回，
    否则刷新页面后点赞/点踩的高亮状态会丢。
    """
    messages = await memory_service.list_messages(db, conversation_id)
    ratings = await observability_service.list_feedback_for_conversation(db, conversation_id, user_id)

    result = []
    for message in messages:
        try:
            sources = [SourceItem(**item) for item in json.loads(message.sources or "[]")]
        except (json.JSONDecodeError, TypeError):
            sources = []
        result.append(
            MessageOut(
                id=message.id,
                role=message.role,
                content=message.content,
                created_at=message.created_at,
                sources=sources,
                reasoning=message.reasoning or "",
                trace_id=message.trace_id,
                feedback=ratings.get(message.id, ""),
            )
        )
    return result


@router.delete("/conversations/{conversation_id}", response_model=dict)
async def delete_conversation(conversation_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    await memory_service.delete_conversation(db, conversation_id)
    return {"deleted": True, "id": conversation_id}


@router.get("/memory", response_model=list[MemoryOut])
async def list_memory(
    user_id: str = Query("default_user"), db: AsyncSession = Depends(get_db)
) -> list[MemoryOut]:
    """查看某用户的长期记忆（演示跨会话记忆用）。"""
    items = await memory_service.list_memories(db, user_id)
    return [MemoryOut.model_validate(item) for item in items]


@router.get("/memory/users", response_model=list[dict])
async def list_memory_users(db: AsyncSession = Depends(get_db)) -> list[dict]:
    """列出有长期记忆的用户及条数（记忆页的用户切换用）。"""
    return await memory_service.list_memory_users(db)


@router.delete("/memory/{memory_id}", response_model=dict)
async def delete_memory(memory_id: int, db: AsyncSession = Depends(get_db)) -> dict:
    await memory_service.delete_memory(db, memory_id)
    return {"deleted": True, "id": memory_id}
