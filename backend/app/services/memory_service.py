"""记忆服务：短期记忆（会话内）+ 长期记忆（跨会话）。

数据流向：
1. 每次提问 → 先存 user 消息，再存 assistant 消息（短期记忆落库）。
2. 提问前 → 从 message 表读最近 N 轮，从 long_term_memory 表读全部长期记忆。
3. 每累计 N 轮 → 触发一次摘要，把要点写进长期记忆（自动记忆）。
4. 用户说「记住…」→ 立刻抽取要点写进长期记忆（显式记忆）。
"""
import json
import uuid
from datetime import datetime

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Conversation, Feedback, LongTermMemory, Message, Trace
from app.services import llm

NAMESPACE = "school_handbook"


# ---------------- 会话 ----------------
async def get_or_create_conversation(
    db: AsyncSession, conversation_id: str | None, user_id: str, first_question: str
) -> Conversation:
    """有 conversation_id 就复用，没有就新建一个会话。"""
    if conversation_id:
        existing = await db.get(Conversation, conversation_id)
        if existing is not None:
            existing.updated_at = datetime.now()
            await db.commit()
            return existing

    new_id = conversation_id or uuid.uuid4().hex
    conversation = Conversation(
        id=new_id,
        user_id=user_id,
        title=(first_question[:30] or "新会话"),
    )
    db.add(conversation)
    await db.commit()
    return conversation


async def list_conversations(db: AsyncSession, user_id: str) -> list[dict]:
    """会话列表，附带消息条数（用于侧栏显示）。"""
    result = await db.execute(
        select(Conversation).where(Conversation.user_id == user_id).order_by(Conversation.updated_at.desc())
    )
    conversations = result.scalars().all()

    items = []
    for conversation in conversations:
        count = await db.scalar(
            select(func.count()).select_from(Message).where(Message.conversation_id == conversation.id)
        )
        items.append(
            {
                "id": conversation.id,
                "title": conversation.title,
                "created_at": conversation.created_at,
                "updated_at": conversation.updated_at,
                "message_count": int(count or 0),
            }
        )
    return items


async def delete_conversation(db: AsyncSession, conversation_id: str) -> None:
    """删除会话，连同它的消息、反馈和链路一起清掉。

    只删消息是不够的：Feedback.message_id、Trace.message_id 指向的就是消息，
    消息没了它们就成了悬空数据——满意度统计把它们算进去，
    badcase 列表里点开却找不到那条回答。删会话时一并清掉，统计口径才自洽。

    顺序上先删子表再删父表（conversation），避免中途失败留下更乱的半截状态。
    """
    await db.execute(delete(Feedback).where(Feedback.conversation_id == conversation_id))
    await db.execute(delete(Trace).where(Trace.conversation_id == conversation_id))
    await db.execute(delete(Message).where(Message.conversation_id == conversation_id))
    await db.execute(delete(Conversation).where(Conversation.id == conversation_id))
    await db.commit()


# ---------------- 短期记忆 ----------------
async def add_message(
    db: AsyncSession,
    conversation_id: str,
    role: str,
    content: str,
    sources: list[dict] | None = None,
    reasoning: str = "",
    trace_id: str | None = None,
) -> Message:
    """写一条消息。assistant 消息会额外带上来源、思维链和 trace 关联。"""
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        sources=json.dumps(sources or [], ensure_ascii=False),
        reasoning=reasoning or "",
        trace_id=trace_id,
    )
    db.add(message)
    await db.commit()
    await db.refresh(message)
    return message


async def count_user_rounds(db: AsyncSession, conversation_id: str) -> int:
    """这个会话里用户问了几轮（用来判断是否该触发自动摘要）。"""
    count = await db.scalar(
        select(func.count())
        .select_from(Message)
        .where(Message.conversation_id == conversation_id, Message.role == "user")
    )
    return int(count or 0)


async def load_history(db: AsyncSession, conversation_id: str, rounds: int | None = None) -> list[dict]:
    """读最近 N 轮对话，按时间正序返回，可直接拼进 prompt。"""
    rounds = rounds or settings.short_term_rounds
    limit = rounds * 2  # 一轮 = 一问一答

    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.id.desc())
        .limit(limit)
    )
    messages = list(reversed(result.scalars().all()))
    return [{"role": m.role, "content": m.content} for m in messages]


async def list_messages(db: AsyncSession, conversation_id: str) -> list[Message]:
    result = await db.execute(
        select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id.asc())
    )
    return list(result.scalars().all())


# ---------------- 长期记忆 ----------------
async def save_memory(db: AsyncSession, user_id: str, key: str, value: str) -> None:
    """按 (user_id, namespace, key) 覆盖式写入，同一个 key 只保留最新值。"""
    result = await db.execute(
        select(LongTermMemory).where(
            LongTermMemory.user_id == user_id,
            LongTermMemory.namespace == NAMESPACE,
            LongTermMemory.key == key,
        )
    )
    existing = result.scalar_one_or_none()
    if existing is not None:
        existing.value = value
        existing.updated_at = datetime.now()
    else:
        db.add(
            LongTermMemory(
                user_id=user_id, namespace=NAMESPACE, key=key, value=value
            )
        )
    await db.commit()


async def list_memories(db: AsyncSession, user_id: str) -> list[LongTermMemory]:
    result = await db.execute(
        select(LongTermMemory)
        .where(LongTermMemory.user_id == user_id, LongTermMemory.namespace == NAMESPACE)
        .order_by(LongTermMemory.updated_at.desc())
    )
    return list(result.scalars().all())


async def list_memory_users(db: AsyncSession) -> list[dict]:
    """列出所有有长期记忆的用户及其条数（前端记忆页切换查看用）。"""
    result = await db.execute(
        select(LongTermMemory.user_id, func.count(LongTermMemory.id))
        .where(LongTermMemory.namespace == NAMESPACE)
        .group_by(LongTermMemory.user_id)
        .order_by(func.count(LongTermMemory.id).desc())
    )
    return [{"user_id": user_id, "count": count} for user_id, count in result.all()]


async def delete_memory(db: AsyncSession, memory_id: int) -> None:
    await db.execute(delete(LongTermMemory).where(LongTermMemory.id == memory_id))
    await db.commit()


async def process_explicit_memory(db: AsyncSession, user_id: str, user_text: str) -> str | None:
    """检测并保存显式记忆（用户说「记住…」）。返回保存的内容，没保存则返回 None。"""
    extracted = await llm.extract_memory(user_text)
    if not extracted:
        return None

    key = extracted["key"]
    # 同一个 key 已存在时加时间后缀，避免覆盖掉之前存的事实
    existing = await list_memories(db, user_id)
    if any(item.key == key for item in existing):
        key = f"{key}_{int(datetime.now().timestamp())}"

    await save_memory(db, user_id, key, extracted["value"])
    return extracted["value"]


def format_memory_for_prompt(memories: list[LongTermMemory]) -> str:
    """把长期记忆渲染成一段 system 提示文本。"""
    if not memories:
        return ""
    lines = [f"- {item.value}" for item in memories if item.value.strip()]
    if not lines:
        return ""
    return "关于这位用户，你已知晓以下长期信息（回答时可以自然地利用）：\n" + "\n".join(lines)
