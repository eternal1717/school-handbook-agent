"""长期记忆表：跨会话记忆。

为什么不用 LangGraph Store？官方 Store 只提供 InMemory / SQLite / Postgres 实现，
MySQL 与统一表结构下自己维护更可控。这里用 (user_id, namespace, key) 做唯一键，
value 存 JSON 字符串，既能存字符串也能存列表/字典。
"""
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class LongTermMemory(Base):
    __tablename__ = "long_term_memory"
    __table_args__ = (
        UniqueConstraint("user_id", "namespace", "key", name="uq_memory_user_ns_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    namespace: Mapped[str] = mapped_column(String(64), default="school_handbook")
    key: Mapped[str] = mapped_column(String(128))
    value: Mapped[str] = mapped_column(Text)  # JSON 字符串
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )
