"""消息表：短期记忆的存储介质。

每轮问答都往这里写两条（user + assistant），下次请求时按会话读出来拼进 prompt。
"""
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class Message(Base):
    __tablename__ = "message"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16))  # user / assistant
    content: Mapped[str] = mapped_column(Text)
    # 引用来源，存 JSON 字符串，便于历史会话回看时还原来源卡片
    sources: Mapped[str] = mapped_column(Text, default="")
    # 思维链：单独存一列而不是拼进 content。
    # 拼进去会让前端、评测脚本、历史回看都要做字符串切割，迟早出错。
    reasoning: Mapped[str] = mapped_column(Text, default="")
    # 关联的链路追踪 id。用户点踩时带上它，就能直接跳到那一问的完整过程记录。
    trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
