"""回答反馈表：点赞 / 点踩 + 备注。

## 为什么这是必需的一环

没有反馈，评测集就只能靠人工想象问题，永远覆盖不到真实用户的问法。
有了反馈，「点踩的那批问题」直接就是最有价值的 badcase 池——
它们比随机抽的问题更能暴露系统短板。

所以这张表不是产品装饰，它是**评测集的来源**：
每天挑几条点踩记录，补进 tests/eval_dataset.json，再跑一次评测看指标变化。

用户对同一条回答重新评价时按 (message_id, user_id) 覆盖，不产生多条记录。
"""
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class Feedback(Base):
    __tablename__ = "feedback"
    __table_args__ = (
        UniqueConstraint("message_id", "user_id", name="uq_feedback_message_user"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_id: Mapped[int] = mapped_column(Integer, index=True)
    conversation_id: Mapped[str] = mapped_column(String(64), index=True)
    trace_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)

    rating: Mapped[str] = mapped_column(String(8))  # up / down
    comment: Mapped[str] = mapped_column(Text, default="")
    question: Mapped[str] = mapped_column(Text, default="")  # 冗余存一份，方便直接导出 badcase

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
