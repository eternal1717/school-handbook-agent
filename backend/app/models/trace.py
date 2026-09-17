"""链路追踪表：把「这一问是怎么答出来的」完整记下来。

## 为什么需要它

改造前，一个回答质量不好，只能靠猜：是改写把问题带偏了？是检索没召回？
是重排排错了？还是模型拿到了正确资料却没用好？

这四个环节的故障在用户眼里长得一模一样——「答得不对」。
没有 trace 就只能一个个改参数试。有了它，打开记录一眼就能定位到哪一环掉链子。

一条 trace = 一次提问。字段分成三组：
1. **输入侧**：原始问题、改写后的问题、意图、复杂度
2. **过程侧**：各阶段耗时、检索统计、重排结果、工具调用、注入风险
3. **输出侧**：是否拒答、回答长度、token 消耗

存 JSON 字符串而不是拆成多张表，是因为这些结构会随着流程演进而变化，
拆表会让每次加一个字段都要写迁移。分析时用 json_extract 就够了。
"""
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class Trace(Base):
    __tablename__ = "trace"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex)
    conversation_id: Mapped[str] = mapped_column(String(64), index=True)
    # 关联到具体的 assistant 消息，反馈表靠它和回答对上
    message_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)

    question: Mapped[str] = mapped_column(Text)
    rewritten: Mapped[str] = mapped_column(Text, default="")     # 最终用于检索的查询
    plan: Mapped[str] = mapped_column(Text, default="{}")        # 意图 / 复杂度 / 子查询 / 判断依据
    stages: Mapped[str] = mapped_column(Text, default="[]")      # 各阶段耗时与决策
    retrieval: Mapped[str] = mapped_column(Text, default="{}")   # 两路召回统计 + 融合统计
    rerank: Mapped[str] = mapped_column(Text, default="{}")      # 重排前后顺序
    tool_calls: Mapped[str] = mapped_column(Text, default="[]")  # Agent 工具调用记录
    guard_risks: Mapped[str] = mapped_column(Text, default="[]") # 命中的注入风险
    usage: Mapped[str] = mapped_column(Text, default="{}")       # token 统计

    hops: Mapped[int] = mapped_column(Integer, default=0)        # 自省循环实际跳数
    total_ms: Mapped[int] = mapped_column(Integer, default=0)
    source_count: Mapped[int] = mapped_column(Integer, default=0)
    answer_chars: Mapped[int] = mapped_column(Integer, default=0)
    refused: Mapped[bool] = mapped_column(Boolean, default=False)
    model: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)
