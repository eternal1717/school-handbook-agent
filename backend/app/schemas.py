"""接口的请求/响应模型（Pydantic）。"""
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

# 提问长度上限。定这个数的依据：
# - 真正的校园规章提问都在 100 字以内，2000 字已经宽松到不可能误伤；
# - 但不设上限时，把几万字贴进来会**原样送进大模型**——token 费用、延迟
#   都不可控，而且检索层也捞不出有效信号，纯属浪费。
QUESTION_MAX_CHARS = 2000


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=QUESTION_MAX_CHARS, description="用户提问")
    user_id: str = Field("default_user", description="用户标识，长期记忆按它隔离")
    conversation_id: str | None = Field(None, description="不传则自动新建会话")

    @field_validator("question")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        """拦掉「全是空白」的提问。

        只靠 min_length=1 不够：`"   "` 长度是 3，能过校验，但它 strip 之后
        是空字符串——会被判成闲聊，还会在库里留下一个标题为「新会话」的空会话。
        这里直接去掉首尾空白，顺便让下游不用再 strip。
        """
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("问题不能为空")
        return cleaned


class SourceItem(BaseModel):
    """一条引用来源。

    除了展示用的基本信息，还带上两路检索各自的排名和融合分——
    这样前端可以说明「这条同时被向量和关键词命中」，也让 trace 有据可查。
    """

    id: str = ""
    # 定位原文用：doc_id 找到文档，chunk_index 找到具体第几块
    doc_id: str = ""
    chunk_index: int | None = None
    filename: str
    section: str = ""
    page: int | None = None
    similarity: float | None = Field(None, description="向量余弦相似度；仅 BM25 命中时为 null")
    bm25_score: float | None = Field(None, description="BM25 分数，无上界，只能横向比较")
    rrf_score: float | None = Field(None, description="RRF 融合分")
    vector_rank: int | None = None
    bm25_rank: int | None = None
    risk: list[str] = Field(default_factory=list, description="命中的注入风险类型")
    text: str = ""


class DocumentOut(BaseModel):
    id: str
    filename: str
    file_type: str
    size_bytes: int
    char_count: int
    chunk_count: int
    skipped_chunk_count: int = 0
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationOut(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime
    sources: list[SourceItem] = []
    # 思维链：改造后会把思考过程一起存下来，历史会话回看时仍能看到
    reasoning: str = ""
    trace_id: str | None = None
    feedback: str = ""  # 当前用户对这条回答的评价：up / down / 空


class MemoryOut(BaseModel):
    id: int
    user_id: str
    key: str
    value: str
    updated_at: datetime

    model_config = {"from_attributes": True}


class FeedbackRequest(BaseModel):
    message_id: int
    rating: str = Field(..., description="up 或 down")
    user_id: str = Field("default_user")
    conversation_id: str = ""
    trace_id: str | None = None
    question: str = ""
    comment: str = ""


class TraceOut(BaseModel):
    id: str
    conversation_id: str
    message_id: int | None = None
    user_id: str
    question: str
    rewritten: str = ""
    plan: dict = {}
    stages: list = []
    retrieval: dict = {}
    rerank: dict = {}
    tool_calls: list = []
    guard_risks: list = []
    usage: dict = {}
    hops: int = 0
    total_ms: int = 0
    source_count: int = 0
    answer_chars: int = 0
    refused: bool = False
    model: str = ""
    created_at: datetime
