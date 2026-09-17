"""聊天接口：SSE 流式输出。"""
import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas import ChatRequest
from app.services import rag_service

router = APIRouter(prefix="/api", tags=["chat"])


def _sse(event: dict) -> str:
    """按 SSE 协议格式化：每条事件以 data: 开头，用空行分隔。"""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(payload: ChatRequest, db: AsyncSession = Depends(get_db)) -> StreamingResponse:
    async def event_generator():
        try:
            async for event in rag_service.answer_stream(
                db=db,
                question=payload.question.strip(),
                user_id=payload.user_id,
                conversation_id=payload.conversation_id,
            ):
                yield _sse(event)
        except Exception as exc:  # noqa: BLE001  兜底：任何异常都变成一条 error 事件，避免前端卡死
            yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        # end 事件表示「流正常结束」，前端据此关闭 loading 状态
        yield _sse({"type": "end"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
