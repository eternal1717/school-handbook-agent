"""FastAPI 应用入口：注册路由、CORS、静态页面、启动初始化。"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select

from app.config import BACKEND_DIR, settings
from app.db.session import SessionLocal, init_db
from app.models import Conversation, KnowledgeDocument, LongTermMemory
from app.routers import chat, history, knowledge, observability
from app.services import hybrid, vector_store
from app.services.utils import run_sync

STATIC_DIR = BACKEND_DIR / "static"
FRONTEND_DIST = BACKEND_DIR.parent / "frontend" / "dist"


def _startup_banner() -> str:
    mode = "演示模式（未配置 API Key）" if settings.use_mock_llm else f"真实调用 {settings.llm_model}"
    # 把实际生效的链路拼出来，启动就能确认配置有没有按预期生效
    chain = []
    if settings.query_rewrite_enabled:
        chain.append("改写")
    chain.append("混合检索" if settings.hybrid_enabled else "纯向量")
    if settings.rerank_enabled:
        chain.append("重排")
    if settings.agent_enabled:
        chain.append(f"自省×{settings.agent_max_hops}")
    if settings.guard_enabled:
        chain.append("注入防护")

    thinking = f"开（{settings.llm_thinking}）" if settings.thinking_for("answer") else "关"

    return (
        "\n" + "=" * 66 + "\n"
        "  知津 · 校园规章智能问答\n"
        f"  访问地址: http://{settings.host}:{settings.port}/\n"
        f"  接口文档: http://{settings.host}:{settings.port}/docs\n"
        f"  大模型  : {mode}\n"
        f"  思维链  : {thinking}\n"
        f"  检索链  : {' → '.join(chain)}\n"
        f"  召回池  : 向量 {settings.vector_candidates} + BM25 {settings.bm25_candidates}"
        f" → 融合 {settings.fused_candidates} → 重排取 {settings.rerank_top_n}\n"
        f"  向量库  : {settings.chroma_path}\n"
        f"  数据库  : {settings.database_url_resolved}\n"
        "=" * 66 + "\n"
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_dirs()
    await init_db()
    print(_startup_banner())
    yield


app = FastAPI(
    title="知津 · 校园规章智能问答",
    description="上传校园规章文件，支持带引用来源的问答与跨会话本地记忆。",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本地开发放开，方便 Vite 前端（5173）直接联调
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(knowledge.router)
app.include_router(history.router)
app.include_router(observability.router)


@app.get("/health", tags=["system"])
async def health() -> dict:
    return {"status": "ok", "mock": settings.use_mock_llm, "model": settings.llm_model}


@app.get("/api/stats", tags=["system"])
async def stats() -> dict:
    """给前端首页用的概览数据。"""
    from app.services import observability_service

    async with SessionLocal() as db:
        doc_count = await db.scalar(select(func.count()).select_from(KnowledgeDocument))
        conversation_count = await db.scalar(select(func.count()).select_from(Conversation))
        memory_count = await db.scalar(select(func.count()).select_from(LongTermMemory))
        feedback = await observability_service.feedback_summary(db)

    # BM25 的词表大小：拿不到也算正常（首次启动尚未建索引），不要让统计接口挂掉
    try:
        bm25_terms = len(hybrid.get_bm25().idf)
    except Exception:  # noqa: BLE001
        bm25_terms = 0

    return {
        "documents": int(doc_count or 0),
        "conversations": int(conversation_count or 0),
        "memories": int(memory_count or 0),
        "vectors": await run_sync(vector_store.count),
        "bm25_terms": bm25_terms,
        "llm_mode": "mock" if settings.use_mock_llm else "api",
        "llm_model": settings.llm_model,
        "llm_thinking": settings.llm_thinking,
        "embedding_model": Path(settings.embedding_model_path).name,
        "threshold": settings.similarity_threshold,
        "top_k": settings.top_k,
        # 检索链路开关，前端在顶部展示成标签，方便演示时一眼看出跑了哪条路径
        "pipeline": {
            "hybrid": settings.hybrid_enabled,
            "rerank": settings.rerank_enabled,
            "agent": settings.agent_enabled,
            "max_hops": settings.agent_max_hops,
            "guard": settings.guard_enabled,
            "trace": settings.trace_enabled,
            "vector_candidates": settings.vector_candidates,
            "bm25_candidates": settings.bm25_candidates,
            "fused_candidates": settings.fused_candidates,
            "rerank_top_n": settings.rerank_top_n,
        },
        "feedback": feedback,
    }


@app.get("/", include_in_schema=False)
async def index():
    """优先返回构建好的 Vite 前端，没有则返回免安装页面。"""
    if (FRONTEND_DIST / "index.html").exists():
        return FileResponse(FRONTEND_DIST / "index.html")
    page = STATIC_DIR / "index.html"
    if page.exists():
        return FileResponse(page)
    return JSONResponse({"message": "前端页面不存在，可直接访问 /docs 调试接口"})


# 静态资源（免安装前端：Vue3 + Element Plus 本地文件）
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 若已构建 Vite 前端，把它挂在根路径（放在所有接口路由之后，避免抢路由）
if FRONTEND_DIST.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
