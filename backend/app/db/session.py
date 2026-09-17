"""数据库连接与会话管理（SQLAlchemy 2.0 异步 + aiosqlite）。"""
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""


engine = create_async_engine(
    settings.database_url_resolved,
    echo=False,
    future=True,
)

SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI 依赖注入：每个请求一个数据库会话。"""
    async with SessionLocal() as session:
        yield session


async def init_db() -> None:
    """建目录 + 建表（幂等，重复启动不会报错）。"""
    from app import models  # noqa: F401  导入模型以便注册到 Base.metadata

    settings.ensure_dirs()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _apply_light_migrations(conn)


# 表结构后加的列：create_all 只建新表，不会给已有表补列，所以这里手动补一次
_ADDED_COLUMNS = {
    "knowledge_document": {
        "skipped_chunk_count": "INTEGER NOT NULL DEFAULT 0",
    },
    # 改造后新增：把思维链和 trace 关联一起存进消息表。
    # 用 TEXT/VARCHAR 而不是 NOT NULL（除 reasoning 给了默认值），
    # 是为了让老数据补列时能直接填上默认值，不会因约束失败。
    "message": {
        "reasoning": "TEXT DEFAULT ''",
        "trace_id": "VARCHAR(64)",
    },
}


async def _apply_light_migrations(conn) -> None:
    """轻量迁移：给已存在的旧表补上后加的列（幂等）。"""
    if not engine.url.get_backend_name().startswith("sqlite"):
        return

    for table, columns in _ADDED_COLUMNS.items():
        result = await conn.exec_driver_sql(f"PRAGMA table_info({table})")
        existing = {row[1] for row in result.fetchall()}
        if not existing:  # 表不存在（理论上 create_all 已建好）
            continue
        for name, ddl in columns.items():
            if name not in existing:
                await conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
