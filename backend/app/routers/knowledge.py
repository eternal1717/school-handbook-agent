"""知识库接口：上传、列表、删除、按本地路径导入。"""
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import UPLOAD_DIR
from app.db.session import get_db
from app.schemas import DocumentOut
from app.services import knowledge_service

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

ALLOWED_SUFFIX = {".docx", ".pdf", ".txt", ".md"}


@router.post("/upload", response_model=dict)
async def upload(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)) -> dict:
    """上传文档并入库。"""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIX:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型：{suffix}")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_DIR / Path(file.filename or "unnamed").name
    content = await file.read()
    target.write_bytes(content)

    try:
        return await knowledge_service.ingest_file(db, target, original_name=file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class LocalIngestRequest(BaseModel):
    path: str


@router.post("/ingest-local", response_model=dict)
async def ingest_local(payload: LocalIngestRequest, db: AsyncSession = Depends(get_db)) -> dict:
    """直接导入本机某个文件（免上传，方便批量灌手册）。"""
    path = Path(payload.path)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=400, detail=f"文件不存在：{path}")
    if path.suffix.lower() not in ALLOWED_SUFFIX:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型：{path.suffix}")

    try:
        return await knowledge_service.ingest_file(db, path, original_name=path.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/list", response_model=list[DocumentOut])
async def list_documents(db: AsyncSession = Depends(get_db)) -> list[DocumentOut]:
    documents = await knowledge_service.list_documents(db)
    return [DocumentOut.model_validate(doc) for doc in documents]


@router.get("/{doc_id}/chunks", response_model=dict)
async def document_chunks(doc_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """查看某文档切成了哪些知识块（按块序号排列）。"""
    chunks = await knowledge_service.get_document_chunks(db, doc_id)
    if chunks is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {"doc_id": doc_id, "count": len(chunks), "chunks": chunks}


@router.delete("/{doc_id}", response_model=dict)
async def delete_document(doc_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    ok = await knowledge_service.delete_document(db, doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {"deleted": True, "id": doc_id}
