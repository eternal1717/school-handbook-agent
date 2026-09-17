"""规章结构化接口：规则库浏览、情境诊断、办事流程。

和检索接口的区别：检索接口回答「哪条这么规定」，
这里的接口回答「我这情况会不会触发它」和「我要按什么顺序去办」。
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.services import rulebook

router = APIRouter(prefix="/api/rulebook", tags=["rulebook"])


class DiagnoseRequest(BaseModel):
    """情境诊断入参。

    `situation` 是学生的自述，不是提问——「我这学期挂了3门课，还缺课20学时」
    这种带具体数字和事实的描述才是它期待的输入。
    """
    situation: str = Field(..., min_length=1, max_length=1000)
    limit: int | None = Field(default=None, ge=1, le=20)


@router.get("/stats")
def rulebook_stats() -> dict:
    """规则库概览：抽了多少条规则、多少条流程、按种类怎么分布。"""
    if not settings.rulebook_enabled:
        return {"enabled": False, "reason": "配置中已关闭 rulebook_enabled"}
    return {"enabled": True, **rulebook.stats()}


@router.post("/rebuild")
def rebuild() -> dict:
    """强制重建规则库。入库新文档后如果统计没跟着变，可以手动触发。"""
    return rulebook.rebuild()


@router.get("/rules")
def get_rules(q: str = "", kind: str = "", limit: int = 50) -> dict:
    """浏览 / 搜索规则。q 走 BM25，复用检索层的同一套实现。"""
    if not settings.rulebook_enabled:
        raise HTTPException(status_code=503, detail="规则库已在配置中关闭")
    return rulebook.list_rules(query=q, kind=kind, limit=max(1, min(limit, 200)))


@router.post("/diagnose")
def diagnose(payload: DiagnoseRequest) -> dict:
    """情境诊断：学生自述情况 → 相关规章规定 + 数值比对提示。

    注意返回里的 note 字段——这不是资格判定，是条文关联，
    前端应把它一起展示出来，避免用户把提示当成官方结论。
    """
    if not settings.rulebook_enabled:
        raise HTTPException(status_code=503, detail="规则库已在配置中关闭")
    return rulebook.diagnose(payload.situation, payload.limit)


@router.get("/procedures")
def get_procedures(q: str = "", limit: int = 30) -> dict:
    """浏览 / 搜索办事流程。不给 q 时按置信度排序返回全部。"""
    if not settings.rulebook_enabled:
        raise HTTPException(status_code=503, detail="规则库已在配置中关闭")
    result = rulebook.list_procedures(query=q, limit=max(1, min(limit, 100)))
    # 不给查询词时按置信度排——high 的多是自带结构标记的（箭头链、序号项），
    # 比句式猜出来的 medium 更值得先看。
    if not q:
        result["items"].sort(key=lambda item: item.get("confidence") != "high")
    return result


@router.get("/procedures/{procedure_id}")
def get_procedure(procedure_id: str) -> dict:
    item = rulebook.get_procedure(procedure_id)
    if not item:
        raise HTTPException(status_code=404, detail="未找到该流程")
    return item
