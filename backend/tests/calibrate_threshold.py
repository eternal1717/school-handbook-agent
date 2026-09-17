"""相似度阈值校准脚本。

目的：用真实手册测出「相关问题」和「无关问题」的相似度分布，
从而把 SIMILARITY_THRESHOLD 定在一个合理的位置（而不是拍脑袋写 0.35）。

用法：D:\\python\\python.exe tests\\calibrate_threshold.py
"""
import asyncio
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(BACKEND_DIR)

from app.config import settings  # noqa: E402
from app.services import embedding as embedding_service  # noqa: E402
from app.services import vector_store  # noqa: E402
from app.services.utils import run_sync  # noqa: E402

IN_MANUAL = [
    "学生在校期间享有哪些权利",
    "请假的流程是什么",
    "考试作弊会怎么处理",
    "奖学金怎么评定",
    "学籍异动需要什么手续",
    "宿舍管理规定有哪些",
]

OUT_OF_MANUAL = [
    "今天天气怎么样",
    "红烧肉怎么做才好吃",
    "Python 怎么读取一个文件",
    "帮我写一首诗",
    "介绍一下量子力学",
]


async def measure(questions: list[str], label: str) -> list[float]:
    print(f"\n===== {label} =====")
    scores = []
    for question in questions:
        vector = await run_sync(embedding_service.encode_query, question)
        hits = await run_sync(vector_store.query, vector, settings.top_k)
        top = hits[0] if hits else None
        score = top["similarity"] if top else 0.0
        scores.append(score)
        section = (top or {}).get("section") or "-"
        print(f"  {score:.4f} | {question}  →  {section[:38]}")
    return scores


async def main() -> None:
    print("向量库现有条目：", await run_sync(vector_store.count))
    inside = await measure(IN_MANUAL, "手册内问题（期望高分）")
    outside = await measure(OUT_OF_MANUAL, "手册外问题（期望低分）")

    print("\n===== 汇总 =====")
    print(f"手册内最高: {max(inside):.4f} | 最低: {min(inside):.4f} | 平均: {sum(inside)/len(inside):.4f}")
    print(f"手册外最高: {max(outside):.4f} | 最低: {min(outside):.4f} | 平均: {sum(outside)/len(outside):.4f}")

    suggest = (min(inside) + max(outside)) / 2
    print(f"\n建议阈值（取两组中间值）: {suggest:.2f}")
    print(f"当前 .env 阈值: {settings.similarity_threshold}")


if __name__ == "__main__":
    asyncio.run(main())
