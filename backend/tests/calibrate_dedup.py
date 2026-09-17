"""语义去重阈值校准：看不同改写程度下，两个文本块的相似度到底是多少。

用途：DEDUP_SIMILARITY 这个阈值拍脑袋定很容易出错——
  定太低 → 把真正的新内容误判成重复，内容就丢了（危险）；
  定太高 → 换了说法的重复内容还是会被存进去（白配）。
跑这个脚本看真实数值，再决定阈值。
"""
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.config import settings  # noqa: E402
from app.services import embedding  # noqa: E402

LONG_A = (
    "学生请假应当履行审批手续。请假一天以内的，由辅导员批准；请假一天以上三天以内的，"
    "由辅导员签署意见后报所在学院学生工作办公室批准；请假三天以上的，需报学生工作处备案。"
    "请假学生应当在批准后按时返校，并及时办理销假手续。未经批准擅自离校的，按旷课处理。"
)
LONG_B = (
    "学生请假应当履行审批手续。请假一天以内的，由辅导员批准；请假一天以上三天以内的，"
    "由辅导员签署意见后报所在学院学生工作办公室批准；请假四天以上的，需报学生工作处备案。"
    "请假学生应当在批准后按时返校，并及时办理销假手续。未经批准擅自离校的，按旷课处理。"
)

SAMPLES: list[tuple[str, str, str]] = [
    ("完全相同（只差空格）", "学生应当按时到校上课。", "学生应当  按时到校上课。"),
    ("同义改写", "请假需要向辅导员提交书面申请。", "请假要向辅导员递交书面申请材料。"),
    ("条款只改一个数字", "请假三天以内由辅导员审批。", "请假五天以内由辅导员审批。"),
    ("同主题相邻条款", "请假需要向辅导员提交书面申请。", "请假三天以上需报学院审批。"),
    ("同章节不同条款", "考试作弊将记入学生档案。", "考试迟到十五分钟不得入场。"),
    ("无关内容", "宿舍熄灯时间为二十三点。", "图书馆早上八点开门。"),
    ("短标题", "第二章 宿舍管理", "第三章 考试管理"),
    ("长段落只改一个字", LONG_A, LONG_B),
]


def main() -> None:
    texts: list[str] = []
    for _, left, right in SAMPLES:
        texts.extend([left, right])

    vectors = embedding.encode_documents(texts)
    threshold = settings.dedup_similarity

    print(f"\n当前阈值 DEDUP_SIMILARITY = {threshold}\n")
    print(f"{'场景':<20} {'相似度':>8}   判定")
    print("-" * 52)
    for i, (name, _, _) in enumerate(SAMPLES):
        a, b = vectors[i * 2], vectors[i * 2 + 1]
        similarity = sum(x * y for x, y in zip(a, b))  # 已归一化，点积即余弦
        verdict = "判为重复（跳过）" if similarity >= threshold else "判为新内容（入库）"
        print(f"{name:<20} {similarity:>8.4f}   {verdict}")

    print(
        "\n提示：如果「条款只改一个数字」「长段落只改一个字」这类被判成重复，\n"
        "      说明阈值偏低——手册改版时的新内容会被丢掉，建议调高（如 0.99）\n"
        "      或把 DEDUP_SEMANTIC 关掉，只保留精确指纹去重。"
    )


if __name__ == "__main__":
    main()
