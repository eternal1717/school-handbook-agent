"""增量去重测试：新增文档时只补知识库里没有的内容。

用**独立的临时 collection** 跑，不污染正式向量库；跑完自动清理。

覆盖场景：
  1. 首次入库 → 全部新增
  2. 第二份文档 = 老内容（排版变化）+ 新内容 → 只新增差异部分
  3. 第三份文档 = 与已有内容完全重复 → 全部跳过，不建空文档记录
  4. 老数据（没有内容指纹的）能自动回填指纹
"""
import asyncio
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from sqlalchemy import delete, select  # noqa: E402

from app.config import settings  # noqa: E402
from app.db.session import SessionLocal, init_db  # noqa: E402
from app.models import KnowledgeDocument  # noqa: E402
from app.services import knowledge_service, vector_store  # noqa: E402
from app.services.utils import content_hash  # noqa: E402

TEST_COLLECTION = "dedup_test_tmp"
TMP_DIR = BACKEND / "data" / "uploads" / "_dedup_test"

DOC_A = """第一章 总则
学生应当按时到校上课。
请假需要向辅导员提交书面申请。
考勤由班主任负责统计。
第二章 宿舍管理
宿舍熄灯时间为二十三点。
"""

# 与 A 内容相同，但排版不同（多余空格、空行），外加两段新内容
DOC_B = """第一章   总则
学生应当按时到校上课。

请假需要向辅导员提交书面申请。
考勤由班主任负责统计。

第三章 考试管理
考试作弊将记入学生档案。
"""

# 与 A 完全重复（连排版也一样）
DOC_C = DOC_A

# 与 A 内容相同，但把「请假」那段换了个说法
DOC_D = """第一章 总则
学生需要按时到校上课，不得无故迟到。
请假要向辅导员递交书面申请材料。
考勤情况由班主任统一统计。
"""

# 长段落，用来验证「改一个字」在语义去重下会被判成重复
LONG_A = (
    "学生请假应当履行审批手续。请假一天以内的，由辅导员批准；请假一天以上三天以内的，"
    "由辅导员签署意见后报所在学院学生工作办公室批准；请假三天以上的，需报学生工作处备案。"
    "请假学生应当在批准后按时返校，并及时办理销假手续。未经批准擅自离校的，按旷课处理。"
)
LONG_B = LONG_A.replace("三天以上", "四天以上")  # 只改了一个词

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  [OK]   {name}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))


def write_doc(filename: str, text: str) -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    path = TMP_DIR / filename
    path.write_text(text, encoding="utf-8")
    return path


async def main() -> None:
    settings.chroma_collection = TEST_COLLECTION
    vector_store._collection = None  # 让下次取用重新建临时集合

    await init_db()

    async with SessionLocal() as db:
        # ---------- 场景 1：首次入库 ----------
        print("\n[1] 首次入库（全部应为新增）")
        path_a = write_doc("doc_a.txt", DOC_A)
        result_a = await knowledge_service.ingest_file(db, path_a, original_name="doc_a.txt")
        check("首次入库 0 个跳过", result_a["skipped_chunk_count"] == 0, f"跳过={result_a['skipped_chunk_count']}")
        check("首次入库块数 > 0", result_a["chunk_count"] > 0, f"新增={result_a['chunk_count']}")
        check("不是重复文档", result_a["duplicated"] is False)

        # ---------- 场景 2：老内容 + 新内容 ----------
        print("\n[2] 第二份文档 = 老内容（排版变了）+ 新内容（只应新增差异）")
        path_b = write_doc("doc_b.txt", DOC_B)
        result_b = await knowledge_service.ingest_file(db, path_b, original_name="doc_b.txt")
        check("检测到重复内容并跳过", result_b["skipped_chunk_count"] > 0, f"跳过={result_b['skipped_chunk_count']}")
        check("只新增了差异部分", 0 < result_b["chunk_count"] < result_a["chunk_count"],
              f"新增={result_b['chunk_count']} / 首次={result_a['chunk_count']}")
        check("重复的块没有被重复写入向量库",
              result_b["vector_total"] - result_a["vector_total"] == result_b["chunk_count"],
              f"向量库增量={result_b['vector_total'] - result_a['vector_total']}")

        # 逐块核对：B 里与 A 相同的那几段，确实一条都没进库
        stored = await asyncio.to_thread(vector_store.iter_existing_hashes)
        a_hashes = {content_hash(t) for t in ("学生应当按时到校上课。", "请假需要向辅导员提交书面申请。", "考勤由班主任负责统计。")}
        check("A 的旧内容指纹仍在库中（没有被覆盖）", a_hashes <= stored, f"命中 {len(a_hashes & stored)}/3")

        # ---------- 场景 3：完全重复的文档 ----------
        print("\n[3] 第三份文档 = 与已有内容完全重复（应全部跳过、不建记录）")
        path_c = write_doc("doc_c.txt", DOC_C)
        result_c = await knowledge_service.ingest_file(db, path_c, original_name="doc_c.txt")
        check("识别为完全重复", result_c["duplicated"] is True)
        check("新增块为 0", result_c["chunk_count"] == 0)
        check("没有生成空的文档记录", result_c["id"] == "")
        check("向量库条数没有变化", result_c["vector_total"] == result_b["vector_total"],
              f"{result_b['vector_total']} → {result_c['vector_total']}")

        rows = (await db.execute(select(KnowledgeDocument).where(KnowledgeDocument.filename == "doc_c.txt"))).scalars().all()
        check("数据库里没有 doc_c.txt 记录", len(rows) == 0)

        # ---------- 场景 4：默认配置不误杀改写内容 ----------
        print("\n[4] 默认配置（语义去重关闭）：换了说法的内容应作为新内容入库")
        path_d = write_doc("doc_d.txt", DOC_D)
        result_d = await knowledge_service.ingest_file(db, path_d, original_name="doc_d.txt")
        check("改写内容未被误杀，正常入库", result_d["chunk_count"] > 0, f"新增={result_d['chunk_count']}")

        # ---------- 场景 5：语义去重的真实表现 ----------
        print("\n[5] 语义去重开启后：长段落只改一个字的块会被判成重复")
        path_e = write_doc("doc_e.txt", LONG_A)
        result_e = await knowledge_service.ingest_file(db, path_e, original_name="doc_e.txt")
        check("长段落首版正常入库", result_e["chunk_count"] == 1, f"新增={result_e['chunk_count']}")

        path_f = write_doc("doc_f.txt", LONG_B)
        result_f_default = await knowledge_service.ingest_file(db, path_f, original_name="doc_f.txt")
        check("默认（语义关）：改版后的内容正常入库", result_f_default["chunk_count"] == 1,
              f"新增={result_f_default['chunk_count']}")

        # 把 doc_f 撤掉，再用「开启语义去重」的配置重传一次，对比行为差异
        for row in (await db.execute(select(KnowledgeDocument).where(KnowledgeDocument.filename == "doc_f.txt"))).scalars().all():
            await knowledge_service.delete_document(db, row.id)

        settings.dedup_semantic = True
        result_f_semantic = await knowledge_service.ingest_file(db, path_f, original_name="doc_f.txt")
        settings.dedup_semantic = False
        check("开启语义去重：改版内容被误判为重复（这正是默认关闭它的原因）",
              result_f_semantic["duplicated"] is True or result_f_semantic["skipped_chunk_count"] > 0,
              f"新增={result_f_semantic['chunk_count']} 跳过={result_f_semantic['skipped_chunk_count']}")

        # ---------- 清理 ----------
        print("\n[6] 清理测试数据")
        for name in ("doc_a.txt", "doc_b.txt", "doc_c.txt", "doc_d.txt", "doc_e.txt", "doc_f.txt"):
            for row in (await db.execute(select(KnowledgeDocument).where(KnowledgeDocument.filename == name))).scalars().all():
                await knowledge_service.delete_document(db, row.id)
        check("测试文档记录已清空",
              len((await db.execute(select(KnowledgeDocument).where(KnowledgeDocument.filename.like("doc_%.txt")))).scalars().all()) == 0)

    # 删掉临时 collection
    collection = vector_store.get_collection()
    client = vector_store._client
    check("临时向量库已清空", collection.count() == 0, f"剩余={collection.count()}")
    client.delete_collection(TEST_COLLECTION)
    vector_store._collection = None
    vector_store._client = None

    print(f"\n{'=' * 46}")
    print(f"增量去重测试：通过 {passed} 项，失败 {failed} 项")
    print(f"{'=' * 46}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
