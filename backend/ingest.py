"""批量导入脚本：把一个或多个文件/目录灌进知识库。

用法（在 backend 目录下执行）：
    D:\\python\\python.exe ingest.py "D:\\某目录\\学生手册.docx"
    D:\\python\\python.exe ingest.py "D:\\某目录\\"          # 导入整个目录
"""
import asyncio
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(BACKEND_DIR)

from app.config import settings  # noqa: E402
from app.db.session import SessionLocal, init_db  # noqa: E402
from app.services import knowledge_service  # noqa: E402

SUPPORTED = {".docx", ".pdf", ".txt", ".md"}


def collect_files(targets: list[str]) -> list[Path]:
    files: list[Path] = []
    for target in targets:
        path = Path(target)
        if path.is_dir():
            files.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in SUPPORTED))
        elif path.is_file():
            files.append(path)
        else:
            print(f"[跳过] 路径不存在：{path}")
    return files


async def ingest_all(files: list[Path]) -> None:
    await init_db()
    async with SessionLocal() as db:
        for path in files:
            try:
                result = await knowledge_service.ingest_file(db, path, original_name=path.name)
                print(
                    f"[成功] {result['filename']}：{result['chunk_count']} 个知识块，"
                    f"{result['char_count']} 字，向量库共 {result['vector_total']} 条"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[失败] {path.name}：{type(exc).__name__}: {exc}")


def main() -> None:
    targets = sys.argv[1:]
    if not targets:
        print(__doc__)
        return

    files = collect_files(targets)
    if not files:
        print("没有找到可导入的文件（支持 docx / pdf / txt / md）")
        return

    if settings.use_mock_llm:
        print("[提示] 未配置 API Key，仅影响问答生成，不影响知识库导入。\n")

    asyncio.run(ingest_all(files))


if __name__ == "__main__":
    main()
