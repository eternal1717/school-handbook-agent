"""启动入口：在 PyCharm 里直接右键 Run 这个文件即可。

它会做两件事：
1. 把工作目录切到 backend/，保证 data/ 等相对路径始终正确；
2. 用 uvicorn 启动 FastAPI 应用。
"""
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(BACKEND_DIR)  # 关键：不管 PyCharm 的工作目录设成什么，都统一到 backend/

import uvicorn  # noqa: E402

from app.config import settings  # noqa: E402


def main() -> None:
    if settings.use_mock_llm:
        print("[提示] 当前没有配置 DEEPSEEK_API_KEY，将使用演示模式（不做真实大模型调用）。")
        print("[提示] 在 backend/.env 里填入 Key 后重启即可获得完整回答能力。\n")

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
