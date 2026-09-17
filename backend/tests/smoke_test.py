"""端到端自测脚本：把每个接口和每条核心逻辑都跑一遍。

用法（在 backend 目录下）：
    D:\\python\\python.exe tests\\smoke_test.py

覆盖的检查项：
    1. 健康检查与统计接口
    2. 知识库列表（手册是否已入库）
    3. 上传 / 删除文档（用临时 txt 文件，测试完自动清理）
    4. 手册内提问 → 应给出回答并带引用来源
    5. 手册外提问 → 应拒答（走相似度阈值逻辑）
    6. 短期记忆：同一会话多轮，历史消息是否落库
    7. 长期记忆：显式「记住」是否写入，且能被 /api/memory 读到
    8. 会话删除是否连带清理消息
    9. 前端页面与静态资源是否可访问
"""
import json
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(BACKEND_DIR)

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  [通过] {name} {detail}")
    else:
        FAILED.append(f"{name} {detail}")
        print(f"  [失败] {name} {detail}")


def read_sse(client: TestClient, question: str, conversation_id: str | None = None, user_id: str = "test_user") -> dict:
    """发一次提问，把 SSE 事件收集成结构化结果。"""
    events: list[dict] = []
    with client.stream(
        "POST",
        "/api/chat",
        json={"question": question, "user_id": user_id, "conversation_id": conversation_id},
        timeout=120.0,
    ) as response:
        for line in response.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            events.append(json.loads(line[5:].strip()))

    answer = "".join(e["content"] for e in events if e["type"] == "token")
    sources = next((e["sources"] for e in events if e["type"] == "sources"), [])
    meta = [e for e in events if e["type"] == "meta"]
    done = [e for e in events if e["type"] == "done"]
    errors = [e["message"] for e in events if e["type"] == "error"]
    return {
        "answer": answer,
        "sources": sources,
        "meta": meta,
        "errors": errors,
        "ended": any(e["type"] == "end" for e in events),
        "conversation_id": (meta[-1].get("conversation_id") if meta else None),
        # 检索质量指标随 done 事件回传（正常回答和拒答都有）。
        # 早先是挂在 meta 上的，但 meta 只带会话标识，放这里更合语义。
        "max_similarity": (done[-1].get("max_similarity") if done else None),
    }


def main() -> None:
    print("\n================ 开始端到端自测 ================")
    temp_file = BACKEND_DIR / "data" / "uploads" / "_smoke_test_doc.txt"
    conversation_id: str | None = None
    doc_id: str | None = None

    with TestClient(app) as client:
        # 1. 基础接口
        print("\n[1] 基础接口")
        health = client.get("/health")
        check("健康检查返回 200", health.status_code == 200, str(health.json()))
        stats = client.get("/api/stats").json()
        check("统计接口可用", "vectors" in stats and "llm_mode" in stats, str(stats))

        # 2. 知识库
        print("\n[2] 知识库")
        documents = client.get("/api/knowledge/list").json()
        handbook = [d for d in documents if "手册" in d["filename"]]
        # 用户可能同时入库了多份手册（比如完整版 + 简版），所以只要「至少有一份」即可，
        # 不能写死 == 1，否则用户多传一份手册测试就红了。
        check("学生手册已入库", len(handbook) >= 1, f"共 {len(documents)} 个文档，其中手册 {len(handbook)} 份")
        if handbook:
            check("手册知识块数量合理", handbook[0]["chunk_count"] > 100, f"{handbook[0]['chunk_count']} 块")

            # 知识块查看接口（前端「看分块」按钮用的）
            chunks_resp = client.get(f"/api/knowledge/{handbook[0]['id']}/chunks")
            check("知识块接口返回 200", chunks_resp.status_code == 200, str(chunks_resp.text)[:120])
            if chunks_resp.status_code == 200:
                payload = chunks_resp.json()
                indexes = [c["index"] for c in payload["chunks"]]
                check("块数与文档记录一致", payload["count"] == handbook[0]["chunk_count"],
                      f"{payload['count']} vs {handbook[0]['chunk_count']}")
                check("知识块按序号升序", indexes == sorted(indexes))
                check("每块都有正文和内容指纹", all(c["text"] and c["content_hash"] for c in payload["chunks"]))
            check("不存在的文档返回 404",
                  client.get("/api/knowledge/not-exist-id/chunks").status_code == 404)

        # 3. 上传与删除
        print("\n[3] 上传 / 删除文档")
        temp_file.write_text("测试学校规章文档。\n第七十七条 本测试条款用于验证上传与删除流程。", encoding="utf-8")
        with temp_file.open("rb") as handle:
            upload = client.post(
                "/api/knowledge/upload",
                files={"file": (temp_file.name, handle, "text/plain")},
            )
        check("上传接口返回 200", upload.status_code == 200, str(upload.text)[:160])
        if upload.status_code == 200:
            doc_id = upload.json()["id"]
            check("上传后知识块数 > 0", upload.json()["chunk_count"] > 0, str(upload.json()))

        # 4. 手册内提问
        print("\n[4] 手册内提问（应回答 + 带来源）")
        inside = read_sse(client, "学生在校期间享有哪些权利？", user_id="test_user")
        conversation_id = inside["conversation_id"]
        check("没有服务端错误", not inside["errors"], str(inside["errors"]))
        check("返回了回答内容", len(inside["answer"]) > 20, f"{len(inside['answer'])} 字")
        check("带引用来源", len(inside["sources"]) > 0, f"{len(inside['sources'])} 条来源")
        check("来源含章节信息", any(s.get("section") for s in inside["sources"]))
        check("相似度高于阈值", (inside["max_similarity"] or 0) >= 0.42, f"max={inside['max_similarity']}")
        check("流正常结束", inside["ended"])

        # 5. 手册外提问 → 拒答
        print("\n[5] 手册外提问（应拒答）")
        outside = read_sse(client, "红烧肉怎么做才好吃？", user_id="test_user")
        # 拒答有两条路径：检索不到 → 硬拒答文案；被判成越界提问 → 分流说明。
        # 两条都是「明确说自己答不了」，都该算通过。
        # 只认死一句话会在行为优化后误报失败（实测踩过：越界提问现在会被
        # 直接判成 out_of_scope 并给出更具体的说明，反而比原来那句通用文案更好）。
        outside_text = outside["answer"]
        check(
            "触发拒答（硬拒答或越界分流）",
            ("手册中未找到相关内容" in outside_text) or ("不在我的职责范围" in outside_text),
            outside_text[:60].replace("\n", " "),
        )
        check("拒答时不给来源", len(outside["sources"]) == 0)

        # 6. 短期记忆
        print("\n[6] 短期记忆（会话历史落库）")
        if conversation_id:
            follow = read_sse(client, "那申诉需要准备哪些材料？", conversation_id=conversation_id)
            check("复用同一会话", follow["conversation_id"] == conversation_id)
            history = client.get(f"/api/conversations/{conversation_id}/messages").json()
            roles = [m["role"] for m in history]
            check("历史消息已落库", len(history) >= 4, f"{len(history)} 条：{roles}")
            check("一问一答成对出现", roles.count("user") == roles.count("assistant"), str(roles))

        # 7. 长期记忆
        print("\n[7] 长期记忆（显式「记住」）")
        remembered = read_sse(
            client, "请记住我是2023级软件工程专业的学生，我在3班。", user_id="test_user"
        )
        check("未报错", not remembered["errors"], str(remembered["errors"]))
        memories = client.get("/api/memory?user_id=test_user").json()
        check("长期记忆已写入", len(memories) > 0, json.dumps(memories, ensure_ascii=False)[:160])
        other_user = client.get("/api/memory?user_id=another_user").json()
        check("记忆按用户隔离", len(other_user) == 0, f"另一个用户有 {len(other_user)} 条")

        users = client.get("/api/memory/users").json()
        check("用户列表接口可用", any(u["user_id"] == "test_user" for u in users),
              json.dumps(users, ensure_ascii=False)[:120])

        # 8. 会话删除
        print("\n[8] 删除会话")
        if conversation_id:
            deleted = client.delete(f"/api/conversations/{conversation_id}")
            check("删除会话返回 200", deleted.status_code == 200)
            remaining = client.get(f"/api/conversations/{conversation_id}/messages").json()
            check("消息已连带清理", len(remaining) == 0, f"剩余 {len(remaining)} 条")

        # 9. 文档删除 + 前端资源
        print("\n[9] 删除文档 / 前端资源")
        if doc_id:
            removed = client.delete(f"/api/knowledge/{doc_id}")
            check("删除文档返回 200", removed.status_code == 200)
            after = client.get("/api/knowledge/list").json()
            check("文档已从列表移除", all(d["id"] != doc_id for d in after), f"剩余 {len(after)} 个")

        index = client.get("/")
        check("首页可访问", index.status_code == 200 and 'id="app"' in index.text)
        for asset in ("styles.css", "app.js", "vendor/vue.global.prod.js", "vendor/element-plus.full.min.js"):
            response = client.get(f"/static/{asset}")
            check(f"静态资源可访问：{asset}", response.status_code == 200, f"{len(response.content)} 字节")

        # 10. 清理测试产生的数据（测试不该在我库里留垃圾）
        print("\n[10] 清理测试数据")
        for item in client.get("/api/memory?user_id=test_user").json():
            client.delete(f"/api/memory/{item['id']}")
        check("测试长期记忆已清理",
              len(client.get("/api/memory?user_id=test_user").json()) == 0)
        for item in client.get("/api/conversations?user_id=test_user").json():
            client.delete(f"/api/conversations/{item['id']}")

    # 清理临时文件
    if temp_file.exists():
        temp_file.unlink()

    print("\n================ 自测结果 ================")
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    for item in FAILED:
        print(f"  · {item}")
    print("=" * 42 + "\n")

    if FAILED:
        sys.exit(1)


if __name__ == "__main__":
    main()
