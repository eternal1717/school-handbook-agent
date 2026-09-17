"""端到端验证：反馈闭环 + 链路追踪 + 引用溯源。

## 这个脚本验的是什么

前面几层（检索、重排、自省、生成）只要跑通一次就能肉眼确认，但
「数据有没有真的落库、下次读出来还对不对」是另一回事——那类问题
不会报错，只会静默地什么都不存。所以这里逐条对账：

 1. 提问能拿到 message_id / trace_id / 来源（含 doc_id、chunk_index）
 2. 点赞能写进去，再读回来还是「已赞」
 3. 历史消息带着思维链、trace_id、评价状态一起回来（刷新页面不丢状态）
 4. 满意度统计把这次点赞算进去了
 5. 点踩能进 badcase 列表，并导出成评测集格式
 6. 链路概览有各阶段耗时，能看出慢在哪一步
 7. **引用可验证**：来源里给的 chunk_index 能在知识库里定位到同一段原文
    ——「引用」如果不能回溯到原文，那就只是装饰。

用法（服务需先在 8010 跑起来）：
    python tests/verify_observability.py
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8010"
USER = "verify_user"

_passed = 0
_failed: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    global _passed
    if condition:
        _passed += 1
        print(f"  [PASS] {label}")
    else:
        _failed.append(label)
        print(f"  [FAIL] {label}" + (f"  —— {detail}" if detail else ""))


def _request(method: str, path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        f"{BASE}{path}", data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chat(question: str, conversation_id: str | None = None) -> dict:
    """跑一次完整问答，把 SSE 事件折叠成一份便于断言的结果。"""
    payload = {"question": question, "user_id": USER, "conversation_id": conversation_id}
    req = urllib.request.Request(
        f"{BASE}/api/chat", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    result: dict = {"types": [], "sources": [], "reasoning": "", "answer": "", "plan": None}
    with urllib.request.urlopen(req, timeout=240) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if not body:
                continue
            try:
                ev = json.loads(body)
            except json.JSONDecodeError:
                continue
            kind = ev.get("type")
            result["types"].append(kind)
            if kind == "done":
                result.update(ev)
            elif kind == "sources":
                result["sources"] = ev.get("sources") or []
            elif kind == "reasoning":
                result["reasoning"] += ev.get("content", "")
            elif kind == "token":
                result["answer"] += ev.get("content", "")
            elif kind == "plan":
                result["plan"] = ev
            elif kind == "error":
                result["error"] = ev.get("message")
    return result


def main() -> int:
    print("=" * 68)
    print("验证 1 · 提问 → 落库 → 来源可溯源")
    print("=" * 68)
    first = chat("考试作弊会受到什么处分？")
    if first.get("error"):
        print(f"  服务返回错误：{first['error']}")
        return 1

    check("产出 done 事件", "done" in first["types"], f"事件序列={first['types'][:8]}…")
    check("拿到 message_id", bool(first.get("message_id")), str(first.get("message_id")))
    check("拿到 trace_id", bool(first.get("trace_id")), str(first.get("trace_id")))
    check("拿到来源", len(first["sources"]) > 0, f"{len(first['sources'])} 条")
    check("回答非空", len(first["answer"]) > 10, f"{len(first['answer'])} 字")
    check("来源含 doc_id/chunk_index",
          all(s.get("doc_id") and s.get("chunk_index") is not None for s in first["sources"]),
          json.dumps(first["sources"][:1], ensure_ascii=False)[:160])

    # --- 引用溯源：来源给的 chunk_index 必须能定位到同一段原文 ---
    print("\n  -- 引用可验证性（点「定位原文」背后的逻辑）--")
    if first["sources"]:
        src = first["sources"][0]
        chunks = _request("GET", f"/api/knowledge/{src['doc_id']}/chunks")
        matched = [c for c in chunks["chunks"] if c["index"] == src["chunk_index"]]
        check("chunk_index 能在知识库定位", bool(matched),
              f"doc={src['doc_id'][:8]}… chunk={src['chunk_index']}")
        if matched:
            plain = lambda s: "".join((s or "").split())  # noqa: E731
            check("定位到的原文与来源一致",
                  plain(matched[0]["text"])[:200] == plain(src["text"])[:200],
                  f"库内={plain(matched[0]['text'])[:60]!r} / 来源={plain(src['text'])[:60]!r}")
        check("来源标注了命中通道",
              any(s.get("vector_rank") or s.get("bm25_rank") for s in first["sources"]))

    conv_id = first.get("conversation_id")
    msg_id = first.get("message_id")
    trace_id = first.get("trace_id")

    # --- 点赞 ---
    print("\n" + "=" * 68)
    print("验证 2 · 反馈写入 → 读回")
    print("=" * 68)
    up = _request("POST", "/api/feedback", {
        "message_id": msg_id, "rating": "up", "user_id": USER,
        "conversation_id": conv_id, "trace_id": trace_id,
        "question": "考试作弊会受到什么处分？",
    })
    check("点赞接口返回 ok", up.get("ok") is True and up.get("rating") == "up", json.dumps(up, ensure_ascii=False))

    history = _request("GET", f"/api/conversations/{conv_id}/messages?user_id={USER}")
    assistant_msgs = [m for m in history if m["role"] == "assistant"]
    check("历史里能读到这条回答", any(m["id"] == msg_id for m in assistant_msgs), f"共 {len(assistant_msgs)} 条回答")
    target = next((m for m in assistant_msgs if m["id"] == msg_id), None)
    if target:
        check("历史消息带 trace_id", target.get("trace_id") == trace_id,
              f"{target.get('trace_id')} vs {trace_id}")
        check("历史消息带思维链", len(target.get("reasoning") or "") > 0,
              f"{len(target.get('reasoning') or '')} 字")
        check("历史消息带评价状态", target.get("feedback") == "up", repr(target.get("feedback")))
        check("历史消息带来源", len(target.get("sources") or []) > 0,
              f"{len(target.get('sources') or [])} 条")

    summary = _request("GET", "/api/feedback/summary")
    check("满意度统计含本次点赞", summary.get("up", 0) >= 1, json.dumps(summary, ensure_ascii=False))

    # --- 点踩 → badcase ---
    print("\n" + "=" * 68)
    print("验证 3 · 点踩 → badcase → 评测集导出")
    print("=" * 68)
    second = chat("学校食堂几点开门营业？", conv_id)
    down = _request("POST", "/api/feedback", {
        "message_id": second.get("message_id"), "rating": "down", "user_id": USER,
        "conversation_id": conv_id, "trace_id": second.get("trace_id"),
        "question": "学校食堂几点开门营业？", "comment": "验证脚本自动点踩",
    })
    check("点踩接口返回 ok", down.get("ok") is True and down.get("rating") == "down")

    badcases = _request("GET", "/api/feedback/badcases?limit=20")
    ids = [item.get("message_id") for item in badcases.get("items", [])]
    check("点踩进入 badcase 列表", second.get("message_id") in ids, f"列表 message_id={ids[:8]}")

    exported = _request("GET", "/api/feedback/export?limit=20")
    check("能导出评测集", bool(exported.get("items") or exported.get("cases")),
          json.dumps(exported, ensure_ascii=False)[:200])
    check("导出条目含问题字段",
          any((item.get("question") or item.get("query")) for item in (exported.get("items") or exported.get("cases") or [])))

    # --- 链路追踪 ---
    print("\n" + "=" * 68)
    print("验证 4 · 链路追踪聚合")
    print("=" * 68)
    overview = _request("GET", "/api/trace/overview")
    check("概览有总问答数", overview.get("total", 0) >= 2, json.dumps(overview, ensure_ascii=False)[:260])
    stages = overview.get("stage_avg_ms") or {}
    check("概览有各阶段耗时", bool(stages),
          f"阶段={ {k: v for k, v in list(stages.items())[:6]} }")
    check("概览有延迟分位值",
          all(overview.get(key) is not None for key in ("avg_ms", "p50_ms", "p95_ms")),
          f"avg={overview.get('avg_ms')} p50={overview.get('p50_ms')} p95={overview.get('p95_ms')}")
    check("概览有拒答率", overview.get("refusal_rate") is not None,
          f"refusal_rate={overview.get('refusal_rate')}")

    traces = _request("GET", "/api/trace?limit=10")
    check("链路列表非空", len(traces) > 0, f"{len(traces)} 条")
    if traces:
        detail = _request("GET", f"/api/trace/{trace_id}")
        check("能取到单条链路详情", detail.get("id") == trace_id)
        check("详情含改写结果", bool(detail.get("rewritten")), (detail.get("rewritten") or "")[:60])
        check("详情含检索统计", bool(detail.get("retrieval")),
              json.dumps(detail.get("retrieval") or {}, ensure_ascii=False)[:160])
        check("详情含重排前后", bool(detail.get("rerank")),
              json.dumps(detail.get("rerank") or {}, ensure_ascii=False)[:160])
        check("详情含模型用量", bool(detail.get("usage")),
              json.dumps(detail.get("usage") or {}, ensure_ascii=False)[:160])
        check("详情含阶段耗时", bool(detail.get("stages")),
              f"{len(detail.get('stages') or [])} 段")

    # --- 结果 ---
    print("\n" + "=" * 68)
    if _failed:
        print(f"结果：{_passed} 项通过，{len(_failed)} 项失败")
        for name in _failed:
            print(f"  - {name}")
        return 1
    print(f"结果：全部 {_passed} 项通过")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.URLError as exc:
        print(f"连不上服务（{BASE}）：{exc}\n请先启动：python run.py")
        raise SystemExit(2) from None
