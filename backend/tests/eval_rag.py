"""评测脚本：分流 / 检索 / 答案 / 拒答 / 延迟，以及检索层的消融实验。

## 两种跑法

    # 1) 纯检索消融：不启动服务、不调大模型，秒级出结果。
    #    对比「只有向量」和「向量+BM25 混合」，量化混合检索到底带来多少提升。
    python tests/eval_rag.py --retrieval-only

    # 2) 端到端评测：需要服务在 8010 跑着（会真实调用大模型，比较慢）。
    python tests/eval_rag.py
    python tests/eval_rag.py --save result_full.json     # 存下来，便于和基线对比
    python tests/eval_rag.py --baseline result_old.json  # 与历史结果对比

## 为什么要做消融

「我用了混合检索」这句话在面试里不值钱，值钱的是「混合检索把 Recall@5 从
X% 提到 Y%，因为学生手册里全是条款号和部门名，纯向量对这类精确标识符很差」。
消融实验就是把这句话变成数据的那一步——不跑消融，你就只能靠感觉讲。

## 评测集从哪来

`tests/eval_dataset.json`，人工逐条核对过知识库原文。拒答类用例都事先确认
过主题在知识库中出现 0 次，所以「拒答」是这些题唯一正确的答案。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND))

DATASET = HERE / "eval_dataset.json"
BASE = "http://127.0.0.1:8010"
USER = "eval_user"


# ---------------------------------------------------------------- 显示工具

def _width(text: str) -> int:
    """中日韩字符按 2 列宽计算，否则表格会歪。"""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)


def _pad(text: str, size: int, align: str = "left") -> str:
    space = " " * max(0, size - _width(text))
    return text + space if align == "left" else space + text


def _rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ---------------------------------------------------------------- 判定逻辑

def gold_hit(texts: list[str], gold_sets: list[list[str]]) -> bool:
    """某块正文同时含某组全部关键词，即算命中标准答案。"""
    for text in texts:
        flat = "".join(text.split())
        for group in gold_sets:
            if all("".join(kw.split()) in flat for kw in group):
                return True
    return False


def answer_coverage(answer: str, groups: list[list[str]]) -> tuple[int, int]:
    """返回 (命中的要点组数, 总要点组数)。"""
    if not groups:
        return 0, 0
    flat = "".join(answer.split())
    hit = sum(1 for group in groups if any("".join(v.split()) in flat for v in group))
    return hit, len(groups)


# 系统有两条拒答路径，评测时都要认，否则会把「礼貌地说答不了」误判成错误：
#   1. 硬拒答：一条都没检索到，主流程直接发 refusal 事件，压根不调大模型；
#   2. 软拒答：检索到了一些相关条款但没有一条能支撑答案，模型按系统提示
#      回答「手册中未找到相关内容」——`refused` 仍是 false，但语义上就是拒答。
REFUSAL_MARKERS = (
    "未找到", "没有找到", "未收录", "没有收录", "无法回答", "不能回答",
    "没有相关", "未提及", "没有提及", "无相关", "不在本", "超出",
    "只负责", "仅负责", "只回答",
)


def soft_refused(answer: str) -> bool:
    return any(marker in answer for marker in REFUSAL_MARKERS)


# ---------------------------------------------------------------- 检索消融

def retrieval_ablation(dataset: dict) -> dict:
    """在进程内直接跑三种检索配置，不调大模型。

    三种配置：
      vector  —— 只用向量（原项目做法）
      bm25    —— 只用关键词
      hybrid  —— 两路 RRF 融合（改造后）
    指标：Recall@5（前 5 条里有没有标准答案）、MRR@5（标准答案排在第几位的倒数）。
    """
    from app.config import settings
    from app.services import embedding as embedding_service
    from app.services import hybrid, vector_store

    cases = [c for c in dataset["cases"] if c["expect"].get("gold_keyword_sets")]
    if not cases:
        print("评测集里没有带 gold_keyword_sets 的用例，无法做检索消融。")
        return {}

    bm25_index = hybrid.get_bm25()
    print(f"被测用例 {len(cases)} 条；BM25 索引覆盖 {len(bm25_index)} 块")

    configs = ("vector", "bm25", "hybrid")
    stats = {name: {"recall5": 0, "recall20": 0, "mrr": 0.0, "miss": []} for name in configs}

    for case in cases:
        question = case["question"]
        gold = case["expect"]["gold_keyword_sets"]
        embedding = embedding_service.encode_query(question)

        ranking: dict[str, list[dict]] = {}
        vector_hits = vector_store.query(embedding, settings.vector_candidates)
        bm25_hits = bm25_index.search(question, settings.bm25_candidates)
        ranking["vector"] = vector_hits
        ranking["bm25"] = bm25_hits

        fused = hybrid.rrf_fuse([[h["id"] for h in vector_hits], [h["id"] for h in bm25_hits]])
        pool = {h["id"]: h for h in vector_hits}
        for hit in bm25_hits:
            pool.setdefault(hit["id"], hit)
        ranking["hybrid"] = sorted(pool.values(), key=lambda h: fused.get(h["id"], 0.0), reverse=True)

        for name in configs:
            hits = ranking[name]
            texts = [h.get("text") or "" for h in hits]
            top5 = hits[:5]
            if gold_hit([h.get("text") or "" for h in top5], gold):
                stats[name]["recall5"] += 1
            else:
                stats[name]["miss"].append(case["id"])
            if gold_hit([h.get("text") or "" for h in hits[:20]], gold):
                stats[name]["recall20"] += 1
            for rank, hit in enumerate(hits[:5], start=1):
                if gold_hit([hit.get("text") or ""], gold):
                    stats[name]["mrr"] += 1.0 / rank
                    break

    total = len(cases)
    result = {}
    for name in configs:
        item = stats[name]
        result[name] = {
            "recall5": item["recall5"] / total,
            "recall20": item["recall20"] / total,
            "mrr": item["mrr"] / total,
            "miss": item["miss"],
        }

    _rule("检索消融：三种配置对比（不调大模型）")
    labels = {"vector": "只用向量（原方案）", "bm25": "只用关键词 BM25", "hybrid": "混合检索 RRF（现方案）"}
    print(f"{_pad('配置', 26)}{_pad('Recall@5', 12)}{_pad('Recall@20', 12)}{_pad('MRR@5', 10)}")
    print("-" * 72)
    for name in configs:
        item = result[name]
        r5 = f"{item['recall5'] * 100:.1f}%"
        r20 = f"{item['recall20'] * 100:.1f}%"
        mrr = f"{item['mrr']:.3f}"
        print(f"{_pad(labels[name], 26)}{_pad(r5, 12)}{_pad(r20, 12)}{_pad(mrr, 10)}")

    base_r5 = result["vector"]["recall5"]
    hyb_r5 = result["hybrid"]["recall5"]
    print("-" * 72)
    delta = (hyb_r5 - base_r5) * 100
    print(f"混合检索相对纯向量：Recall@5 {base_r5*100:.1f}% → {hyb_r5*100:.1f}%"
          f"（{'+' if delta >= 0 else ''}{delta:.1f} 个百分点）")
    print(f"纯向量漏掉的用例：{result['vector']['miss'] or '无'}")
    print(f"混合检索漏掉的用例：{result['hybrid']['miss'] or '无'}")

    print(
        "\n读这份数据要注意一个口径问题：\n"
        "  消融是把**原始问题**直接丢给检索层，绕过了线上跑的那一步「查询改写」。\n"
        "  所以 coref 类的漏检（「那它要求必须在几天内提出？」这类）不代表线上也会漏——\n"
        "  线上会先把它改写成「申诉 提出 期限 10 个工作日」再检索。\n"
        "  换句话说：查询改写和混合检索是互补的两件事，消融只度量了后者。\n"
        "  真正要下结论的是端到端那份 Recall@5。"
    )
    return result


# ---------------------------------------------------------------- 端到端评测

def _chat(question: str, conversation_id: str | None = None) -> dict:
    payload = {"question": question, "user_id": USER, "conversation_id": conversation_id}
    req = urllib.request.Request(
        f"{BASE}/api/chat", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    out: dict = {"sources": [], "answer": "", "reasoning": "", "plan": {}, "refused": False,
                 "types": [], "conversation_id": conversation_id}
    with urllib.request.urlopen(req, timeout=300) as resp:
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
            out["types"].append(kind)
            if kind == "plan":
                out["plan"] = ev
            elif kind == "sources":
                out["sources"] = ev.get("sources") or []
            elif kind == "token":
                out["answer"] += ev.get("content", "")
            elif kind == "reasoning":
                out["reasoning"] += ev.get("content", "")
            elif kind == "refusal":
                out["refused"] = True
            elif kind == "done":
                out["conversation_id"] = ev.get("conversation_id") or conversation_id
                out["message_id"] = ev.get("message_id")
                out["trace_id"] = ev.get("trace_id")
                out["total_ms"] = ev.get("total_ms")
            elif kind == "error":
                out["error"] = ev.get("message")
    return out


def end_to_end(dataset: dict) -> dict:
    cases = dataset["cases"]
    records: list[dict] = []

    for index, case in enumerate(cases, start=1):
        expect = case["expect"]
        print(f"[{index:>2}/{len(cases)}] {case['id']:<7} {case['question'][:40]}", end="", flush=True)

        conv = None
        # 多轮用例：先把前几轮真实问一遍，让系统自己积累出上下文
        for msg in case.get("history") or []:
            if msg.get("role") == "user":
                seeded = _chat(msg["content"], conv)
                conv = seeded.get("conversation_id") or conv

        try:
            result = _chat(case["question"], conv)
        except urllib.error.URLError as exc:
            print(f"  [连接失败] {exc}")
            raise SystemExit(2) from None

        if result.get("error"):
            print(f"  [服务错误] {result['error']}")

        record = {
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "answer": result["answer"],
            "reasoning_chars": len(result["reasoning"]),
            "sources": len(result["sources"]),
            "plan": {k: result["plan"].get(k) for k in ("intent", "complexity", "needs_retrieval", "rewritten")},
            "refused": result["refused"],
            "total_ms": result.get("total_ms") or 0,
        }

        # 1) 分流是否对
        want_retrieval = expect.get("needs_retrieval")
        got_retrieval = result["plan"].get("needs_retrieval")
        record["routing_ok"] = (want_retrieval is None) or (bool(want_retrieval) == bool(got_retrieval))
        if expect.get("intent"):
            record["intent_ok"] = result["plan"].get("intent") == expect["intent"]

        # 2) 标准答案有没有被检索到（前 5 条来源里）
        gold = expect.get("gold_keyword_sets") or []
        if gold:
            record["recall_ok"] = gold_hit([s.get("text") or "" for s in result["sources"][:5]], gold)
        else:
            record["recall_ok"] = None

        # 3) 答案要点覆盖
        groups = expect.get("answer_groups") or []
        hit, total = answer_coverage(result["answer"], groups)
        record["answer_hit"], record["answer_total"] = hit, total

        # 4) 该不该拒答。答不了的问题上，「硬拒答」和「软拒答」都算答对；
        #    答得了的问题上，只要没被硬拒就算没误伤。
        record["soft_refused"] = soft_refused(result["answer"])
        record["expect_answerable"] = expect.get("answerable")
        if expect.get("answerable") is False:
            record["refusal_ok"] = result["refused"] or record["soft_refused"]
        elif expect.get("answerable") is True:
            record["refusal_ok"] = not result["refused"]
        else:
            record["refusal_ok"] = None

        # 5) 不该出现的内容
        banned = expect.get("must_not_contain") or []
        record["banned_ok"] = all(b not in result["answer"] for b in banned) if banned else None

        # 6) 长度上限
        limit = expect.get("max_answer_chars")
        record["length_ok"] = (len(result["answer"]) <= limit) if limit else None

        flags = []
        if record["routing_ok"] is False:
            flags.append(f"分流错(期望检索={want_retrieval}，实际={got_retrieval}，"
                         f"intent={record['plan'].get('intent')})")
        if record.get("intent_ok") is False:
            flags.append(f"意图错(期望={expect.get('intent')}，实际={record['plan'].get('intent')})")
        if record["recall_ok"] is False:
            flags.append("未召回原条款")
        if record["refusal_ok"] is False:
            flags.append(f"拒答判错(refused={record['refused']}，期望={expect.get('answerable')})")
        if record["banned_ok"] is False:
            flags.append("输出了不该输出的内容")
        if record["length_ok"] is False:
            flags.append(f"超长({len(result['answer'])}字)")
        print("  " + ("；".join(flags) if flags else "OK"))

        records.append(record)

    return _summarize(records)


def _rate(items: list[bool | None]) -> tuple[int, int, float]:
    valid = [x for x in items if x is not None]
    if not valid:
        return 0, 0, 0.0
    passed = sum(1 for x in valid if x)
    return passed, len(valid), passed / len(valid)


def _summarize(records: list[dict]) -> dict:
    _rule("端到端评测结果")

    routing = _rate([r["routing_ok"] for r in records])
    recall = _rate([r["recall_ok"] for r in records])
    refusal = _rate([r["refusal_ok"] for r in records])
    banned = _rate([r["banned_ok"] for r in records])
    length = _rate([r["length_ok"] for r in records])
    intent = _rate([r.get("intent_ok") for r in records])

    answer_hit = sum(r["answer_hit"] for r in records)
    answer_total = sum(r["answer_total"] for r in records)
    full_ok = sum(1 for r in records if r["answer_total"] and r["answer_hit"] == r["answer_total"])
    full_total = sum(1 for r in records if r["answer_total"])

    rows = [
        ("分流准确率", routing, "闲聊/越界不该检索，规章问题必须检索"),
        ("意图分类准确率", intent, "regulation / procedure / chitchat / out_of_scope"),
        ("检索召回率 Recall@5", recall, "标准答案条款出现在前 5 条来源里"),
        ("拒答判定准确率", refusal, "该拒的拒了，不该拒的没拒"),
        ("安全约束通过率", banned, "不输出系统提示等禁止内容"),
        ("回答长度合规率", length, "闲聊/越界不该长篇大论"),
    ]
    print(f"{_pad('指标', 24)}{_pad('通过', 12)}{_pad('占比', 10)}说明")
    print("-" * 72)
    for label, (passed, total, ratio), note in rows:
        print(f"{_pad(label, 24)}{_pad(f'{passed}/{total}', 12)}{_pad(f'{ratio*100:.1f}%', 10)}{note}")

    print()
    # 只统计「本该拒答」的那批用例走的是哪条路径。
    # 不能把全场的 soft_refused 都算进来——回答里出现「其余部分资料中未提及」
    # 也会命中拒绝措辞，但那是系统提示要求的“只答资料覆盖到的部分”，
    # 是诚实作答，不是拒答。
    doomed = [r for r in records if r["expect_answerable"] is False]
    hard = sum(1 for r in doomed if r["refused"])
    soft = sum(1 for r in doomed if r.get("soft_refused") and not r["refused"])
    print(f"该拒答的 {len(doomed)} 条用例中：硬拒答（未检索到，不发模型）{hard} 条；"
          f"软拒答（检索到但撑不起答案）{soft} 条")

    if answer_total:
        print(f"答案要点覆盖：{answer_hit}/{answer_total} = {answer_hit/answer_total*100:.1f}%；"
              f"整题要点全中 {full_ok}/{full_total} = {full_ok/max(1, full_total)*100:.1f}%")

    durations = sorted(r["total_ms"] for r in records if r["total_ms"])
    if durations:
        def pct(values: list[int], ratio: float) -> int:
            return values[min(len(values) - 1, int(len(values) * ratio))]

        print(f"端到端延迟：平均 {int(statistics.mean(durations))} ms / "
              f"P50 {pct(durations, 0.5)} ms / P95 {pct(durations, 0.95)} ms / "
              f"最大 {durations[-1]} ms")

    # 分类别统计
    _rule("分类别明细")
    categories: dict[str, list[dict]] = {}
    for record in records:
        categories.setdefault(record["category"], []).append(record)
    print(f"{_pad('类别', 16)}{_pad('条数', 8)}{_pad('分流', 10)}{_pad('召回@5', 10)}"
          f"{_pad('拒答', 10)}{_pad('要点覆盖', 12)}平均耗时")
    print("-" * 72)
    for name, items in sorted(categories.items()):
        r = _rate([x["routing_ok"] for x in items])
        rc = _rate([x["recall_ok"] for x in items])
        rf = _rate([x["refusal_ok"] for x in items])
        hit = sum(x["answer_hit"] for x in items)
        tot = sum(x["answer_total"] for x in items)
        avg_ms = int(statistics.mean([x["total_ms"] for x in items])) if items else 0
        print(f"{_pad(name, 16)}{_pad(str(len(items)), 8)}"
              f"{_pad(f'{r[0]}/{r[1]}', 10)}"
              f"{_pad(f'{rc[0]}/{rc[1]}' if rc[1] else '—', 10)}"
              f"{_pad(f'{rf[0]}/{rf[1]}' if rf[1] else '—', 10)}"
              f"{_pad(f'{hit}/{tot}' if tot else '—', 12)}{avg_ms} ms")

    # 失败清单
    failed = [r for r in records if False in (r["routing_ok"], r.get("intent_ok"), r["recall_ok"],
                                              r["refusal_ok"], r["banned_ok"], r["length_ok"])]
    if failed:
        _rule("需要人工看一眼的用例")
        for r in failed:
            print(f"- {r['id']}（{r['category']}）{r['question']}")
            print(f"    intent={r['plan'].get('intent')} needs_retrieval={r['plan'].get('needs_retrieval')} "
                  f"refused={r['refused']} 软拒答={r.get('soft_refused')} 来源={r['sources']}")
            print(f"    回答：{r['answer'][:160]}")

    return {
        "summary": {
            "routing": routing[2], "intent": intent[2], "recall5": recall[2],
            "refusal": refusal[2], "banned": banned[2], "length": length[2],
            "answer_coverage": answer_hit / answer_total if answer_total else 0.0,
            "avg_ms": int(statistics.mean(durations)) if durations else 0,
        },
        "records": records,
    }


# ---------------------------------------------------------------- 对比

def compare(baseline_path: str, current: dict) -> None:
    old = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    old_sum, new_sum = old.get("summary", {}), current.get("summary", {})
    _rule(f"与基线对比（{baseline_path}）")
    print(f"{_pad('指标', 24)}{_pad('基线', 12)}{_pad('本次', 12)}变化")
    print("-" * 72)
    for key, label in [("recall5", "Recall@5"), ("routing", "分流准确率"),
                       ("refusal", "拒答准确率"), ("answer_coverage", "答案要点覆盖")]:
        before, after = old_sum.get(key, 0), new_sum.get(key, 0)
        print(f"{_pad(label, 24)}{_pad(f'{before*100:.1f}%', 12)}{_pad(f'{after*100:.1f}%', 12)}"
              f"{(after-before)*100:+.1f} 个百分点")
    before_ms, after_ms = old_sum.get("avg_ms", 0), new_sum.get("avg_ms", 0)
    print(f"{_pad('平均耗时', 24)}{_pad(f'{before_ms} ms', 12)}{_pad(f'{after_ms} ms', 12)}"
          f"{after_ms - before_ms:+d} ms")


def main() -> int:
    parser = argparse.ArgumentParser(description="知津 · 评测脚本")
    parser.add_argument("--retrieval-only", action="store_true", help="只跑检索消融，不启动服务")
    parser.add_argument("--save", metavar="FILE", help="把本次结果存成 JSON")
    parser.add_argument("--baseline", metavar="FILE", help="与历史结果对比")
    parser.add_argument("--only", metavar="CATEGORY", help="只跑某一类用例")
    args = parser.parse_args()

    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    if args.only:
        dataset = {**dataset, "cases": [c for c in dataset["cases"] if c["category"] == args.only]}

    if args.retrieval_only:
        retrieval_ablation(dataset)
        return 0

    print(f"评测集：{len(dataset['cases'])} 条（{DATASET.name}）")
    print(f"目标服务：{BASE}")
    try:
        result = end_to_end(dataset)
    except urllib.error.URLError:
        print(f"\n连不上服务。请先启动：cd backend && python run.py")
        return 2

    if args.baseline:
        compare(args.baseline, result)
    if args.save:
        Path(args.save).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已保存：{args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
