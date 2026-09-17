"""RAG 编排层：规划 → 混合检索 → 重排 → 自省 → 生成。

## 一次问答里到底发生了什么

```
学生提问
   │
   ├─[0] 读短期记忆（本会话最近几轮）+ 长期记忆（跨会话要点）
   │
   ├─[1] 提问理解 ── 意图分流 / 指代消解 / 查询改写 / 复杂度判定
   │        └─ 闲聊、越界 → 不检索，直接回应
   │
   ├─[2] 检索
   │        ├─ simple  → 向量 + BM25 并行召回 → RRF 融合
   │        └─ complex → Agent 工具循环，模型自己决定查几次、怎么查
   │
   ├─[3] 重排 ── 用大模型对 20 条候选做 listwise 排序，取前 5
   │
   ├─[4] 注入扫描 ── 检索内容一律当「证据」而非「指令」
   │
   ├─[5] 自省循环 ── 资料够不够？不够就改写重查，最多 agent_max_hops 跳
   │        └─ 仍不够 → 明确拒答（拒答是特性，不是缺陷）
   │
   ├─[6] 生成 ── 带思维链流式输出，引用来源编号
   │
   └─[7] 落库 ── 回答 + 来源 + 思维链存库，全链路 trace 落库
```

## 复杂度路由为什么必要

不是每个问题都值得走全套。学生问「学生证怎么补办」，单跳检索就能答准，
硬塞进 Agent 循环要多花 3 倍延迟和成本，收益是零。
所以 `simple` 路径跳过工具循环、也跳过自省判定（只在检索明显偏弱时才补判）。

行业共识说得很直白：**先把静态 RAG 做好，只为「静态路径确实失败」的那类问题
（多跳、比较、模糊）才引入循环**。这就是 complex 分支存在的意义。

## 事件协议（SSE）

前端按事件类型分流渲染：
    meta / stage / plan / retrieval / rerank / reflection / guard /
    sources / reasoning / token / status / memory / done / refusal / error
`reasoning` 是思维链增量，`token` 是正文增量——分开是为了让前端能
把「思考过程」折叠起来、正文正常打字机输出。
"""
import re
import time
from collections.abc import AsyncIterator

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import KnowledgeDocument
from app.services import (
    ai_ops,
    guard,
    hybrid,
    llm,
    memory_service,
    tokenizer,
    trace_service,
    vector_store,
)
from app.services.utils import run_sync

# 去重指纹取正文前多少字。取太短会把「第七十七条」和「第七十八条」
# 这种同族条款误判成同一条；取太长又抓不住分块重叠造成的重复。
DEDUP_PREFIX_CHARS = 120

REFUSAL_TEXT = (
    "手册中未找到相关内容。\n\n"
    "我检索了知识库里已收录的规章制度，没有找到能支撑这个问题的条款。"
    "为避免给你错误信息，我不做推测性回答。\n\n"
    "可以试试：换一种说法再问一次；或者确认这份规定是否已经导入知识库。"
)

SYSTEM_PROMPT = """你是「知津」，本校校园规章的问答助手，服务对象是本校学生。

回答规则（必须严格遵守）：
1. 只依据下方资料回答，不允许编造，也不允许使用资料之外的知识。
2. 资料不足以回答时，直接回答「手册中未找到相关内容」，不要猜测、不要脑补。
3. 用简洁的中文回答，条理清晰；能对应到章节或条款时请注明出处。
4. 不要提及「参考资料」「检索」「资料」这类系统词汇，像熟悉校规的老师一样自然作答。
5. 如果资料里只覆盖了问题的某一部分，就只答那一部分，并明确说明其余部分在资料中没有提及。"""

CHITCHAT_SYSTEM = """你是「知津」，本校校园规章的问答助手。
学生这句话不是在问规章。用一两句话友好回应，然后自然地带一句「需要查校规随时问我」即可。
不要长篇大论，不要罗列能力清单，不要输出 markdown 标题。"""

OUT_OF_SCOPE_SYSTEM = """你是「知津」，本校校园规章的问答助手，职责范围是本校规章、学籍、奖惩、宿舍、教务流程这些问题。
学生问的内容超出了你的职责范围。用一两句话说明你只负责校园规章类问题，并举两个你确实能回答的例子。
不要尝试回答越界问题本身。"""


# ---------------- Agent 工具定义 ----------------

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_regulations",
        "description": (
            "在校园规章制度知识库中做混合检索（向量语义 + 关键词），返回最相关的条款原文。"
            "每当你不确定规章的具体内容、或需要补充证据时调用它。"
            "可以从不同角度多次调用，例如一次查「请假 审批 流程」，另一次查「请假 天数 上限」。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "检索用查询。要用规章文件里可能出现的措辞，"
                        "例如用「请假 审批 流程」而不是口语化的「请假要找谁」。"
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

LIST_DOCS_TOOL = {
    "type": "function",
    "function": {
        "name": "list_documents",
        "description": "列出知识库里已经收录了哪些文档。当你不确定某个主题的规定是否被收录时，先用它看一眼。",
        "parameters": {"type": "object", "properties": {}},
    },
}

AGENT_TOOLS = [SEARCH_TOOL, LIST_DOCS_TOOL]

AGENT_SYSTEM = """你是校园规章知识库的检索规划助手。

你的任务**不是**回答问题，而是收集足够回答学生问题的证据。

工作方式：
1. 先想清楚这个问题需要哪些规定支持。
2. 调用 search_regulations 检索。如果第一次结果不理想，换一种措辞再查一次——
   注意用规章制度里更可能出现的书面措辞。
3. 当你认为收集到的条款足以回答问题时，停止调用工具。

约束：
- 调用工具不要超过 3 次，避免过度检索把无关内容混进来。
- 如果查了两三次都没有相关内容，直接停止，不要反复尝试——知识库里没有就是没有，
  后面会由主流程明确告知学生找不到，这比强行拼凑答案更好。"""


# ---------------- 事件构造 ----------------

def _ev(event_type: str, **payload) -> dict:
    return {"type": event_type, **payload}


def _source_item(hit: dict) -> dict:
    """把内部命中的块转成前端要的来源卡片。"""
    return {
        "id": hit.get("id") or "",
        # doc_id + chunk_index 供前端「定位原文」用：点一下就能跳回知识库，
        # 看到答案到底是从哪一段原文来的。引用可验证，才谈得上可信。
        "doc_id": hit.get("doc_id") or "",
        "chunk_index": hit.get("chunk_index"),
        "filename": hit.get("filename") or "",
        "section": hit.get("section") or "",
        "page": hit.get("page"),
        "similarity": hit.get("vector_similarity"),
        "bm25_score": hit.get("bm25_score"),
        "rrf_score": hit.get("rrf_score"),
        "vector_rank": hit.get("vector_rank"),
        "bm25_rank": hit.get("bm25_rank"),
        "risk": hit.get("risk") or [],
        "text": hit.get("text") or "",
    }


# ---------------- 消息组装 ----------------

def build_qa_messages(
    question: str, hits: list[dict], history: list[dict], memory_text: str
) -> list[dict]:
    """组装「基于资料回答」的消息列表。

    顺序遵循「长上下文在前、问题在后」——把问题放在最后，
    模型对最近的 token 注意力更强，减少答非所问。
    """
    system = SYSTEM_PROMPT
    if settings.guard_enabled:
        system = f"{system}\n\n{guard.EVIDENCE_POLICY}"
    if memory_text:
        system = f"{system}\n\n{memory_text}"

    evidence = ai_ops.build_evidence(hits)

    messages: list[dict] = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({
        "role": "user",
        "content": (
            f"【资料】\n{guard.wrap_evidence(evidence)}\n\n"
            f"【学生的问题】\n{question}"
        ),
    })
    return messages


def build_direct_messages(
    question: str, history: list[dict], memory_text: str, intent: str
) -> list[dict]:
    """闲聊 / 越界时的消息列表：不带资料。"""
    system = OUT_OF_SCOPE_SYSTEM if intent == "out_of_scope" else CHITCHAT_SYSTEM
    if memory_text:
        system = f"{system}\n\n{memory_text}"
    messages: list[dict] = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})
    return messages


# ---------------- Agent 工具循环 ----------------

async def _run_agent(
    db: AsyncSession, question: str, history: list[dict], recorder: trace_service.TraceRecorder
) -> dict:
    """让模型自己调工具收集证据。返回 {hits, tool_calls, rounds, ok}。"""
    pool: dict[str, dict] = {}
    trace_log: list[dict] = []

    # 文档清单预先取好，工具被调用时直接返回，不查库
    result = await db.execute(select(KnowledgeDocument.filename))
    filenames = [row[0] for row in result.all()]

    async def executor(name: str, arguments: dict) -> str:
        if name == "search_regulations":
            query = str(arguments.get("query") or "").strip()
            if not query:
                return "查询为空，请给出具体的检索词。"
            found = await hybrid.retrieve_multi([query])
            hits = found["hits"]
            for hit in hits:
                pool.setdefault(hit["id"], hit)
            trace_log.append({"query": query, "returned": len(hits)})
            if not hits:
                return (f"没有检索到与「{query}」相关的条款。"
                        "这可能说明该主题未被收录，或措辞差异过大，你可以换一种说法再试一次。")
            lines = [f"「{query}」共命中 {len(hits)} 条，下面是前 {min(len(hits), 6)} 条："]
            for index, hit in enumerate(hits[:6], start=1):
                section = hit.get("section") or ""
                text = (hit.get("text") or "").replace("\n", " ")[:220]
                lines.append(f"[{index}] {section}\n{text}")
            return "\n\n".join(lines)

        if name == "list_documents":
            if not filenames:
                return "知识库目前是空的，还没有导入任何文档。"
            return "已收录的文档：\n" + "\n".join(f"- {name}" for name in filenames)

        return f"没有名为 {name} 的工具。"

    messages = [
        {"role": "system", "content": AGENT_SYSTEM},
        *history,
        {"role": "user", "content": f"学生的问题是：{question}\n\n请开始检索，收集回答这个问题所需的条款。"},
    ]

    outcome = await llm.run_tool_loop(
        messages, AGENT_TOOLS, executor, max_rounds=settings.agent_max_hops + 1
    )
    recorder.add_tool_calls(outcome.get("tool_calls") or [])
    recorder.stage("agent_loop", 0, rounds=outcome.get("rounds", 0),
                   queries=[item["query"] for item in trace_log])

    return {
        "hits": list(pool.values()),
        "rounds": outcome.get("rounds", 0),
        "ok": outcome.get("ok", False),
        "queries": [item["query"] for item in trace_log],
    }


# ---------------- 主流程 ----------------

def _max_similarity(hits: list[dict]) -> float:
    """候选里最高的向量相似度。

    只 BM25 命中的块 similarity 是 None——它的相关性由 BM25 分数表达，
    和余弦相似度不是一个量纲，所以这里跳过。返回 0 意味着「纯关键词命中」。
    """
    values = [hit.get("vector_similarity") for hit in hits if hit.get("vector_similarity") is not None]
    return max(values) if values else 0.0


def _dedup_key(hit: dict) -> str:
    """去重指纹 = 文档 id + 归一化正文前缀。

    为什么不直接用 chunk id 去重：分块是带重叠的，同一段条款的尾巴会被切进
    下一块。两块的 id 不同，正文前 100 多字却一模一样——学生看到的就是
    「来源里引了三条一模一样的第七十七条」。所以指纹必须落在正文上。

    归一化是必要的：「第七十七条」和「第 77 条」得指纹相同。
    """
    text = tokenizer.normalize(hit.get("text") or "")
    return f"{hit.get('doc_id') or ''}|{text[:DEDUP_PREFIX_CHARS]}"


def _prepare_candidates(pool: dict[str, dict] | list[dict]) -> list[dict]:
    """合并候选池 → 按融合分排序 → 去重 → 截断，得到送进重排的最终候选。

    两个动作缺一不可：

    1. **截断**。Agent 多轮检索会把池子撑到七八十条，全量送进重排实测要 1.9s，
       而且池子里越靠后的越不相关，纯粹是在烧 token 买噪声。
       截到 `fused_candidates`（默认 20）之后，重排又快又准。

    2. **去重**。多轮检索换着措辞查，命中的往往是同一段条款的不同分块。
       不去重的话，重排后的 5 条里可能有 3 条引用同一款，学生看到的
       「依据」就越看越虚——明明只找到一条。

    排序用 rrf_score 而不是相似度：RRF 只看排名不看分数，天然抹平了
    「向量余弦 0.8」和「BM25 分数 27」两个不可比的量纲。融合分高就意味着
    「两路都挤进了前排」——这比任何单路分数都更可靠。
    """
    items = list(pool.values()) if isinstance(pool, dict) else list(pool)
    ranked = sorted(items, key=lambda item: item.get("rrf_score") or 0.0, reverse=True)

    seen: set[str] = set()
    picked: list[dict] = []
    for hit in ranked:
        key = _dedup_key(hit)
        if key in seen:
            continue
        seen.add(key)
        picked.append(hit)
        if len(picked) >= settings.fused_candidates:
            break
    return picked


async def answer_stream(
    db: AsyncSession,
    question: str,
    user_id: str,
    conversation_id: str | None,
) -> AsyncIterator[dict]:
    """完整的问答流程，按事件产出结果。"""
    # ===== [0] 会话与记忆 =====
    conversation = await memory_service.get_or_create_conversation(
        db, conversation_id, user_id, question
    )
    yield _ev("meta", conversation_id=conversation.id, title=conversation.title)

    await memory_service.add_message(db, conversation.id, "user", question)
    history = await memory_service.load_history(db, conversation.id)
    history_without_current = history[:-1]  # 最后一条是刚存进去的本轮问题，不重复带

    memories = await memory_service.list_memories(db, user_id)
    memory_text = memory_service.format_memory_for_prompt(memories)

    recorder = trace_service.TraceRecorder(question, user_id, conversation.id)
    usage_sink = llm.start_usage_tracking()

    # ===== [1] 提问理解 =====
    yield _ev("stage", stage="understand", label="正在理解问题…")
    started = time.perf_counter()
    plan = await ai_ops.plan_query(question, history_without_current)
    recorder.set_plan(plan)
    recorder.measure("understand", started)

    yield _ev(
        "plan",
        intent=plan["intent"],
        complexity=plan["complexity"],
        # needs_retrieval 要发给前端：分流是「闲聊/越界不检索」这一步的产物，
        # 前端得能显示「此问题无需查规章」，评测脚本也要靠它断言分流是否正确。
        needs_retrieval=plan["needs_retrieval"],
        rewritten=plan["rewritten"],
        sub_queries=plan["sub_queries"],
        reason=plan.get("reason", ""),
        fallback=plan.get("_fallback", False),
    )

    sources: list[dict] = []

    # ===== 注入扫描（提问侧）=====
    # 攻击有两个来源：知识库里被塞进的内容，以及用户提问本身。
    # 检索侧的那次扫描在第 [5] 步（要等检回来才知道扫什么），这里先扫提问。
    # 提问侧不删改内容——问题就是问题，改了就答偏了——只做标记与留痕，
    # 真正让模型不被带跑的是 SYSTEM_PROMPT 的职责边界和 guard.EVIDENCE_POLICY。
    if settings.guard_enabled:
        question_risks = guard.scan(question)
        if question_risks:
            question_risk_items = [{
                "chunk_id": "",
                "filename": "(用户提问)",
                "categories": question_risks,
                "preview": question[:120],
            }]
            recorder.add_guard_risks(question_risk_items)
            yield _ev("guard", risks=question_risk_items)

    # ===== [2] 意图分流：闲聊 / 越界不检索 =====
    if not plan["needs_retrieval"]:
        recorder.stage("skip_retrieval", 0, intent=plan["intent"])
        messages = build_direct_messages(question, history_without_current, memory_text, plan["intent"])
    else:
        # ===== [3] 检索 =====
        pool: dict[str, dict] = {}
        retrieval_stats: dict = {}

        if settings.agent_enabled and plan["complexity"] == "complex":
            yield _ev("stage", stage="agent", label="问题较复杂，正在多轮检索…")
            started = time.perf_counter()
            agent_result = await _run_agent(db, plan["rewritten"], history_without_current, recorder)
            elapsed = recorder.measure("agent_retrieval", started,
                                       rounds=agent_result["rounds"], hits=len(agent_result["hits"]))
            for hit in agent_result["hits"]:
                pool.setdefault(hit["id"], hit)
            retrieval_stats = {
                "mode": "agent", "rounds": agent_result["rounds"],
                "ok": agent_result["ok"], "queries": agent_result["queries"],
                "pool_size": len(pool), "ms": elapsed,
            }
            # 工具循环没捞到东西时兜底：至少做一次常规混合检索，
            # 不能让 Agent 的规划失败变成学生收到「找不到」。
            if not pool:
                fallback = await hybrid.retrieve_multi([plan["rewritten"]])
                for hit in fallback["hits"]:
                    pool.setdefault(hit["id"], hit)
                retrieval_stats["fallback"] = "agent 未返回结果，已回退单跳混合检索"

        else:
            yield _ev("stage", stage="retrieval", label="正在检索知识库…")
            queries = [plan["rewritten"]] + plan["sub_queries"]
            started = time.perf_counter()
            found = await hybrid.retrieve_multi(queries)
            for hit in found["hits"]:
                pool.setdefault(hit["id"], hit)
            recorder.measure("retrieval", started,
                             queries=len(queries), pool=len(pool))
            retrieval_stats = {"mode": "single_hop", **found["stats"]}

        recorder.set_retrieval(retrieval_stats)
        yield _ev("retrieval", stats=retrieval_stats)

        # ===== [4] 重排 =====
        candidates = _prepare_candidates(pool)
        yield _ev("stage", stage="rerank", label="正在筛选最相关的条款…")
        started = time.perf_counter()
        reranked = await ai_ops.rerank(plan["rewritten"], candidates)
        recorder.measure("rerank", started, candidates=len(candidates),
                         result=len(reranked["hits"][:settings.rerank_top_n]))
        recorder.set_rerank(reranked["stats"], candidates, reranked["hits"][:settings.rerank_top_n])
        yield _ev("rerank", **reranked["stats"])

        top_hits = reranked["hits"][:settings.rerank_top_n]

        # ===== [5] 注入扫描 =====
        if settings.guard_enabled:
            top_hits, risks = guard.mark_hits(top_hits)
            if risks:
                recorder.add_guard_risks(risks)
                yield _ev("guard", risks=risks)

        # ===== [6] 自省循环 =====
        # 触发条件：问题本身复杂，或者检索结果明显偏弱（最高相似度低于阈值）。
        # 简单问题 + 强命中时跳过，省下 2 秒延迟和一次调用。
        max_similarity = _max_similarity(top_hits)
        weak_retrieval = max_similarity < settings.similarity_threshold
        need_reflection = settings.agent_enabled and (
            plan["complexity"] == "complex" or weak_retrieval
        )

        hops = 0
        while need_reflection and hops < settings.agent_max_hops:
            yield _ev("stage", stage="reflect", label="正在核查资料是否够用…")
            started = time.perf_counter()
            verdict = await ai_ops.judge_sufficiency(plan["rewritten"], top_hits)
            recorder.measure(f"reflect_{hops + 1}", started,
                             sufficient=verdict["sufficient"], skipped=verdict.get("skipped", False))
            yield _ev(
                "reflection", hop=hops + 1, sufficient=verdict["sufficient"],
                missing=verdict.get("missing", ""), rewritten=verdict.get("rewritten", ""),
            )

            if verdict["sufficient"] or not verdict["rewritten"]:
                break

            hops += 1
            yield _ev("stage", stage="retry", label=f"资料不足，换个问法重查（第 {hops} 次）…")
            started = time.perf_counter()
            extra = await hybrid.retrieve_multi([verdict["rewritten"]])
            added = 0
            for hit in extra["hits"]:
                if hit["id"] not in pool:
                    pool[hit["id"]] = hit
                    added += 1
            recorder.measure(f"retry_{hops}", started, added=added,
                             query=verdict["rewritten"][:80])

            if not added:
                break  # 换了说法也没有新东西，再查下去只是烧钱

            # 自省重查后池子又变大了，同样要先截断+去重——否则这一轮重排会成为
            # 整个请求里最慢的一步，还会把重复条款重新带进来源列表。
            candidates = _prepare_candidates(pool)
            reranked = await ai_ops.rerank(verdict["rewritten"], candidates)
            top_hits = reranked["hits"][:settings.rerank_top_n]
            if settings.guard_enabled:
                top_hits, risks = guard.mark_hits(top_hits)
                if risks:
                    recorder.add_guard_risks(risks)
            max_similarity = _max_similarity(top_hits)

        recorder.hops = hops

        # ===== [7] 拒答判定 =====
        # 两种情况拒答：一条都没检索到；或者检索明显偏弱。
        # 注意「偏弱」不再直接拒答——原来的实现用它一刀切，
        # 现在会先看资料实际够不够（weak_retrieval 触发了自省）。
        # 这样「向量没召回、但关键词精确命中」的条款不再被误杀。
        if not top_hits:
            sources = []
            yield _ev("sources", sources=[])
            yield _ev("refusal", reason="no_hits", max_similarity=round(max_similarity, 4))
            async for chunk in _stream_text(REFUSAL_TEXT):
                yield _ev("token", content=chunk)
            # 拒答也要落库、也要留 trace——「问了什么被拒了」是最该被复盘的信号，
            # 它直接告诉你知识库缺哪一块内容。
            async for event in _finish(db, conversation, REFUSAL_TEXT, [], "", recorder,
                                       usage_sink, refused=True):
                yield event
            return

        sources = [_source_item(hit) for hit in top_hits]
        yield _ev("sources", sources=sources)
        messages = build_qa_messages(question, top_hits, history_without_current, memory_text)

    # ===== [8] 生成（带思维链流式）=====
    answer_parts: list[str] = []
    reasoning_parts: list[str] = []

    yield _ev("stage", stage="generate", label="正在组织回答…")
    started = time.perf_counter()

    try:
        if settings.use_mock_llm:
            from app.services import llm as _llm
            text = _llm.mock_answer(question, sources)
            async for chunk in _stream_text(text):
                answer_parts.append(chunk)
                yield _ev("token", content=chunk)
        else:
            async for event in llm.stream_chat(messages, purpose="answer"):
                if event["type"] == "reasoning":
                    reasoning_parts.append(event["text"])
                    yield _ev("reasoning", content=event["text"])
                else:
                    answer_parts.append(event["text"])
                    yield _ev("token", content=event["text"])
    except Exception as exc:  # noqa: BLE001  模型调用失败要给出可读提示，不能让连接挂着
        error_text = (
            f"\n\n[调用大模型失败] {exc}\n"
            "排查提示：先确认网络能访问 api.deepseek.com；若报 Connection error，"
            "通常是进程继承了残留的代理环境变量（本项目默认已忽略 HTTP_PROXY/HTTPS_PROXY，"
            "除非 .env 里把 LLM_USE_ENV_PROXY 设成了 true）。"
        )
        answer_parts.append(error_text)
        yield _ev("token", content=error_text)

    recorder.measure("generate", started, chars=len("".join(answer_parts)))

    answer = "".join(answer_parts).strip()
    reasoning = "".join(reasoning_parts).strip()

    # 思维链把 max_tokens 额度吃光 → 正文一个字都没出来（复杂多跳问题最容易触发）。
    # 别急着甩一条「模型没返回正文」给用户：关掉思考、拿同一份资料再问一次，
    # 这一次整个额度都留给正文。多花一次调用，换一个真答案，值。
    if not answer and reasoning and not settings.use_mock_llm:
        yield _ev("status", message="思考过程占满了输出额度，已关闭思考重新作答…")
        heal_started = time.perf_counter()
        try:
            async for event in llm.stream_chat(messages, purpose="answer", thinking=False):
                # 这一轮已关掉思考，理论上只会来 content；真来了 reasoning 也忽略，
                # 否则前端会出现第二段思考过程，看起来像重复了。
                if event["type"] == "content":
                    answer_parts.append(event["text"])
                    yield _ev("token", content=event["text"])
        except Exception as exc:  # noqa: BLE001  自愈失败也要给出可读提示
            healing_error = f"{type(exc).__name__}: {exc}"
            yield _ev("status", message=f"关闭思考后重答也失败了：{healing_error}")
        answer = "".join(answer_parts).strip()
        recorder.measure("self_heal_no_thinking", heal_started,
                         ok=bool(answer), reasoning_chars=len(reasoning))

    # 模型既没报错、也一个字符没吐出来时，界面会是一片空白，让人以为程序坏了。
    # 走到这里说明连「关掉思考重答」都没能拿到正文，只能如实说明。
    if not answer and not settings.use_mock_llm:
        answer = (
            "[模型本次没有返回正文] 已成功检索到相关资料，但大模型只输出了思维链、没有输出正文。\n\n"
            "原因：deepseek-flash 是推理模型，思维链和正文**共享** max_tokens 额度。\n"
            "已自动尝试关闭思考重答一次但仍未成功。可以手动把 backend\\.env 里的 "
            "LLM_MAX_TOKENS 调大（例如 8192）后重启；或者把 LLM_THINKING 设为 off 来彻底关闭思维链。"
        )
        yield _ev("token", content=answer)

    # ===== [9] 长期记忆更新 =====
    rounds = await memory_service.count_user_rounds(db, conversation.id)
    need_explicit = "记住" in question
    need_summary = settings.summary_every_rounds > 0 and rounds % settings.summary_every_rounds == 0

    if need_explicit or need_summary:
        yield _ev("status", message="正在整理长期记忆…")
        saved: list[str] = []

        if need_explicit:
            value = await memory_service.process_explicit_memory(db, user_id, question)
            if value:
                saved.append(value)

        if need_summary:
            summary = await llm.summarize(
                await memory_service.load_history(db, conversation.id, rounds=settings.summary_every_rounds)
            )
            if summary:
                await memory_service.save_memory(db, user_id, f"summary_{conversation.id[:8]}_{rounds}", summary)
                saved.append(f"本轮话题摘要：{summary}")

        if saved:
            yield _ev("memory", items=saved)

    async for event in _finish(db, conversation, answer, sources, reasoning, recorder,
                               usage_sink, refused=False):
        yield event


async def _finish(
    db: AsyncSession,
    conversation,
    answer: str,
    sources: list[dict],
    reasoning: str,
    recorder: trace_service.TraceRecorder,
    usage_sink: list,
    refused: bool,
) -> AsyncIterator[dict]:
    """收尾：落库消息 → 落库 trace → 发 done 事件。

    trace 放在最后落库，是因为它要关联 assistant 的 message_id——
    而 message_id 只有消息写进库之后才有。前端在收到 done 时拿到 trace_id，
    之后点赞点踩带上它，就能把「用户觉得不好」和「链路里哪一环出问题」对上。
    """
    # 回答、思维链、trace 关联一起落库。
    # 思维链单独一列，回看历史时前端仍能把「思考过程」折叠出来。
    message = await memory_service.add_message(
        db, conversation.id, "assistant", answer, sources,
        reasoning=reasoning, trace_id=recorder.trace_id,
    )

    recorder.source_count = len(sources)
    recorder.answer_chars = len(answer)
    recorder.refused = refused
    recorder.set_usage(_summarize_usage(usage_sink))

    trace_id = await trace_service.save(db, recorder, message_id=message.id)

    yield _ev(
        "done",
        conversation_id=conversation.id,
        message_id=message.id,
        trace_id=trace_id,
        source_count=len(sources),
        refused=refused,
        total_ms=recorder.elapsed_ms(),
    )


def _summarize_usage(usage_sink: list) -> dict:
    """把多次调用的用量汇总成一份。"""
    if not usage_sink:
        return {}
    return {
        "calls": len(usage_sink),
        "prompt_tokens": sum(item["prompt_tokens"] for item in usage_sink),
        "completion_tokens": sum(item["completion_tokens"] for item in usage_sink),
        "reasoning_tokens": sum(item["reasoning_tokens"] for item in usage_sink),
        "total_tokens": sum(item["total_tokens"] for item in usage_sink),
        "detail": usage_sink,
    }


async def _stream_text(text: str, size: int = 24) -> AsyncIterator[str]:
    """按固定长度切片，让演示模式和拒答文本也有打字机效果。"""
    for start in range(0, len(text), size):
        yield text[start : start + size]
