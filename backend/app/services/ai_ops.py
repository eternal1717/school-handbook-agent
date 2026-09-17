"""查询规划、结果重排、资料自省——Agentic RAG 的三个决策点。

这三个模块都走「thinking 关闭」的快通道（实测快 4 倍、省 3.6 倍 token），
因为它们只需要输出一小段 JSON，不需要深度推理。

## 三者各自解决什么问题

**plan_query（提问理解）**
   把学生的大白话变成能检索的形式。核心是**指代消解**：
   学生问「那它要几天？」——「它」指的是上一轮的「请假」。
   不做这一步，这句话会被原样丢进检索，几乎必然拒答。
   这是 RAG 最常见、也最影响体感的失败模式。
   顺带判断意图（闲聊 / 越界 / 问规章）和复杂度（单跳够不够）。

**rerank（重排）**
   向量 + BM25 融合后拿到 20 条候选，但真正能进 prompt 的只有 5 条。
   融合分是「统计相关性」，模型看一遍全部候选再排是「语义相关性」，
   后者明显更准。行业标准做法就是召回一大把、精挑一小撮。

**judge_sufficiency（自省）**
   检索完先自问一句「这些资料够回答问题吗」。不够就改写重查，
   查满上限还不够就**明确拒答**。
   这是 Agentic RAG 和传统 RAG 的分水岭：传统 RAG 拿到什么就用什么，
   答不出来也硬答；自省循环给了系统「意识到自己没找到」的能力。

⚠️ 循环必须有硬上限（agent_max_hops）。没有上限的自省是最典型的翻车方式：
遇到语料里根本没有的问题，它会一直改写、一直重查，烧钱烧时间还不给结论。
"""
from app.config import settings
from app.services import llm

# ---------------- 1. 提问理解 ----------------

PLAN_PROMPT = """你是校园规章问答系统的「提问理解器」。把学生的问题转换成最适合检索的形式。

【最近的对话】（可能为空，用来消解指代）
__HISTORY__

【学生这次说】
__QUESTION__

输出要求：只输出一个 JSON 对象，不要输出任何其它文字。格式：
{
  "intent": "regulation | procedure | chitchat | out_of_scope",
  "needs_retrieval": true 或 false,
  "rewritten": "改写成适合检索的查询：补全指代、去掉口语和语气词、保留关键名词",
  "sub_queries": ["若是多主题或需要比较的问题，拆成 2-3 个独立子查询；否则给空数组"],
  "complexity": "simple 或 complex",
  "reason": "一句话说明判断依据"
}

intent 的判定标准：
- regulation：问规章制度的内容，如「考试作弊怎么处理」
- procedure：问办事流程，如「请假要办什么手续」
- chitchat：打招呼、感谢、闲聊，如「你好」「谢谢」
- out_of_scope：与校园规章完全无关，如「今天天气如何」

complexity 的判定标准：
- simple：单点事实，一次检索就能答，如「奖学金多少钱」
- complex：需要比较多条规定、跨章节整合、或条件分支，如「挂科和违纪哪个影响保研更大」

注意：rewritten 要把「它」「这个」「那」这类代词替换成前文提到的具体事物。
如果这句话本身就是完整问题，rewritten 可以基本等于原句，但要去掉「请问」「我想问一下」这类口语。"""


def _format_history(history: list[dict], limit: int = 4) -> str:
    """把最近几轮对话压成一行行文本，供指代消解使用。"""
    if not history:
        return "（这是第一轮对话，没有前文）"
    lines = []
    for item in history[-limit:]:
        role = "学生" if item.get("role") == "user" else "助手"
        content = (item.get("content") or "").replace("\n", " ")[:150]
        lines.append(f"{role}：{content}")
    return "\n".join(lines)


def _default_plan(question: str) -> dict:
    """规划失败时的兜底：按原句检索、走单跳。绝不因为规划失败就不答。"""
    return {
        "intent": "regulation",
        "needs_retrieval": True,
        "rewritten": question,
        "sub_queries": [],
        "complexity": "simple",
        "reason": "（规划失败，回退为原句单跳检索）",
        "_fallback": True,
    }


async def plan_query(question: str, history: list[dict]) -> dict:
    """理解提问：意图、是否检索、改写、子查询、复杂度。"""
    if settings.use_mock_llm or not settings.query_rewrite_enabled:
        return _default_plan(question)

    prompt = (PLAN_PROMPT
              .replace("__HISTORY__", _format_history(history))
              .replace("__QUESTION__", question))

    data = await llm.chat_json([{"role": "user", "content": prompt}])
    if not data:
        return _default_plan(question)

    plan = _default_plan(question)
    plan["_fallback"] = False

    intent = str(data.get("intent") or "regulation").strip().lower()
    plan["intent"] = intent if intent in {"regulation", "procedure", "chitchat", "out_of_scope"} else "regulation"
    plan["needs_retrieval"] = bool(data.get("needs_retrieval", True))

    rewritten = str(data.get("rewritten") or "").strip()
    plan["rewritten"] = rewritten or question

    raw_sub = data.get("sub_queries") or []
    plan["sub_queries"] = [str(item).strip() for item in raw_sub if str(item).strip()][:3]

    complexity = str(data.get("complexity") or "simple").strip().lower()
    plan["complexity"] = complexity if complexity in {"simple", "complex"} else "simple"

    plan["reason"] = str(data.get("reason") or "")[:200]

    # 闲聊和越界不需要检索，直接把 needs_retrieval 摁掉，
    # 避免模型自相矛盾（说自己是 chitchat 却又要检索）
    if plan["intent"] in {"chitchat", "out_of_scope"}:
        plan["needs_retrieval"] = False
        plan["complexity"] = "simple"

    return plan


# ---------------- 2. 重排 ----------------

RERANK_PROMPT = """你是检索结果重排器。下面有一个学生问题和若干条候选资料。
请按照「对回答这个问题有多有用」从高到低排序。

判断标准：
- 直接写明答案的排最前
- 相关但只是背景介绍的排中间
- 只是碰巧包含相同关键词、实际答非所问的排最后

【学生问题】
__QUESTION__

【候选资料】
__CANDIDATES__

只输出一个 JSON 对象，不要任何其它文字：
{"ranking": [编号从高到低排列，必须包含上面出现过的每一个编号，不重不漏]}"""


def _sanitize_ranking(raw, size: int) -> list[int] | None:
    """清洗模型给的排序：去重、剔除越界编号、补齐漏掉的编号。

    模型经常只给前几名，或者把编号写错。这里做两件事：
    1. 合法且未出现过的编号按顺序收下；
    2. 剩下的编号按原顺序补到末尾。
    这样即使模型输出很潦草，也不会丢候选。
    """
    if not isinstance(raw, list):
        return None
    ordered: list[int] = []
    for item in raw:
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= number <= size and number not in ordered:
            ordered.append(number)
    if not ordered:
        return None
    for number in range(1, size + 1):
        if number not in ordered:
            ordered.append(number)
    return ordered


async def rerank(question: str, candidates: list[dict]) -> dict:
    """用大模型对候选做 listwise 重排。

    候选太少的直接跳过——只有 5 条候选时，重排的收益不抵一次 API 调用的延迟。
    """
    stats = {"enabled": settings.rerank_enabled, "skipped": False, "reason": ""}

    if not settings.rerank_enabled:
        stats["reason"] = "配置关闭"
        return {"hits": candidates, "stats": stats}
    if settings.use_mock_llm:
        stats["reason"] = "演示模式"
        return {"hits": candidates, "stats": stats}
    if len(candidates) <= settings.rerank_top_n:
        stats["skipped"] = True
        stats["reason"] = f"候选仅 {len(candidates)} 条，无需重排"
        return {"hits": candidates, "stats": stats}

    blocks = []
    for index, hit in enumerate(candidates, start=1):
        snippet = (hit.get("text") or "").replace("\n", " ")[:settings.rerank_chars]
        section = hit.get("section") or ""
        label = f"{section} " if section else ""
        blocks.append(f"[{index}] {label}{snippet}")

    prompt = (RERANK_PROMPT
              .replace("__QUESTION__", question)
              .replace("__CANDIDATES__", "\n".join(blocks)))

    data = await llm.chat_json([{"role": "user", "content": prompt}], max_tokens=300)
    ranking = _sanitize_ranking((data or {}).get("ranking"), len(candidates))

    if ranking is None:
        stats["reason"] = "重排失败，保留融合顺序"
        return {"hits": candidates, "stats": stats}

    reordered = []
    for position, number in enumerate(ranking, start=1):
        hit = dict(candidates[number - 1])
        hit["rerank_position"] = position
        reordered.append(hit)

    stats["reranked"] = len(reordered)
    stats["reason"] = f"重排 {len(reordered)} 条候选"
    return {"hits": reordered, "stats": stats}


# ---------------- 3. 资料自省 ----------------

JUDGE_PROMPT = """你是资料充分性判定器。判断给出的资料是否足以回答学生的问题。

【学生问题】
__QUESTION__

【检索到的资料】
__CONTEXT__

判定标准：
- sufficient = true：资料里有能直接回答问题依据的内容（哪怕需要归纳整理）
- sufficient = false：资料完全没提到问题的主题，或只有边缘提及、无法支撑回答

只输出一个 JSON 对象，不要任何其它文字：
{
  "sufficient": true 或 false,
  "missing": "如果不充分，说明缺什么；充分时给空字符串",
  "rewritten": "如果不充分，给出一个更可能命中资料的查询（换个说法、或用更接近规章文件的措辞）；充分时给空字符串"
}"""


RECORD_SEPARATOR = "=" * 60


def build_evidence(hits: list[dict], char_limit: int = 400) -> str:
    """把命中的块拼成带编号的证据文本。编号用于让模型引用来源。"""
    blocks = []
    for index, hit in enumerate(hits, start=1):
        location = " · ".join(
            str(x) for x in (hit.get("filename"), hit.get("section"),
                             f"第{hit['page']}页" if hit.get("page") else None) if x
        )
        text = (hit.get("text") or "")[:char_limit]
        blocks.append(f"[{index}] 来源：{location or '未知'}\n{text}")
    return "\n\n".join(blocks)


async def judge_sufficiency(question: str, hits: list[dict]) -> dict:
    """判断检索到的资料是否足够回答问题。"""
    if not hits:
        return {"sufficient": False, "missing": "一条资料都没检索到",
                "rewritten": "", "skipped": False}
    if settings.use_mock_llm:
        return {"sufficient": True, "missing": "", "rewritten": "",
                "skipped": True, "reason": "演示模式跳过自省"}

    prompt = (JUDGE_PROMPT
              .replace("__QUESTION__", question)
              .replace("__CONTEXT__", build_evidence(hits, char_limit=300)))

    data = await llm.chat_json([{"role": "user", "content": prompt}], max_tokens=400)
    if not data:
        # 判定失败时倾向于「放行」：宁可让生成阶段用现有资料尽力回答，
        # 也不要因为一次判定调用抖动就把本来能答的问题拒掉。
        return {"sufficient": True, "missing": "", "rewritten": "",
                "skipped": True, "reason": "自省调用失败，默认放行"}

    return {
        "sufficient": bool(data.get("sufficient", True)),
        "missing": str(data.get("missing") or "")[:200],
        "rewritten": str(data.get("rewritten") or "").strip()[:200],
        "skipped": False,
    }
