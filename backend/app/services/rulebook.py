"""规章结构化：把「条文」抽成「规则」与「流程」两套可计算的结构。

## 它补的是哪一层

检索层解决的是「找到相关条文」。但学生真正想知道的两件事，
检索层答不了：

1. **「我这情况会不会违规？」**——条文是普遍陈述，学生手里的是具体事实。
   「本科学生累计休学不得超过 2 次」这条规定，对已经休了 3 次的人
   和没休过的人，含义完全不同。检索层只会把这条原文端出来，
   **比对的活儿留给了学生自己**。
2. **「我要按什么顺序去办？」**——条文里的办理环节是散文形式，
   学生要自己从「经所在学院初审、教务处审核、学校审批后生效」
   里拆出三步。

所以这一层做的是**从非结构化文本里抽出结构化知识**：
- 规则：`IF 条件 THEN 后果`，带上可比较的阈值（数量 + 单位）
- 流程：有序步骤 + 材料清单 + 每步依据

## 为什么先探测再动手

写抽取器之前先量化了语料的真实形态，结果推翻了最初的设想：

| 设想 | 实测 |
| --- | --- |
| 条文之间互相引用，可以做引用图谱 | **「依据第X条」全文仅 3 处，涉及 2 个块** → 否决 |

所以图谱那条路走不通——这本手册是 29 份规章拼接的，
虽然 131 处出现《》，但大多是《入党志愿书》这类**表单名**和上位法，
不构成条款间的引用网络。

真正的结构信号在另外三处，这也决定了下面模板的重心：

| 信号 | 频次 | 用途 |
| --- | --- | --- |
| 条件句（应当/不得/可以/须） | 321 | 抽规则 |
| 流程动词（申请/审核/审批/备案） | 634 | 抽流程 |
| 数字阈值（3 个工作日 / 2 次 / 4 门） | 399 | 规则里可比对的部分 |

编号形态也测过，**括号中文「（一）」703 次**是绝对主力（对应「第X条第（一）项」），
圈号①②③ 只有 6 次——所以切分序号项时以括号中文为准。

## 为什么用模板而不是大模型

大模型抽得准，但：慢（2192 块要跑很多轮）、贵、不可复现、
且抽错了没法解释。这批条文句式相当规整（公文体），
模板能覆盖主干，而且**每条规则都能说清是从哪个正则、哪段文本抽出来的**——
这比一个说不出理由的抽取结果可信得多。

抽不全的部分由检索层兜底：诊断页会同时给出「直接命中的条文原文」，
用户不会因为规则抽漏了就什么都看不到。
"""
import json
import re
import time
from pathlib import Path

from app.config import settings
from app.services import vector_store
from app.services.bm25 import BM25Index
from app.services.tokenizer import normalize

# 归一化后仍可能残留的单字中文数字（normalize 只处理含单位的或长度≥2 的串）
# 例如「三次」的「三」是单字无单位，不会被 normalize 转换，但对阈值比对很关键。
_SINGLE_CN = {"一": "1", "二": "2", "两": "2", "三": "3", "四": "4",
              "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}
# 「分」刻意排除：「十分」「一分」在中文里常是程度副词，误转会污染数值
_SINGLE_CN_RE = re.compile(
    r"(?P<n>[一二两三四五六七八九])\s*(?P<u>次|门|天|日|年|周|个月|学分|学时|课时|小时|人次|工作日)"
)

# PDF 抽取经常在汉字之间留下空格（「学 生 手 册」），会让所有模板失配
_CJK_SPACE_RE = re.compile(r"(?<=[\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff])")

# 句子切分：句号 / 分号 / 换行。逗号不切——条文里的条件从句经常横跨逗号。
_SENTENCE_RE = re.compile(r"[。；;！!？?\n]+")

# 子句分隔符——所有模板的字符类都必须用它。
#
# 这里有个实测踩出来的坑：normalize() 会把全角「，；：！？」折成半角「,;:!?」
# （U+FF0C 在 FEE0 偏移区间内），只有「。」不在那个区间、保持原样。
# 所以字符类里只写全角「，；：」等于**什么都没排除**，
# 匹配串会一路吃到句尾，抽出来的 subject 变成一整段话。
# 必须全角 + 半角都写上。
#
# 「、」刻意不收：它是并列顿号，「警告、严重警告由学院审批决定」里的顿号
# 属于主语内部，排除掉会把主语切残。
_CLAUSE_BREAK = r"，,。；;：:！!？?"


def _not_clause(*extra: str) -> str:
    """生成「不是子句分隔符」的字符类内容，便于按模板额外排除字符。"""
    return "^" + _CLAUSE_BREAK + "".join(extra)


def _not_sentence(*extra: str) -> str:
    """同上，但**逗号不在排除之列**。

    权责链经常横跨逗号——「退学由所在学院提交材料,教务处审核,学校研究审定后…」
    是一条完整的责任链。用 _CLAUSE_BREAK 去切会把 who 截断成「所在学院提交材料」，
    后面接不上 act，整条规则直接消失（实测 authority 从 32 条掉到 3 条）。
    """
    return "^。；;：:！!？?" + "".join(extra)


# ======================= 规则模板 =======================
# 每个模板对应一种公文体常见句式。命名沿用「T序号」方便在注释里互相引用。

# T1 情形列举：「有下列情形之一，不予办理转学：入学未满一学期、毕业前一年、…」
# 这是信息量最大的一类——条件是一张清单，后果是明确动作。
_RE_T1_LIST = re.compile(
    r"有下列(?:情形|情况)之一[，,]?\s*(?P<cons>[" + _not_clause(r"：:") + r"]{2,40})[：:]\s*(?P<items>[^。]{6,})"
)

# T2 时限：「学生需在 3 个工作日内提交书面申辩材料」
# 「受理后 15 个工作日内完成复查」
_RE_T2_DEADLINE = re.compile(
    r"(?P<subject>[" + _not_clause() + r"]{0,16}?)(?:需|应|须|应当|必须|可以|可|要)?(?:在|于|自)?[" + _not_clause() + r"]{0,10}?"
    r"(?P<num>\d+)\s*(?P<unit>个工作日|工作日|个月|周|天|日|小时|分钟|学年|学期)\s*内\s*"
    r"(?P<action>[" + _not_clause() + r"]{2,44})"
)

# T3 禁止：「未经批准不得擅自离校」「不予办理学籍注册」
# 「不可」刻意不收——它在手册里几乎只以「不可抗力」出现，
# 收了会把「除不可抗力等正当理由外」切成 modal=不可 / 后果=抗力…，
# 是实测踩出来的误报。同理「不得」要排除「不得抗力」这种切法。
_RE_T3_PROHIBIT = re.compile(
    r"(?P<subject>[" + _not_clause() + r"]{0,20}?)(?P<modal>不得(?!抗力)|不予|禁止|严禁)"
    r"(?P<cons>[" + _not_clause() + r"]{2,44})"
)

# T4 权限归属：「警告、严重警告由学院审批决定」
# 「由」前排除「理**由**」「自**由**」「经**由**」——它们的「由」不是权责引导词。
#
# 机构名的校验刻意放在 Python 里而不是正则里：
# 一开始写成 `who` 必须以机构词**结尾**，结果全军覆没——
# 「学院审批决定」里「学院」在开头、结尾是「审批」，
# 而正则需要「审批」紧跟在 who 末尾的机构词之后，于是怎么回溯都匹配不上，
# 32 条权责规则直接掉到 1 条。改成「正则管结构、代码管词表」后恢复正常：
# 正则只负责切出「由 X 审批」里的 X，X 里有没有机构交给 _ORG_RE 判。
_ORG_RE = re.compile(
    r"学院|学部|教务部|教务处|学工部|学生工作部|办公会|委员会|办公室|中心|学校|系部|部|处|系|科"
)
_RE_T4_AUTHORITY = re.compile(
    r"(?P<matter>[" + _not_clause() + r"]{2,24}?)(?<![理自经来缘故有])由"
    r"(?P<who>[" + _not_sentence() + r"]{2,44}?)"
    r"(?P<act>审批|审定|核准|批准|核定|决定|受理|备案)"
)

# T5 阈值约束：「本科学生累计休学不得超过 2 次」
_RE_T5_LIMIT = re.compile(
    r"(?P<subject>[" + _not_clause() + r"]{0,20}?)(?P<modal>不得超过|不超过|不得高于|不得低于|不低于|最多|最少|至少)\s*"
    r"(?P<num>\d+)\s*(?P<unit>学年|学期|个月|周|个工作日|工作日|天|门|次|学分|学时|课时|小时|年)"
)

# T6 条件从句：「学生对处分不服的，可在收到决定书 10 个工作日内提交校内申诉」
_RE_T6_IF = re.compile(
    r"(?P<cond>[" + _not_clause() + r"]{5,44}?)的[，,]\s*(?P<cons>[^。；;]{5,70})"
)

# ======================= 流程模板 =======================

# P2 「经 A、B、C 后」：「经所在学院初审、教务处审核、学校审批后生效」
_RE_P2_VIA = re.compile(r"经(?P<steps>[^。；;：:，,]{4,70}?)(?:后|方可)")

# P3 「由 A 提交，B 审核，C 审定」——逗号链，每段是「主体 + 动词」
_RE_P3_CHAIN = re.compile(r"由(?P<steps>[^。；;]{6,90})")

# 「（一）…（二）…」序号项，以括号中文为主力形态（实测 703 次）
_RE_ENUM_ITEM = re.compile(r"[（(](?P<no>[一二三四五六七八九十]{1,3})[)）]\s*(?P<text>[^（(。]{4,140})")

# 流程动词：判断一段文字是否在描述「办理环节」而非「实体规定」
_PROCESS_VERBS = (
    "申请", "申报", "提交", "审核", "初审", "复核", "审批", "审定", "核准",
    "批准", "报送", "备案", "办理", "填写", "公示", "评议", "认定", "发放", "签收",
)
_PROCESS_VERB_RE = re.compile("|".join(_PROCESS_VERBS))

# section 里出现这些词，说明这一条大概率是在讲「怎么办」而不是「什么算违规」。
# 用来给 P2/P3 抽出来的流程打置信度——它们不像箭头链那样自带结构，
# 误报率更高，标出来比藏着强。
_PROCEDURE_HINT_RE = re.compile(r"流程|程序|步骤|办理|评审|申报|审批|认定|管理办法")

# 材料：「提交书面申请及相关证明材料」「(如病假须有医院证明)」
_RE_MATERIAL = re.compile(
    r"(?:提交|提供|出具|附|附上|携带|持)\s*(?P<items>[^。；;：:，,]{2,56}?)(?:材料|证明|申请书|申请表|文件|证件)"
)
_RE_MATERIAL_BRACKET = re.compile(r"[（(]\s*如(?P<items>[^）)]{2,50}?)[)）]")

# 规则的种类 → 给人看的标签
# 抽取器版本。**改动任何模板、归一化逻辑或单位表后必须 +1。**
# 规则库按「块数 + 版本」判断缓存是否新鲜：只比块数的话，
# 改了抽取逻辑而知识库没变时旧缓存会被一直复用，
# 表现为「代码明明改了、结果却一点没变」，排查起来很费劲。
EXTRACTOR_VERSION = 2

KIND_LABELS = {
    "list": "情形列举",
    "deadline": "时限要求",
    "prohibit": "禁止性规定",
    "authority": "权责归属",
    "limit": "数量上限",
    "condition": "条件与后果",
}

# ======================= 运行时状态 =======================
_rulebook: dict | None = None
_match_index: BM25Index | None = None


# ======================= 文本清理 =======================
def _clean(text: str) -> str:
    """归一化 + 修掉 PDF 抽取留下的汉字间空格 + 单字中文数字转阿拉伯。

    三步的顺序有讲究：先 normalize（全角→半角、含单位的中文数字→数字），
    再补单字数字（normalize 不管的「三次」），最后清空格——
    如果先清空格，「第 77 条」会被粘成「第77条」而丢掉「第 条」的边界，
    反倒不影响，但先 normalize 能少走一遍正则。
    """
    cleaned = normalize(text or "")
    cleaned = _SINGLE_CN_RE.sub(lambda m: _SINGLE_CN[m.group("n")] + m.group("u"), cleaned)
    return _CJK_SPACE_RE.sub("", cleaned).strip()


def _sentences(text: str) -> list[str]:
    """把知识块切成句子。过滤过短的碎片，它们几乎不承载完整规范。"""
    return [s.strip() for s in _SENTENCE_RE.split(text) if len(s.strip()) >= 8]


def _split_items(blob: str) -> list[str]:
    """把「A、B、C」或「A，B，C」切成清单项。"""
    parts = re.split(r"[、,，;；]", blob)
    return [p.strip(" 　·") for p in parts if len(p.strip()) >= 3][:12]


def _is_process_text(text: str) -> bool:
    """判断一段文字是否在描述办理环节。

    至少要出现两个流程动词才算——只出现一个是巧合
    （「学生应当申请…」这种单动词句是实体规定，不是流程）。
    """
    return len(_PROCESS_VERB_RE.findall(text)) >= 2


# 句首残留的编号：PDF 抽取常把「5.」「第三十二条」「（一）」留在句子里
_LEAD_NOISE_RE = re.compile(r"^(?:[0-9]+[.．、]|[（(][一二三四五六七八九十0-9]+[)）]|第[一二三四五六七八九十百0-9]+[条章节款项])\s*")


def _trim_subject(text: str, limit: int = 16) -> str:
    """把主语裁到「紧挨着情态动词的那一小段」。

    模板里的 subject 是从句首非贪婪截出来的，中间允许任意字符，
    于是「因故无法按期入学的,必须提前向学校履行书面请假手续,请假时长一般不超过2周」
    会截出「提前向学校履行书面请假手续,请假时长一般不超过」这种一长串前缀。

    用正则去约束边界会写成一团难以维护的字符类，这里直接后处理：
    **只留最后 limit 个字**——不管前面挂了多少修饰，真正的主语总是紧贴情态词的。
    """
    cleaned = _LEAD_NOISE_RE.sub("", text.strip(" ，,、；;：:　"))
    return cleaned[-limit:] if len(cleaned) > limit else cleaned


def _source_of(chunk: dict, evidence: str) -> dict:
    """规则/流程的来源坐标，供前端「定位原文」用。"""
    return {
        "doc_id": chunk.get("doc_id") or "",
        "filename": chunk.get("filename") or "",
        "section": chunk.get("section") or "",
        "page": chunk.get("page"),
        "chunk_index": chunk.get("chunk_index"),
        "chunk_id": chunk.get("id") or "",
        "evidence": evidence[:300],
    }


# ======================= 规则抽取 =======================
def _extract_rules(chunks: list[dict]) -> list[dict]:
    """逐块扫描，套用全部模板产出规则。

    这里是纯函数式的：同样的语料必然产出同样的规则，可复现、可 diff。
    每条规则都带 `template` 字段记录它由哪个模板产出，便于回溯误报。
    """
    rules: list[dict] = []
    seen: set[str] = set()

    def add(rule: dict | None) -> None:
        """去重后追加。去重键用「种类 + 主陈述」的前 40 字——
        同一句话可能同时命中 T5 和 T6，重复展示只会干扰阅读。"""
        if not rule:
            return
        statement = rule.get("statement") or ""
        if len(statement) < 6:
            return
        key = f"{rule['kind']}|{statement[:40]}"
        if key in seen:
            return
        seen.add(key)
        rule["id"] = f"rule-{len(rules) + 1:04d}"
        rules.append(rule)

    for chunk in chunks:
        text = _clean(chunk.get("text") or "")
        if not text:
            continue
        for sentence in _sentences(text):
            src = _source_of(chunk, sentence)

            for m in _RE_T1_LIST.finditer(sentence):
                cons = m.group("cons").strip()
                add({
                    "kind": "list",
                    "template": "T1",
                    "condition": "存在下列情形之一",
                    "consequence": cons,
                    "items": _split_items(m.group("items")),
                    "statement": sentence[:160],
                    "source": src,
                })

            for m in _RE_T5_LIMIT.finditer(sentence):
                subject = _trim_subject(m.group("subject"))
                add({
                    "kind": "limit",
                    "template": "T5",
                    "subject": subject,
                    "condition": f"{subject}{m.group('modal')}" if subject else m.group("modal"),
                    "consequence": f"{m.group('modal')}{m.group('num')}{m.group('unit')}",
                    "threshold": {
                        "value": int(m.group("num")),
                        "unit": m.group("unit"),
                        "modal": m.group("modal"),
                    },
                    "statement": sentence[:160],
                    "source": src,
                })

            for m in _RE_T2_DEADLINE.finditer(sentence):
                action = m.group("action").strip()
                # 排除「第二十条 10 个工作日内」——条号后的数字不是时限
                if re.match(r"^[条章节款项]", action):
                    continue
                add({
                    "kind": "deadline",
                    "template": "T2",
                    "subject": _trim_subject(m.group("subject")),
                    "condition": f"在{m.group('num')}{m.group('unit')}内",
                    "consequence": action,
                    "threshold": {
                        "value": int(m.group("num")),
                        "unit": m.group("unit"),
                        "modal": "须在",
                    },
                    "statement": sentence[:160],
                    "source": src,
                })

            for m in _RE_T4_AUTHORITY.finditer(sentence):
                matter = _trim_subject(m.group("matter"), 20)
                who = m.group("who").strip()
                # 主体必须是机构。不校验的话「…并附原处分决定书复印件」这类
                # 没有机构的长串也会被当成本主体——这是实测踩出来的误报。
                if not _ORG_RE.search(who):
                    continue
                add({
                    "kind": "authority",
                    "template": "T4",
                    "condition": matter,
                    "consequence": f"由{who}{m.group('act')}",
                    "statement": sentence[:160],
                    "source": src,
                })

            for m in _RE_T3_PROHIBIT.finditer(sentence):
                cons = m.group("cons").strip()
                if len(cons) < 3:
                    continue
                subject = _trim_subject(m.group("subject"), 20)
                add({
                    "kind": "prohibit",
                    "template": "T3",
                    "subject": subject,
                    # 禁止性条文的主语常是「未按规定缴纳学费、住宿费的」这种情境，
                    # 它本身就是触发条件，比单放一个「不予」有信息量得多。
                    "condition": subject or m.group("modal"),
                    "consequence": m.group("modal") + cons,
                    "modality": m.group("modal"),
                    "statement": sentence[:160],
                    "source": src,
                })

            for m in _RE_T6_IF.finditer(sentence):
                cond = _LEAD_NOISE_RE.sub("", m.group("cond").strip())
                # 「的，」前必须是完整条件短语；带括号的说明文字不是条件
                if len(cond) < 6 or re.search(r"[（(]", cond):
                    continue
                add({
                    "kind": "condition",
                    "template": "T6",
                    "condition": cond,
                    "consequence": m.group("cons").strip(),
                    "statement": sentence[:160],
                    "source": src,
                })

    return rules


# ======================= 流程抽取 =======================
def _steps_from_text(blob: str) -> list[str]:
    """把一段「A、B、C」或「A，B，C」的描述切成步骤，并剔掉明显不是步骤的碎片。"""
    parts = re.split(r"[、,，]", blob)
    steps = []
    for part in parts:
        step = part.strip(" 　·。；")
        # 剥掉「流程为 / 程序是」这类引导语，它不是步骤内容
        step = re.sub(r"^(?:流程为|流程是|程序为|程序是|步骤为|步骤是)\s*", "", step)
        if len(step) < 3:
            continue
        # 步骤里应当出现动作；纯名词短语（如「教务处」）不算一步
        if not _PROCESS_VERB_RE.search(step) and not re.search(r"初审|复核|审议|备案|公示", step):
            continue
        steps.append(step)
    return steps


def _confidence_for(template: str, section: str) -> str:
    """给流程的置信度定级。

    P1（箭头链）和 P4（序号项）自带结构标记，是人工整理过的形态，直接 high。
    P2/P3 是靠句式猜的，「由…，…」在手册里也可能只是普通的权责陈述，
    所以要看 section 是不是在讲「怎么办」——section 是「…退学办理」就 high，
    是「…考核与成绩记载」就降为 medium，让前端能把它排在后面。
    """
    if template in ("P1", "P4"):
        return "high"
    return "high" if _PROCEDURE_HINT_RE.search(section or "") else "medium"


def _extract_procedures(chunks: list[dict]) -> list[dict]:
    """抽出办事流程。

    难点在于「事项名」——用户搜「休学」得能命中流程，
    但条文里很少直接写「休学流程」这几个字。
    解决办法是**借 section 标题当事项归属**：
    「第五章 …奖学金管理办法 · 第六条 评审流程」天然说明了这条流程属于哪件事。
    实测下来 section 的信息量远高于正文里的裸句子。
    """
    procedures: list[dict] = []
    seen: set[str] = set()

    def add(chunk: dict, section: str, doc: str, steps: list[str], template: str, evidence: str) -> None:
        key = "|".join(steps)[:60]
        if key in seen:
            return
        seen.add(key)
        topic = _topic_of(section, doc)
        procedures.append({
            "id": f"proc-{len(procedures) + 1:04d}",
            "title": topic or section or doc,
            "topic": topic,
            "section": section,
            "steps": steps,
            "materials": _materials_from(evidence),
            "template": template,
            "confidence": _confidence_for(template, section),
            "source": _source_of(chunk, evidence),
        })

    for chunk in chunks:
        raw = _clean(chunk.get("text") or "")
        if not raw:
            continue
        section = (chunk.get("section") or "").strip()
        doc = chunk.get("filename") or ""

        # ---- P1 箭头链：质量最高（「个人申报→学院初审公示5个工作日→学校复核」）----
        # 全库只有 7 处，但每一处都是人工整理过的标准流程，优先级最高。
        if "→" in raw:
            steps: list[str] = []
            for piece in raw.split("→"):
                # 箭头链的最后一段常常粘着后续说明（「…发文发证。弄虚作假者取消资格」），
                # 按句号切一次，只留前半句
                head = _SENTENCE_RE.split(piece.strip())[0].strip(" 　·。；,，")
                if len(head) >= 2:
                    steps.append(head)
            if len(steps) >= 3:
                add(chunk, section, doc, steps, "P1", raw)
                continue

        # ---- P2/P3 句子级：经…后 / 由…链 ----
        for sentence in _sentences(raw):
            if not _is_process_text(sentence):
                continue
            steps = []
            template = ""
            for pattern, tag in ((_RE_P2_VIA, "P2"), (_RE_P3_CHAIN, "P3")):
                m = pattern.search(sentence)
                if not m:
                    continue
                # 「未经」「已经」「曾经」里的「经」不是环节引导词
                head = sentence[:m.start()]
                if tag == "P2" and re.search(r"[未已曾]{1}$", head):
                    continue
                candidate = _steps_from_text(m.group("steps"))
                if len(candidate) >= 2:
                    steps = candidate
                    template = tag
                    break
            if len(steps) >= 2:
                add(chunk, section, doc, steps, template, sentence)

        # ---- P4 序号项：整个块是「（一）…（二）…」的步骤列表 ----
        # 只在 section 明示「流程 / 程序 / 步骤」时才收，否则会把处分档次表
        # 当成办事流程，那是两回事。
        if re.search(r"流程|程序|步骤", section):
            items = [m.group("text").strip() for m in _RE_ENUM_ITEM.finditer(raw)]
            if len(items) >= 3:
                add(chunk, section, doc, items[:12], "P4", raw)

    return procedures


def _topic_of(section: str, filename: str) -> str:
    """从 section 里剥出事项名。

    section 形如「第五章 柳州工学院学生奖学金管理办法（校发〔2024〕121号） · 第六条 评审流程」，
    去掉章号、校发文号、条号后剩下的「学生奖学金管理办法」才是能拿去匹配的事项名。
    """
    tail = section.split("·")[-1] if "·" in section else section
    head = section.split("·")[0] if "·" in section else ""
    base = head or tail or filename
    base = re.sub(r"^第[一二三四五六七八九十百]+[章条节]\s*", "", base)
    base = re.sub(r"[（(][^）)]*校发[^）)]*[)）]", "", base)
    base = re.sub(r"[（(][^）)]*[)）]", "", base)
    return base.strip(" 　·")[:60]


def _materials_from(text: str) -> list[str]:
    """抽材料清单。两种形态：显式「提交…材料」和括号补充说明「(如病假须有医院证明)」。"""
    found: list[str] = []
    for m in _RE_MATERIAL.finditer(text):
        items = m.group("items").strip(" 　·和及与")
        if 2 <= len(items) <= 56:
            found.append(items)
    for m in _RE_MATERIAL_BRACKET.finditer(text):
        items = m.group("items").strip(" 　·")
        if 2 <= len(items) <= 50:
            found.append(items)
    # 去重且保序
    unique: list[str] = []
    for item in found:
        if item not in unique:
            unique.append(item)
    return unique[:8]


# ======================= 构建与持久化 =======================
def _build() -> dict:
    """全量重建规则库。2192 块纯正则扫描，秒级完成。"""
    chunks = list(vector_store.iter_all_chunks())
    rules = _extract_rules(chunks)
    procedures = _extract_procedures(chunks)

    by_kind: dict[str, int] = {}
    for rule in rules:
        by_kind[rule["kind"]] = by_kind.get(rule["kind"], 0) + 1

    return {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "extractor_version": EXTRACTOR_VERSION,
        "chunk_total": len(chunks),
        "rules": rules,
        "procedures": procedures,
        "stats": {
            "rule_total": len(rules),
            "procedure_total": len(procedures),
            "by_kind": by_kind,
            "with_threshold": sum(1 for r in rules if r.get("threshold")),
            "documented_rules": len({r["source"]["chunk_id"] for r in rules}),
        },
    }


def rebuild() -> dict:
    """强制重建并落盘。入库新文档后应调用它。"""
    global _rulebook, _match_index
    book = _build()
    path = settings.rulebook_path_resolved
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(book, ensure_ascii=False), encoding="utf-8")
    _rulebook = book
    _match_index = None          # 规则变了，匹配索引必须重建
    return _summary(book)


def get_rulebook() -> dict:
    """取规则库。缓存文件有效就直接用，否则重建。

    和 BM25 索引同样的策略：ChromaDB 是事实源，规则库是派生缓存，
    按「块数是否一致」判断缓存是否过期——比维护增量更新可靠得多。
    """
    global _rulebook
    expected = _count_chunks()

    # 内存缓存也要校验块数（对齐 get_bm25 的做法）。
    # 只靠调用方记得 invalidate 是不够的：多进程、脚本直接入库、
    # 以后新增的入库入口，任何一个漏了都会让规则库一直用旧数据，
    # 表现成「新传的文档怎么都抽不出规则」——这种问题很难查。
    # count() 是 Chroma 的内部计数，很便宜，换来的是永远不用旧数据。
    if _rulebook is not None and _rulebook.get("chunk_total") == expected:
        return _rulebook
    _rulebook = None

    path = settings.rulebook_path_resolved
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cached = None
        # 只有「块数一致」且「抽取器版本一致」才认这个缓存。
        # 块数管的是知识库变了的情况；版本管的是抽取逻辑变了的情况——
        # 少了后一条，改完模板跑起来会发现结果没变（旧缓存还在用），
        # 很容易误判成「代码没生效」而白白排查半天。
        if (cached
                and cached.get("chunk_total") == expected
                and cached.get("extractor_version") == EXTRACTOR_VERSION):
            _rulebook = cached
            return _rulebook

    rebuild()
    return _rulebook or {"rules": [], "procedures": [], "stats": {}}


def _count_chunks() -> int:
    """只数条数，不拉内容。校验缓存是否新鲜时会被调用，必须便宜。"""
    return vector_store.count()


def _summary(book: dict) -> dict:
    return {
        "built_at": book.get("built_at"),
        "extractor_version": book.get("extractor_version"),
        "chunk_total": book.get("chunk_total"),
        **(book.get("stats") or {}),
    }


def stats() -> dict:
    return _summary(get_rulebook())


def invalidate() -> None:
    """入库/删文档后调用，下次访问时重建。"""
    global _rulebook, _match_index
    _rulebook = None
    _match_index = None


# ======================= 匹配 =======================
def _ensure_match_index() -> BM25Index:
    """把规则库建成 BM25 索引。

    这里刻意复用检索层的 BM25Index 而不是另写一套相似度计算：
    规则匹配的本质就是「用学生的一句话去检索规则库」，
    和「用学生的一句话去检索知识库」是同一个问题。
    复用同一套实现意味着同样的分词、同样的中文数字归一化、
    同样的 IDF 逻辑——两处的行为天然一致，不会各调各的。
    """
    global _match_index
    if _match_index is not None:
        return _match_index

    book = get_rulebook()
    docs = []
    for rule in book.get("rules") or []:
        # 拼成一段可检索文本：把结构化的各个字段都摊进来，
        # 让「我挂了3门课」能同时命中 condition 和 consequence 两侧的词。
        parts = [
            rule.get("statement", ""),
            rule.get("condition", ""),
            rule.get("consequence", ""),
            rule.get("subject", ""),
            " ".join(rule.get("items") or []),
            (rule.get("source") or {}).get("section", ""),
        ]
        docs.append({
            "id": rule["id"],
            "text": " ".join(p for p in parts if p),
            "filename": (rule.get("source") or {}).get("filename", ""),
            "section": (rule.get("source") or {}).get("section", ""),
            "page": None,
        })

    index = BM25Index(k1=settings.bm25_k1, b=settings.bm25_b)
    index.build(docs)
    _match_index = index
    return index


# 学生口述里出现的数值 + 单位，用于和规则阈值比对
_USER_NUM_RE = re.compile(
    r"(?P<num>\d+)\s*(?P<unit>个工作日|工作日|门课|门|次|天|日|学分|学时|课时|小时|个月|周|学年|学期|年)"
)
_UNIT_ALIAS = {"门课": "门", "日": "天", "个工作日": "工作日"}


def _user_numbers(text: str) -> dict[str, int]:
    """抽出学生情境里的「数值 + 单位」。同单位取最大值——
    「挂了3门，其中2门重修」里 3 门才是关键的那个数。"""
    found: dict[str, int] = {}
    for m in _USER_NUM_RE.finditer(_clean(text)):
        unit = _UNIT_ALIAS.get(m.group("unit"), m.group("unit"))
        value = int(m.group("num"))
        if value > found.get(unit, 0):
            found[unit] = value
    return found


def _verdict(rule: dict, user_nums: dict[str, int]) -> dict | None:
    """把规则阈值和学生自述的数值比对，给出提示。

    **刻意保守**：只做数值大小关系的陈述（「你 3 门，规定上限 4 门，距上限还差 1 门」），
    不做「你违规了」这种断言。原因有二：
    1. 条文常有前置条件和例外，模板抽出来的只是主干，据此下结论不负责任；
    2. 这是规章场景，误判的代价由学生承担。
    所以 status 只表示「数值关系」，文案也明确写成「据此条文」而非「你将被处分」。
    """
    threshold = rule.get("threshold")
    if not threshold:
        return None
    unit = threshold["unit"]
    if unit not in user_nums:
        return None

    mine = user_nums[unit]
    limit = threshold["value"]
    modal = threshold.get("modal", "")
    gap = abs(limit - mine)

    # 上限型：「不得超过 2 次」——超过就违反
    if modal in ("不得超过", "不超过", "不得高于", "最多"):
        if mine > limit:
            return {"status": "over", "text": f"你提到 {mine}{unit}，已超过该条文上限 {limit}{unit}"}
        if mine == limit:
            return {"status": "edge", "text": f"你提到 {mine}{unit}，正好等于该条文上限 {limit}{unit}"}
        return {"status": "under", "text": f"你提到 {mine}{unit}，距该条文上限 {limit}{unit} 还差 {gap}{unit}"}

    # 下限型：「不得低于 / 至少」——不足就不达标
    if modal in ("不得低于", "不低于", "至少", "最少"):
        if mine < limit:
            return {"status": "under", "text": f"你提到 {mine}{unit}，低于该条文要求 {limit}{unit}，差 {gap}{unit}"}
        return {"status": "ok", "text": f"你提到 {mine}{unit}，达到该条文要求 {limit}{unit}"}

    # 时限型：「须在 3 个工作日内」——只陈述时限，不判断是否超期（缺少起始时点）
    if modal == "须在":
        return {"status": "info", "text": f"该条文要求 {limit}{unit} 内完成，你提到的是 {mine}{unit}"}

    return None


def diagnose(situation: str, limit: int | None = None) -> dict:
    """情境诊断：学生描述自己的情况 → 找出与之相关的规章规定。

    返回的每条都带：
    - 规则本身（条件 / 后果 / 阈值）
    - 数值比对提示（如果单位对得上）
    - 原文出处坐标（可一键定位）
    """
    situation = (situation or "").strip()
    top_n = limit or settings.rulebook_match_top_n
    if not situation:
        return {"situation": "", "numbers": [], "matches": [], "note": "请先描述你的情况"}

    book = get_rulebook()
    index = _ensure_match_index()
    by_id = {r["id"]: r for r in book.get("rules") or []}

    # 多取一些候选，因为下面还要按数值相关度重排
    hits = index.search(situation, top_n * 4)
    user_nums = _user_numbers(situation)

    matches = []
    for hit in hits:
        rule = by_id.get(hit["id"])
        if not rule:
            continue
        verdict = _verdict(rule, user_nums)
        # 单位对得上的规则优先——学生报了「3 门」，涉及「门」的条文比泛泛相关更值得看
        bonus = 1.0 if verdict else 0.0
        matches.append({
            "rule_id": rule["id"],
            "kind": rule["kind"],
            "kind_label": KIND_LABELS.get(rule["kind"], rule["kind"]),
            "template": rule.get("template"),
            "condition": rule.get("condition", ""),
            "consequence": rule.get("consequence", ""),
            "subject": rule.get("subject", ""),
            "items": rule.get("items") or [],
            "threshold": rule.get("threshold"),
            "statement": rule.get("statement", ""),
            "verdict": verdict,
            "score": round(hit.get("bm25_score", 0.0) * (1 + bonus), 3),
            "source": rule.get("source"),
        })

    matches.sort(key=lambda item: (item["verdict"] is not None, item["score"]), reverse=True)

    return {
        "situation": situation,
        "numbers": [{"unit": u, "value": v} for u, v in sorted(user_nums.items())],
        "matches": matches[:top_n],
        "candidate_total": len(hits),
        "note": (
            "以下内容由条文模板自动抽取，仅反映条文与描述的关联及数值大小关系，"
            "不构成处分或资格判定。请以学校最新正式文本及主管部门答复为准。"
        ),
    }


def list_rules(query: str = "", kind: str = "", limit: int = 50) -> dict:
    """列出 / 搜索规则，供「规则库」浏览页用。"""
    book = get_rulebook()
    rules = book.get("rules") or []
    if kind:
        rules = [r for r in rules if r["kind"] == kind]
    if query:
        index = _ensure_match_index()
        ranked = index.search(query, limit)
        by_id = {r["id"]: r for r in rules}
        rules = [by_id[h["id"]] for h in ranked if h["id"] in by_id]
    else:
        rules = rules[:limit]
    return {
        "total": len(book.get("rules") or []),
        "returned": len(rules),
        "kinds": KIND_LABELS,
        "items": [
            {
                "id": r["id"],
                "kind": r["kind"],
                "kind_label": KIND_LABELS.get(r["kind"], r["kind"]),
                "condition": r.get("condition", ""),
                "consequence": r.get("consequence", ""),
                "subject": r.get("subject", ""),
                "threshold": r.get("threshold"),
                "items": (r.get("items") or [])[:8],
                "statement": r.get("statement", ""),
                "source": r.get("source"),
            }
            for r in rules
        ],
    }


def list_procedures(query: str = "", limit: int = 30) -> dict:
    """列出 / 搜索办事流程。"""
    book = get_rulebook()
    items = book.get("procedures") or []

    if query:
        # 先用 BM25 在流程语料上排一遍，命中不足时再用子串兜底——
        # 「休学」这类词可能只出现在 title 里而没进 steps，
        # 子串匹配能补上 BM25 因分词粒度漏掉的case。
        index = BM25Index(k1=settings.bm25_k1, b=settings.bm25_b)
        index.build([
            {
                "id": p["id"],
                "text": " ".join([p.get("title", ""), p.get("topic", ""), " ".join(p.get("steps") or [])]),
                "filename": "", "section": "", "page": None,
            }
            for p in items
        ])
        ranked = index.search(query, limit)
        by_id = {p["id"]: p for p in items}
        picked = [by_id[h["id"]] for h in ranked if h["id"] in by_id]
        if len(picked) < limit:
            picked_ids = {p["id"] for p in picked}
            blob = _clean(query)
            for p in items:
                if p["id"] in picked_ids:
                    continue
                haystack = p.get("title", "") + p.get("topic", "") + "".join(p.get("steps") or [])
                if blob and blob in _clean(haystack):
                    picked.append(p)
                    picked_ids.add(p["id"])
                if len(picked) >= limit:
                    break
        items = picked

    return {
        "total": len(book.get("procedures") or []),
        "returned": len(items),
        "items": items[:limit],
    }


def get_procedure(procedure_id: str) -> dict | None:
    book = get_rulebook()
    for item in book.get("procedures") or []:
        if item["id"] == procedure_id:
            return item
    return None
