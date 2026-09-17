"""零依赖中文分词与文本归一化。

## 为什么需要它

BM25 是「词袋」模型，效果对分词方式极其敏感。中文没有天然空格，
常规做法是引入 jieba 这类分词器——但本项目约定**不新增任何依赖**，
所以这里走另一条路：**字符二元组（char bigram）**。

## 为什么二元组反而更合适

「学生手册问答」这个场景有几个特点，恰好都对二元组有利：

1. **专业词表外词多**——「补考」「缓考」「留级试读」这类词，
   通用分词器未必收录，切错了整段就废了；二元组不依赖词典，不存在切错。
2. **精确标识符多**——「第七十七条」「学号」「教务处」。
   二元组保留了所有相邻字的组合，字符级命中率极高。
3. **容错**——用户把「教务处」打成「教务出」，二元组仍能命中「教务」这半个词，
   分词器则会直接切成两个不存在的词。

代价是索引变大、精确度略低于理想分词。实测（tests/eval_rag.py）
对 200 多个知识块的召回效果完全够用，且省掉一个依赖和一份词典。

## 中文数字归一化

学生手册里条款号几乎都是中文数字（「第七十七条」），而用户提问时
经常打阿拉伯数字（「第77条」）。如果不处理，这两者的一元/二元组
完全不重叠，BM25 直接失联。所以这里把「七十七」→「77」，
让两种写法落到同一个词上。
"""
import re

# 汉字区间：基本区 + 扩展 A，覆盖手册里可能出现的生僻字
_CJK = r"\u3400-\u4dbf\u4e00-\u9fff"
# 抽取「最小单位」：连续英文 / 连续数字（含小数）/ 单个汉字
_UNIT_RE = re.compile(rf"[A-Za-z]+|[0-9]+(?:\.[0-9]+)?|[{_CJK}]")
_IS_CJK_RE = re.compile(rf"[{_CJK}]")

# 中文数字表
_CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000}
_CN_NUM_RE = re.compile(r"[零〇一二三四五六七八九十百千两]+")

# 全角 → 半角：全角数字/字母/标点会让 token 分裂，必须先统一
_FULLWIDTH_OFFSET = 0xFEE0


def _to_halfwidth(text: str) -> str:
    """把全角字符折成半角（！＂＃…～ 以及全角数字字母）。

    不处理这一步的话，「第７７条」（全角）和「第77条」（半角）
    会变成两组毫不相干的 token。
    """
    out = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:            # 全角空格
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:  # 全角 ！ 到 ～
            out.append(chr(code - _FULLWIDTH_OFFSET))
        elif code == 0x200B:          # 零宽空格，直接丢弃
            continue
        else:
            out.append(ch)
    return "".join(out)


def _cn_to_int(text: str) -> int | None:
    """中文数字转整数。无法解析时返回 None。

    支持「七十七」→77、「一百二十三」→123、「十七」→17、「十」→10。
    只处理到千位——手册条款号不会超过这个量级。
    """
    section = 0
    number = 0
    for ch in text:
        if ch in _CN_DIGIT:
            number = _CN_DIGIT[ch]
        elif ch in _CN_UNIT:
            unit = _CN_UNIT[ch]
            # 「十七」里的「十」前面没有数字，按 1 算
            section += (number or 1) * unit
            number = 0
        else:
            return None
    return section + number


def _normalize_cn_numbers(text: str) -> str:
    """把长得像数字的中文数字串换成阿拉伯数字。

    只替换「含单位（十/百/千）」或「长度 ≥ 2」的串，避免误伤普通词：
    - 「一直」→ 只有一个「一」且没有单位 → 保持原样
    - 「第七十七条」→ 命中「七十七」→ 变成「第77条」
    """
    def replace(match: re.Match) -> str:
        raw = match.group(0)
        has_unit = any(ch in _CN_UNIT for ch in raw)
        if not has_unit and len(raw) < 2:
            return raw
        value = _cn_to_int(raw)
        return str(value) if value is not None else raw

    return _CN_NUM_RE.sub(replace, text)


def normalize(text: str) -> str:
    """归一化：全角转半角 → 中文数字转数字 → 压掉多余空白。"""
    return re.sub(r"\s+", " ", _normalize_cn_numbers(_to_halfwidth(text or "")))


def tokenize(text: str) -> list[str]:
    """把文本切成检索用的 token 序列。

    规则：
    - 连续的汉字：输出**每个单字** + **每对相邻字**（二元组）
      单字保证召回，二元组保证区分度。
    - 连续的英文/数字：作为一个整体输出（小写），例如「77」「gpa」「3.5」
    """
    tokens: list[str] = []
    cjk_run: list[str] = []

    def flush_cjk() -> None:
        """把攒下的连续汉字展开成单字 + 二元组。"""
        if not cjk_run:
            return
        tokens.extend(cjk_run)
        tokens.extend(cjk_run[i] + cjk_run[i + 1] for i in range(len(cjk_run) - 1))
        cjk_run.clear()

    for match in _UNIT_RE.finditer(normalize(text)):
        unit = match.group(0)
        if _IS_CJK_RE.fullmatch(unit):
            cjk_run.append(unit)
        else:
            flush_cjk()
            tokens.append(unit.lower())
    flush_cjk()
    return tokens


def tokenize_query(text: str) -> list[str]:
    """查询侧分词。去重后返回，避免用户重复输入同一个词时权重虚高。"""
    seen: dict[str, None] = {}
    for token in tokenize(text):
        seen[token] = None
    return list(seen)
