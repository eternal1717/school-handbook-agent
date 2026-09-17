"""大模型调用层（DeepSeek，OpenAI 兼容接口）。

## 这一层解决的问题

原来只有「流式对话」和「一次性调用」两个函数。改造后要支撑 Agent 流程，
所以统一收拢成四件事：

1. **思维链开关**——实测数据决定策略（2026-09 实测，见 .env 注释）：
   关掉 thinking，速度提 4 倍、token 省 3.6 倍。所以内部调用
   （意图分类 / 查询改写 / 重排 / 自省判定）一律走 off 的快通道，
   只有面向用户的最终回答用 on，并把思维链作为独立事件吐给前端展示。

2. **思维链与正文分流**——`deepseek-flash` 是推理模型，
   `reasoning_content`（思考过程）和 `content`（正文）是**两个独立字段**。
   原实现只看 `content`，思维链被直接丢掉：既没展示给用户，
   也让"为什么模型这么答"变成黑盒。现在两者分开成事件。

3. **额度预算**——思维链和正文**共享 max_tokens**。
   实测 max_tokens=1500 时思维链全部吃光、正文输出 0 个片段，
   用户看到一条空回答。所以：需要正文的调用给足额度，
   不需要正文的内部调用直接关 thinking。

4. **工具调用循环**——DeepSeek 官方约束：带 `tools` 的请求，
   后续每一轮都必须把 `reasoning_content` 完整回传，否则 400。
   正确做法是 `messages.append(response.choices[0].message)`，
   绝不要手工拼 assistant 字典。

## 两种模式

- api ：真实调用 DeepSeek。
- mock：演示模式，不联网，直接用检索到的原文拼出回答。
       没填 API Key 时自动降级到 mock，保证界面和链路可以先跑通。
"""
import asyncio
import contextvars
import json
import re
import time
from collections.abc import AsyncIterator

from app.config import settings

_client = None

# 流式请求要不要带 stream_options={"include_usage": True}（用于统计 token）。
# 不是所有 OpenAI 兼容服务都支持这个参数，传了不支持会直接 400 把整次回答搞挂。
# 所以这里做成「乐观开启 + 自动降级」：一旦被拒就永久关闭，本次及后续都不再传。
# 比写死成配置更省心——用户不需要知道自己的服务端点支不支持。
_stream_usage_supported = True

# token 用量收集器。用 ContextVar 而不是模块级 list，
# 是为了让并发的多个问答请求各记各的，不会串味。
_usage_sink: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "llm_usage_sink", default=None
)


def start_usage_tracking() -> list:
    """开始收集本次请求内所有大模型调用的 token 用量。"""
    sink: list = []
    _usage_sink.set(sink)
    return sink


def _record_usage(purpose: str, usage) -> None:
    """把一次调用的用量记进当前上下文的收集器。"""
    sink = _usage_sink.get()
    if sink is None or usage is None:
        return
    try:
        detail = getattr(usage, "completion_tokens_details", None)
        sink.append({
            "purpose": purpose,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "reasoning_tokens": (getattr(detail, "reasoning_tokens", 0) or 0) if detail else 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        })
    except Exception:  # noqa: BLE001  统计字段结构变了不该影响主流程
        pass


def get_client():
    global _client
    if _client is None:
        if not settings.deepseek_api_key.strip():
            raise RuntimeError(
                ".env 里没有填 DEEPSEEK_API_KEY，请复制 .env.example 为 .env 并填入你的 Key"
            )
        import httpx
        from openai import AsyncOpenAI

        # trust_env=False：让 httpx 忽略 HTTP_PROXY / HTTPS_PROXY / ALL_PROXY 等环境变量。
        # 为什么必须这么做：如果进程从带残留代理变量的环境启动（IDE、终端、CI 里都可能发生），
        # httpx 会静默把请求发给那个代理，失败后只抛一句 "Connection error."，
        # 看起来像网络不通，实际是走了不存在的代理，极难定位。
        http_client = None
        if not settings.llm_use_env_proxy:
            http_client = httpx.AsyncClient(
                timeout=settings.llm_timeout, trust_env=False
            )

        _client = AsyncOpenAI(
            api_key=settings.deepseek_api_key.strip(),
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout,
            http_client=http_client,
        )
    return _client


# ---------------- 思维链开关 ----------------

def _thinking_body(purpose: str, thinking: bool | None = None) -> dict | None:
    """按用途拼 extra_body。

    DeepSeek 用 `extra_body={"thinking": {"type": "disabled"|"enabled"}}` 控制。
    实测 deepseek-flash 默认「开」，所以关闭时必须显式传 disabled。

    `thinking` 传布尔值时强制覆盖配置——用于「思维链吃光额度、正文为空」
    之后关掉思考重答一次的自愈路径。
    """
    enabled = settings.thinking_for(purpose) if thinking is None else thinking
    return {"thinking": {"type": "enabled" if enabled else "disabled"}}


# ---------------- 1. 流式对话（正文 + 思维链分流）----------------

async def stream_chat(
    messages: list[dict],
    purpose: str = "answer",
    thinking: bool | None = None,
    max_tokens: int | None = None,
) -> AsyncIterator[dict]:
    """流式调用，逐段产出事件。

    yield 两种事件：
        {"type": "reasoning", "text": "..."}  ← 思维链（思考过程）
        {"type": "content",   "text": "..."}  ← 正文

    `thinking` / `max_tokens` 传值时覆盖配置，供上层在「正文被思维链挤空」
    之后关掉思考重答使用。

    网络抖动（Connection error / 超时）会按 LLM_RETRIES 自动重试。
    重试有硬约束：**只在还没吐出任何内容时才能重试**——已经输出了一半再重来，
    用户会看到重复的半截回答，比直接报错更糟。注意「任何内容」包含思维链：
    思维链已经开始输出再重试，前端会看到两段思考过程。
    """
    attempts = max(0, settings.llm_retries) + 1
    last_error: Exception | None = None

    global _stream_usage_supported

    for attempt in range(attempts):
        emitted = False
        try:
            client = get_client()
            kwargs: dict = {
                "model": settings.llm_model,
                "messages": messages,
                "temperature": settings.llm_temperature,
                "max_tokens": max_tokens or settings.llm_max_tokens,
                "stream": True,
                "extra_body": _thinking_body(purpose, thinking),
            }
            if _stream_usage_supported:
                kwargs["stream_options"] = {"include_usage": True}

            stream = await client.chat.completions.create(**kwargs)
            async for chunk in stream:
                # 开启 include_usage 后，最后一个 chunk 的 choices 是空的、只在 usage 上带数据，
                # 所以要在 continue 之前把 usage 捞出来，否则统计永远是空的。
                if getattr(chunk, "usage", None) is not None:
                    _record_usage(purpose, chunk.usage)
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                # 思维链字段在 delta 上，字段名 reasoning_content
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    emitted = True
                    yield {"type": "reasoning", "text": reasoning}
                content = getattr(delta, "content", None)
                if content:
                    emitted = True
                    yield {"type": "content", "text": content}
            return
        except Exception as exc:  # noqa: BLE001  重试耗尽后原样抛出，交给上层提示
            last_error = exc
            # 服务端点不支持 stream_options 时，把它关掉重试一次。
            # 只在还没输出任何内容时这么做，避免重复吐出半截回答。
            if _stream_usage_supported and not emitted and _looks_like_unsupported_param(exc):
                _stream_usage_supported = False
                print("[llm] 该端点不支持 stream_options，已自动关闭流式用量统计")
                continue
            if emitted or attempt == attempts - 1:
                raise
            # 指数退避，给网络一点恢复时间
            await asyncio.sleep(0.8 * (attempt + 1))

    if last_error is not None:
        raise last_error


def _looks_like_unsupported_param(exc: Exception) -> bool:
    """判断异常是不是「参数不被支持」。

    不按具体异常类判断（不同 SDK 版本不一样），直接看消息里有没有
    关键字段名和 400/invalid 这类词——够用且不挑版本。
    """
    text = f"{type(exc).__name__} {exc}".lower()
    if "stream_options" not in text:
        return False
    return any(word in text for word in ("400", "invalid", "unrecognized", "unsupported", "unknown"))


# ---------------- 2. 一次性调用 ----------------

async def chat_once(
    messages: list[dict],
    max_tokens: int | None = None,
    purpose: str = "internal",
    temperature: float = 0.1,
) -> str:
    """一次性调用（不流式）。失败时返回空串，绝不影响主流程。

    为什么单独加一层业务级重试（llm_task_retries）：实测这类接口的
    Connection error 出现频率不低。非流式调用重试是**幂等安全**的
    （不会重复输出），所以这里可以放心重试，而流式那边不能。
    """
    attempts = max(1, settings.llm_task_retries)
    for attempt in range(attempts):
        try:
            client = get_client()
            response = await client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or settings.llm_internal_max_tokens,
                stream=False,
                extra_body=_thinking_body(purpose),
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001  这是增强项，失败不应中断问答
            if attempt == attempts - 1:
                print(f"[llm] 调用失败（已重试 {attempts} 次），跳过该步骤：{exc}")
                return ""
            await asyncio.sleep(0.6 * (attempt + 1))
    return ""


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.S)
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)


def parse_json_object(raw: str) -> dict | None:
    """从模型输出里抠出第一个 JSON 对象。

    模型经常不守规矩——加 markdown 代码围栏、前面写一句"好的，结果如下"、
    或者结尾多一段解释。这里逐层剥离，尽量不因为格式问题丢掉整次调用。
    """
    if not raw:
        return None
    text = raw.strip()

    # 1) 直接解析
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass

    # 2) 去掉 ```json 围栏
    stripped = _FENCE_RE.sub("", text)
    try:
        data = json.loads(stripped)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass

    # 3) 暴力抠第一个 {...}
    match = _JSON_BLOCK_RE.search(stripped)
    if match:
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None
    return None


async def chat_json(
    messages: list[dict],
    max_tokens: int | None = None,
    purpose: str = "internal",
    temperature: float = 0.0,
) -> dict | None:
    """调一次模型并解析成 JSON 对象。解析失败返回 None。"""
    raw = await chat_once(messages, max_tokens=max_tokens, purpose=purpose,
                          temperature=temperature)
    return parse_json_object(raw)


# ---------------- 3. 工具调用循环 ----------------

async def run_tool_loop(
    messages: list[dict],
    tools: list[dict],
    executor,
    max_rounds: int = 2,
    purpose: str = "internal",
) -> dict:
    """让模型自主调用工具收集证据，返回收集到的结果。

    executor: 可调用对象，签名 `await executor(name, arguments: dict) -> str`，
              返回给模型的工具执行结果（会被塞进 role="tool" 的消息里）。
              设计成异步是因为工具的底层（检索）本身是异步的。

    返回 {"messages": [...], "tool_calls": [...], "rounds": n, "ok": bool}。

    ⚠️ 关键约束（DeepSeek 官方文档）：
    带 tools 的请求，后续每一轮都必须把上一轮 assistant 消息里的
    `reasoning_content` 完整回传，否则直接 400。
    唯一稳妥的写法就是 `messages.append(response.choices[0].message)`
    ——SDK 返回的 message 对象本身就带 reasoning_content 和 tool_calls，
    手工拼 dict 一定会漏字段。
    """
    work = list(messages)
    tool_calls: list[dict] = []
    rounds = 0

    for round_index in range(max(1, max_rounds)):
        try:
            client = get_client()
            response = await client.chat.completions.create(
                model=settings.llm_model,
                messages=work,
                tools=tools,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=settings.llm_internal_max_tokens,
                stream=False,
                extra_body=_thinking_body(purpose),
            )
        except Exception as exc:  # noqa: BLE001  工具循环失败要能降级，不能拖垮整条链路
            print(f"[llm] 工具调用第 {round_index + 1} 轮失败：{exc}")
            return {"messages": work, "tool_calls": tool_calls, "rounds": rounds, "ok": False}

        message = response.choices[0].message
        _record_usage(f"tool_round_{round_index + 1}", response.usage)
        work.append(message)  # 必须整对象回传，reasoning_content 才会被带上
        rounds += 1

        pending = getattr(message, "tool_calls", None)
        if not pending:
            break  # 模型认为证据够了，不再调工具

        for call in pending:
            name = call.function.name
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            result = await executor(name, arguments)
            tool_calls.append({"name": name, "arguments": arguments, "result_preview": str(result)[:200]})
            work.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })

    return {"messages": work, "tool_calls": tool_calls, "rounds": rounds, "ok": True}


# ---------------- 演示模式（无 Key）----------------

def mock_answer(question: str, hits: list[dict]) -> str:
    """演示模式回答：不改写、不编造，直接整理检索到的原文片段。"""
    if not hits:
        return "手册中未找到相关内容。"

    lines = [
        "【演示模式】当前未配置大模型 API Key，以下是我从规章文档中检索到的原文片段：",
        "",
    ]
    for hit in hits[:3]:
        section = f"（{hit['section']}）" if hit.get("section") else ""
        snippet = hit["text"].replace("\n", " ")[:120]
        lines.append(f"· {section}{snippet}……")
    lines += [
        "",
        "在 .env 里填入 DEEPSEEK_API_KEY 后，这里会变成完整的自然语言回答。",
    ]
    return "\n".join(lines)


# ---------------- 长期记忆相关 ----------------

EXTRACT_PROMPT = """你是记忆抽取器。判断用户这句话里是否包含值得长期记住的信息（个人身份、偏好、长期事实）。
只输出 JSON，不要输出其它内容，格式：
{"save": true/false, "key": "简短英文键名", "value": "用中文概括这句话里的长期信息"}

用户说：__USER_TEXT__"""

# 注意：上面这段 prompt 里带了 JSON 示例（一堆花括号），
# 所以只能用 replace 填值，绝不能用 str.format —— format 会把 {"save": ...}
# 当成格式化占位符解析，抛 KeyError: '"save"'，而且是运行时才炸。
EXTRACT_PROMPT_PLACEHOLDER = "__USER_TEXT__"


def _rule_based_memory(user_text: str) -> dict | None:
    """兜底规则：用户明确说「记住」时，即使没有大模型也要能存下来。"""
    if "记住" not in user_text:
        return None
    cleaned = re.sub(r"(请|帮我|你要|记住|一下|哦|啊)", "", user_text).strip(" ，,。")
    if len(cleaned) < 2:
        return None
    return {"key": "user_note", "value": cleaned[:120]}


async def extract_memory(user_text: str) -> dict | None:
    """抽取长期记忆。有 Key 时用大模型判断，没 Key 时走规则兜底。"""
    if settings.use_mock_llm:
        return _rule_based_memory(user_text)

    data = await chat_json(
        [{"role": "user",
          "content": EXTRACT_PROMPT.replace(EXTRACT_PROMPT_PLACEHOLDER, user_text)}],
    )
    if data and data.get("save") and data.get("value"):
        key = (data.get("key") or f"note_{int(time.time())}").strip()[:120]
        key = re.sub(r"[^\w\-]", "_", key) or f"note_{int(time.time())}"
        return {"key": key, "value": str(data["value"])[:120]}
    return _rule_based_memory(user_text)


async def summarize(history: list[dict]) -> str:
    """把最近若干轮对话压缩成一句要点，用于长期记忆。"""
    if not history:
        return ""

    dialogue = "\n".join(f"{item['role']}：{item['content'][:200]}" for item in history)
    if settings.use_mock_llm:
        # 演示模式：直接取用户问过的问题作为话题摘要
        questions = [item["content"] for item in history if item["role"] == "user"]
        return ("；".join(questions))[:80]

    summary = await chat_once(
        [
            {
                "role": "user",
                "content": f"把下面这段师生对话压缩成不超过 60 字的中文要点，只输出要点本身：\n{dialogue}",
            }
        ],
        max_tokens=600,
    )
    return summary[:120]
