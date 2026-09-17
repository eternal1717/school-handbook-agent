"""全局配置。

所有可调参数都来自 backend/.env，业务代码禁止硬编码，也禁止直接读 os.environ。
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ 目录（本文件在 backend/app/config.py，parents[1] 即 backend）
BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- 大模型 ----------
    deepseek_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-flash"
    llm_mode: str = "auto"  # auto: 有 Key 走真实 API，没 Key 自动降级 mock；api / mock 可强制
    llm_temperature: float = 0.3
    # 思维链（reasoning_content）开关，实测数据（2026-09 探针）：
    #   不传该参数   → 11.0s，输出 1426 tok，其中思维链 943 tok
    #   thinking 关闭 →  2.7s，输出  394 tok，无思维链   ← 快 4 倍、省 3.6 倍
    #   thinking 开启 →  6.2s，输出  891 tok，思维链 526 tok
    # 结论：内部调用（意图分类/改写/重排/自省）一律走 off 的快通道，
    # 只有面向用户的最终回答才用 on，并把思维链展示出来。
    # auto = 内部调用 off、最终回答 on。
    llm_thinking: str = "auto"  # auto | on | off
    # deepseek-flash 是推理模型，**思维链和正文共享 max_tokens 额度**。
    # 实测 max_tokens=1500 时思维链会把额度吃光，正文输出 0 个片段——
    # 用户看到的就是一条空回答。真正的解法不是盲目调大，而是：
    #   1) 需要正文的调用给足额度（这里默认 4096）；
    #   2) 不需要正文的内部调用直接关掉 thinking（见 llm_thinking）。
    llm_max_tokens: int = 4096
    # 内部调用（分类/改写/重排/自省）的输出额度。这些调用已经关掉 thinking，
    # 输出只有几百 token 的 JSON，给 4096 纯属浪费额度上限（不影响计费，
    # 但会让出错的调用迟迟不返回）。
    llm_internal_max_tokens: int = 900
    llm_timeout: float = 60.0
    # 大模型请求是否跟随系统的 HTTP_PROXY / HTTPS_PROXY 环境变量。
    # 默认 False（忽略代理）：本地项目一旦从带残留代理变量的终端 / IDE 启动，
    # 请求会静默走一个不存在的代理，报出极难排查的 "Connection error."。
    # 确实需要走代理时，在 .env 里设成 true。
    llm_use_env_proxy: bool = False
    # 网络抖动（Connection error / 超时）时的自动重试次数。
    # 只在「还没吐出任何内容」时重试，避免用户看到重复的半截回答。
    llm_retries: int = 2
    # 「业务级」重试：用于意图分类 / 改写 / 重排这类一次性调用。
    # 它们不流式输出，重试是安全的；实测这类接口的 Connection error
    # 出现频率不低，没有这一层会让整条链路偶尔直接失败。
    llm_task_retries: int = 3

    # ---------- 数据库 ----------
    # SQLite 没有账号密码，只有文件路径；将来换 MySQL 时改成：
    # mysql+aiomysql://root:密码@localhost:3306/school_agent
    database_url: str = "sqlite+aiosqlite:///./data/school.db"

    # ---------- Embedding ----------
    embedding_model_path: str = r"R:\models\bge-small-zh-v1.5"
    embedding_dim: int = 512
    # BGE 中文模型的官方建议：查询侧加指令前缀，文档侧不加，能提升检索效果
    query_instruction: str = "为这个句子生成表示以用于检索相关文章："

    # ---------- 向量库 ----------
    chroma_dir: str = "./data/chroma"
    chroma_collection: str = "school_handbook"
    # BM25 索引落盘位置。它只是 ChromaDB 的派生缓存——内容以 Chroma 为准，
    # 这个文件删掉会自动重建，所以不必备份，也不该提交到 Git。
    bm25_index_path: str = "./data/bm25_index.json"

    # ---------- 检索 ----------
    top_k: int = 4
    similarity_threshold: float = 0.35
    chunk_size: int = 500
    chunk_overlap: int = 80

    # ---------- 混合检索（向量 + BM25）----------
    # 单一向量检索对「第七十七条」「学号」这类精确标识符召回很弱，加一路 BM25 互补。
    hybrid_enabled: bool = True
    vector_candidates: int = 30   # 向量这一路先捞多少候选
    bm25_candidates: int = 30     # BM25 这一路先捞多少候选
    # RRF（Reciprocal Rank Fusion）平滑常数，行业惯例取 60。
    # RRF 只用「排名」不用「分数」，所以不需要把余弦相似度和 BM25 分数
    # 归一化到同一量纲——这正是它比加权求和更省心的地方。
    rrf_k: int = 60
    fused_candidates: int = 20    # 两路融合后，进入重排的候选数
    rerank_top_n: int = 5         # 重排后最终留给大模型的资料条数
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # ---------- 重排 ----------
    # 零依赖方案：用大模型做 listwise 重排（把候选编号丢给模型，让它输出排序）。
    # 备选是本地交叉编码器 BGE-reranker，精度更高但要下 1GB 模型，故默认不开。
    rerank_enabled: bool = True
    rerank_chars: int = 180       # 每个候选截断到多少字再交给重排模型（控 token）

    # ---------- Agent（自省循环 + 工具调用）----------
    agent_enabled: bool = True
    # 自省循环最多再重查几次。行业共识：必须设硬上限，否则遇到答不了的问题
    # 会无限改写重查、烧钱烧时间。上限内仍不充分就明确拒答——
    # 「拒答」是特性，不是缺陷。
    agent_max_hops: int = 2
    # 复杂度路由：简单问题走单跳（快、便宜），复杂问题才进 Agent 循环。
    # 不区分的话，每个「学生证怎么补办」都要多花 3 倍延迟和成本。
    complexity_routing: bool = True
    query_rewrite_enabled: bool = True
    intent_routing: bool = True

    # ---------- 安全 ----------
    # 检索到的文档一律视为「证据」而非「指令」。
    # 手册里若出现「忽略以上指令」这类文本，不做防护时模型会照做。
    guard_enabled: bool = True

    # ---------- 可观测 ----------
    trace_enabled: bool = True

    # ---------- 增量去重（新增文档时只补差异部分）----------
    # 默认只开精确去重。实测（tests/calibrate_dedup.py）：
    #   完全相同              → 1.0000
    #   长段落只改一个字        → 0.9993  ← 改版手册的新内容会被误判成重复
    #   同义改写              → 0.9674
    # 也就是说语义去重「能抓的正好是危险的，想抓的又抓不到」，所以默认关闭。
    dedup_exact: bool = True          # 内容指纹完全相同的块，直接跳过（安全，建议常开）
    dedup_semantic: bool = False      # 语义高度相似的块也跳过（有误杀风险，按需开启）
    dedup_similarity: float = 0.97    # 语义去重阈值：相似度 >= 它就算重复

    # ---------- 记忆 ----------
    short_term_rounds: int = 10
    summary_every_rounds: int = 6

    # ---------- 服务 ----------
    host: str = "127.0.0.1"
    port: int = 8000

    # ---------- 派生属性 ----------
    @property
    def use_mock_llm(self) -> bool:
        """auto 模式下：没填 Key 就降级为演示模式，保证整条链路仍可跑通。"""
        if self.llm_mode == "mock":
            return True
        if self.llm_mode == "api":
            return False
        return not self.deepseek_api_key.strip()

    @property
    def database_url_resolved(self) -> str:
        """把相对路径的 sqlite 地址转成绝对路径。

        目的：不管在 PyCharm 里以什么工作目录启动，都能找到同一个数据库文件。
        """
        prefix = "sqlite+aiosqlite:///"
        if self.database_url.startswith(prefix):
            raw = self.database_url[len(prefix):]
            if raw.startswith("./") or raw.startswith("../"):
                absolute = (BACKEND_DIR / raw).resolve()
                return prefix + absolute.as_posix()
        return self.database_url

    @property
    def chroma_path(self) -> Path:
        p = Path(self.chroma_dir)
        return p if p.is_absolute() else (BACKEND_DIR / p).resolve()

    @property
    def bm25_path(self) -> Path:
        p = Path(self.bm25_index_path)
        return p if p.is_absolute() else (BACKEND_DIR / p).resolve()

    def thinking_for(self, purpose: str) -> bool:
        """按用途决定这次调用要不要开思维链。

        purpose = "answer"（面向用户的最终回答）| "internal"（分类/改写/重排/自省）
        auto 模式下：只有 answer 开。这样内部调用能拿到 4 倍速度、3.6 倍 token 节省。
        """
        mode = (self.llm_thinking or "auto").lower()
        if mode == "on":
            return True
        if mode == "off":
            return False
        return purpose == "answer"

    def ensure_dirs(self) -> None:
        """启动时确保数据目录存在（数据库、上传文件、向量库都在 backend/data 下）。"""
        for d in (DATA_DIR, UPLOAD_DIR, self.chroma_path):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
