# 知津 · 校园规章智能问答

> 「知津」取自《论语》「使子路问津焉」——问津即问路。学生问规章，系统做指路人。

把学生手册这类规章制度喂进去，学生用大白话提问，系统给出**带条款出处的答案**；
查不到就明确说查不到，绝不编。

不是「上传文档 + 向量检索 + 拼一段话」那套演示级 RAG，而是一条完整的
**Agentic RAG** 链路：查询改写 → 混合检索 → 重排 → 自省重查 → 注入防护 → 带思维链生成，
每一步都留痕、可回放、可评测。

在此之上还多抽了一层：把非结构化的规章条文**自动抽成结构化规则与办事流程**，
于是系统不只回答「哪条规定这么写了」，还能回答「**我这情况会触发哪条规定**」，
以及「这件事该按什么顺序去办」。

**零新增依赖。** BM25、RRF 融合、中文分词与数字归一化、规则抽取、注入扫描、
链路追踪全部手写；前端用本地 vendor 文件，克隆下来不用 `npm install` 就能跑。
用到的第三方库只有原本就有的那几样（FastAPI / SQLAlchemy / ChromaDB / sentence-transformers）。

<img src="docs/screenshots/00-chat.png" width="900">

---

## 能力速览

| 能力 | 怎么做的 | 实测数据 |
| --- | --- | --- |
| 带出处的回答 | 每条结论标来源规章 + 章节 + 页码，可一键「定位原文」 | — |
| 混合检索 | 向量 + 手写 BM25 双路召回，RRF 只看排名融合 | Recall@5 **80.0% → 93.3%** |
| 复杂度路由 | 简单问题单跳，复杂的才进 Agent 工具循环 | 多跳 17654 ms vs 单跳 4993 ms |
| 自省重查 | 资料不够就换个问法重查，最多 2 跳 | — |
| 该拒就拒 | 资料撑不住就明说「未找到」，绝不推测 | 拒答判定 **25/25** |
| 注入防护 | 检索内容一律当「证据」不当「指令」，两侧都扫 | 攻击下未泄露系统提示 |
| 规则诊断 | 大白话描述情况 → 匹配条文 + **数值大小比对** | 366 条规则 / 24 条流程 |
| 办事流程 | 从条文抽步骤，带置信度分档，勾选框当待办 | 全量重建 **0.7 秒** |
| 增量入库 | 内容指纹去重，重复内容自动跳过 | 手册复制一份：新增 0 / 跳过 211 |
| 全链路追踪 | 每步耗时、两路召回、重排前后顺序全留痕 | P50 5320 ms / P95 18201 ms |
| 零依赖前端 | Vue 3 + Element Plus 走本地 vendor 文件 | 不用 `npm install` |

> 表格里每个数字的实测过程、以及「为什么最后选了这个方案」，都在
> **[docs/DESIGN.md](docs/DESIGN.md)** 里。

## 快速开始

### 1. 装依赖

```bash
pip install -r backend/requirements.txt
```

### 2. 配置密钥

从样例复制一份配置，填入 DeepSeek 的 Key：

```bash
cp backend/.env.example backend/.env
```

```env
DEEPSEEK_API_KEY=你的Key
```

> **不填也能跑。** 没有 Key 会进入**演示模式**：回答由检索到的原文片段拼成，
> 界面和链路完全可用，只是不做真实大模型调用。
>
> 另外 `EMBEDDING_MODEL_PATH` 要指向本地的 BGE-small-zh-v1.5 模型目录
> （默认 `R:\models\bge-small-zh-v1.5`），这个模型是离线加载的，不联网下载。

### 3. 导入手册

```bash
python backend/ingest.py "D:\路径\学生手册.docx"
```

支持 docx / pdf / txt / md。也可以直接给一个**目录**，会导入目录下所有支持的文件：

```bash
python backend/ingest.py "D:\某目录"
```

重复内容会自动跳过，只补新增部分（见下文「知识库」）。

### 4. 启动

```bash
python backend/run.py
```

控制台出现启动横幅后，打开 <http://127.0.0.1:8000/>。

> `run.py` 会自动把工作目录切到 `backend/`，所以从哪个目录执行都不会跑偏。
> 端口被占用就改 `.env` 里的 `PORT`。

### 5. PyCharm 用户

运行配置已经预置好了，三步启动：**[PYCHARM.md](PYCHARM.md)**
（配解释器 → 配运行项 → 点 Run）。

## 界面与操作

左侧导航分两组，再往下是「会话记录」列出全部历史会话：

- **工作台** —— 问答 / 规则诊断 / 办事流程 / 知识库 / 长期记忆
- **工程可信** —— 链路追踪

地址栏支持 `#rules`、`#proc`、`#trace` 直接定位到对应页面；
右上角可切亮色 / 暗色主题，选择会记住。

<img src="docs/screenshots/01-home.png" width="900">

### 问答 —— 提问，然后看它凭什么这么答

底部输入框提问，回车发送（`Shift + Enter` 换行）。每条回答下面挂着三样东西：

- **来源卡片** —— 命中的条文，标着出自哪份规章、第几页、哪个章节，以及是
  「向量命中」还是「关键词命中」（两路都中会标「双路命中」）。
  点「定位原文」直接跳到知识库里那一段。
- **思考过程** —— 默认折叠。点开能看到意图判定、改写后的查询、子查询拆分、检索计划。
- **耗时流水条** —— 理解 / 检索 / 重排 / 自查 / 生成各花了多少毫秒。

答得不对就点右下角「**有帮助 / 没帮助**」。点踩会连整条链路一起记下来，
还能一键**导出成评测集条目**（见「链路追踪」）。

### 规则诊断 —— 用自己的话描述情况

这是普通 RAG 做不到的一块。不用问「第几条写了什么」，直接说自己的情况，例如
「我已经休学两次了」「等了 12 个工作日还没消息」。

系统会匹配相关条文，并把**数值大小关系**直接摆出来 —— 但**刻意不下结论**：

<img src="docs/screenshots/05-rule-verdict.png" width="900">

条文常有前置条件和例外，模板抽出来的只是主干，据此下结论不负责任。
界面上也明确写着「不构成处分或资格判定」。

<img src="docs/screenshots/02-rules.png">

### 办事流程 —— 抽出来的步骤可以直接当待办

从条文里自动抽出的办理步骤，按顺序排好并标注原文出处。勾选框可以直接当待办清单用，
每一步还能看「需要准备的材料」。

<img src="docs/screenshots/03-proc.png">

### 知识库 —— 上传、查看、删除

点「导入文档」选文件上传。入库后表格里能看到每份文档切成了多少个知识块，
还有一列「跳过重复」显示本次有多少内容是已存在的。

点「**看分块**」能看某份文档具体切成了哪些块 —— 「定位原文」跳的就是这里。

**增量入库**：只有知识库里没有的内容会被写入。每个块入库时存一个**内容指纹**
（归一化文本的 sha256），上传新文档时逐块比对，命中的跳过。

| 上传的文档 | 新增 | 跳过 |
| --- | --- | --- |
| 手册全文复制一份 | 0 | 211 |
| 手册全文 + 加一段新规定 | 2 | 211 |

> ⚠️ **一个要知道的行为**：被跳过的内容仍归属**原文档**。所以如果 A、B 两份文档有重叠内容，
> 删除 A 会连带移除那部分重叠内容。想彻底换手册时，建议先用新文件名上传确认内容无误，
> 再删除旧文档。

### 链路追踪 —— 每一次问答都能回放

链路概览给出平均耗时、**P50 / P95**、拒答率、各阶段平均耗时、平均自省跳数。
可以勾「只看拒答」——「学生问了什么却被拒了」是最该复盘的信号，
它直接告诉你知识库缺哪一块内容。

点进单条链路能看到完整过程：改写前后的查询、两路召回各自的结果、
重排前后的顺序变化、Agent 工具调用、每阶段耗时、token 消耗。

顺带一提，这里也是**反馈闭环**的出口：点踩记录可以「查看点踩记录」，
也可以按「导出评测集格式」一键导出成评测用例 —— 用户踩过的坑，第二天就变成回归测试。

<img src="docs/screenshots/04-trace.png">

### 长期记忆 —— 它会记住

- **短期记忆**：本会话最近 10 轮，外加上一轮的改写结果
  —— 这是指代消解能工作的前提（「那它最长可以请多久」里的「它」）。
- **长期记忆**：跨会话保留。用户说「记住…」时显式写入；每累计 6 轮对话自动摘要一条。

「长期记忆」页右上角可以**切换用户标识**，记忆和会话都按用户隔离，
切过去就能看到各自独立的内容。注意顶部统计栏的「长期记忆」是**所有用户合计**，
这一页显示的是**当前用户**的，两个数不一定相等。

## 设计要点

完整的取舍论证、实测数据、踩坑排查过程都在 **[docs/DESIGN.md](docs/DESIGN.md)**。
这里先给结论，方便快速判断这个项目解决了哪些真问题：

| # | 决策 | 为什么 |
| --- | --- | --- |
| 1 | **混合检索**：向量 + 手写 BM25，RRF 融合 | 两路的失败模式互补。BM25 分数无上界，加权求和得先归一化、容易被单个异常高分带偏；RRF 只看排名绕开这个问题。**Recall@5 80.0% → 93.3%** |
| 2 | **中文分词**用字符二元组，**数字归一化**手写 | 中文没有空格，二元组零依赖且对短查询够用；归一化让「第七十七条」和「第 77 条」返回同一个块 |
| 3 | **复杂度路由**：不是所有问题都走 Agent | 单跳能答准的问题硬塞进 Agent 循环，多花 3 倍延迟和成本、收益为零。检索本身只要 184 ms，慢的全在模型调用上 |
| 4 | **内部任务关掉思维链** | `deepseek-flash` 的思维链和正文**共享 `max_tokens`**。关闭后快 4 倍、省 3.6 倍 token；改写的中间过程没人看，不值得 |
| 5 | **拒答是特性，不是缺陷** | 法规问答最怕的不是答不上来，是**一本正经地编**。资料撑不住就明说「未找到」 |
| 6 | **引用必须能回到原文** | 每条来源带 `doc_id` + `chunk_index`。引用如果无法回溯，那就只是装饰 |
| 7 | **检索内容当「证据」不当「指令」** | 这是 RAG 的真实攻击面。做法是结构上区分 + 边界标记，**标记而非删除**——手册里完全可能正常引述这类句子 |
| 8 | **条文抽成结构化规则** | 纯正则六个模板，**不调大模型**，0.7 秒重建 366 条规则。学生真正想问的是「我这情况会怎样」 |
| 9 | **数值比对只陈述关系、不下结论** | 条文常有前置条件和例外，据模板下结论不负责任 |
| 10 | **语义去重默认关闭** | 实测「长段落只改一个字」相似度高达 0.9993（会误杀改版新内容），而真正的同义改写只有 0.9674（又抓不到）。**能抓的正好是危险的** |
| 11 | **派生缓存用最笨的方式维护** | 按「缓存条数 vs 向量库条数」校验，不一致就整体重建。比维护增量结构可靠得多 |
| 12 | **前端能用原生 DOM 就不用组件库** | 覆写 Element Plus 默认主题的视觉权重，成本比手写还高 |

### 踩过的坑（精选）

这几条都很反直觉 —— **不报错、不崩溃，只是行为悄悄变差**。
完整排查过程在 [DESIGN.md](docs/DESIGN.md)：

- **`LLM_MAX_TOKENS` 给小了会出现「空回答」** —— 推理模型的思维链和正文共享额度，
  思维链吃光后正文输出 0 个字符。看起来像程序坏了，其实是配置。默认给 **8192**。
- **`bool("false")` 恒为 `True`** —— 模型返回的 JSON 里布尔经常是字符串，
  导致「自省重查」整段被静默跳过。**凡是解析模型返回值，禁止直接 `bool()`**，统一走 `_as_bool()`。
- **两侧单位写法对不上就静默跳过比对** —— 规则抽出来是「个工作日」，用户侧归一化成「工作日」，
  一个 `not in` 判断就把这次比对吞了。**归一化函数必须两侧共用**。
- **删会话只删父表** —— `feedback` / `trace` 会变成悬空数据：满意度统计算着已删会话，
  badcase 点开找不到原文。子表要带 `conversation_id` 一并清掉。
- **进程继承残留代理变量** —— 请求会静默发给一个不存在的代理，只抛一句 `Connection error.`，
  极难定位。项目默认忽略系统代理（`LLM_USE_ENV_PROXY=false`）。
- **`str.format` 撞上 JSON 花括号** —— 提示词里含 JSON 示例时只能 `str.replace`，
  用 `format` 会抛 `KeyError`，而且**演示模式下不会暴露**。
- **前端两个静态检查发现不了的坑** —— `svg` 不写尺寸会按 300×150 撑爆容器；
  给 inline 元素设 `width` 会被忽略。只能靠 `tests/ui_shot.py` 截图看。

## 配置项说明

全部配置在 `backend/.env`（从 `.env.example` 复制）。所有项都有合理默认值，
**最小配置只需要填 `DEEPSEEK_API_KEY`**。

### 大模型

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | 空 | 不填则走演示模式 |
| `LLM_BASE_URL` | `https://api.deepseek.com` | OpenAI 兼容接口 |
| `LLM_MODEL` | `deepseek-flash` | 推理模型 |
| `LLM_MODE` | `auto` | `auto` = 有 Key 走真实 API、无 Key 自动降级；也可强制 `api` / `mock` |
| `LLM_THINKING` | `auto` | 内部调用关思维链、最终回答开并把思考过程展示出来 |
| `LLM_MAX_TOKENS` | **8192** | ⚠️ **必须给足**，理由见上「踩过的坑」 |
| `LLM_INTERNAL_MAX_TOKENS` | `900` | 内部调用额度（已关思维链，只输出几百 token 的 JSON） |
| `LLM_USE_ENV_PROXY` | `false` | 忽略 `HTTP_PROXY` / `HTTPS_PROXY` 等残留代理变量 |
| `LLM_RETRIES` / `LLM_TASK_RETRIES` | `2` / `3` | 只重试「还没输出任何内容」的情况，避免用户看到重复的半截回答 |

### 检索链路（逐项可关，正好用来做消融）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `HYBRID_ENABLED` | `true` | 向量 + BM25 混合检索 |
| `RERANK_ENABLED` | `true` | 大模型 listwise 重排（零依赖；备选是 BGE-reranker，精度更高但要下 1GB 模型，故默认不开） |
| `AGENT_ENABLED` | `true` | 自省循环 + 工具调用 |
| `AGENT_MAX_HOPS` | `2` | 自省最多再查几次。**必须设硬上限**，否则答不了的问题会无限改写重查、烧钱烧时间 |
| `COMPLEXITY_ROUTING` | `true` | 简单问题走单跳，复杂问题才进 Agent 循环 |
| `QUERY_REWRITE_ENABLED` | `true` | 查询改写与子查询拆分 |
| `INTENT_ROUTING` | `true` | 意图分流（闲聊 / 越界不检索） |
| `GUARD_ENABLED` | `true` | 注入防护 |
| `TRACE_ENABLED` | `true` | 全链路追踪 |

关掉其中任意一项，跑一次 `tests/eval_rag.py --retrieval-only` 就能看到它对召回的影响。

### 其他

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `EMBEDDING_MODEL_PATH` | `R:\models\bge-small-zh-v1.5` | 本地模型，**禁止联网下载** |
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/school.db` | 换 MySQL 只改这一行（装 `aiomysql` 即可，代码不用动） |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `80` | 文本切分粒度 |
| `TOP_K` | `4` | 最终留给大模型的资料条数 |
| `SIMILARITY_THRESHOLD` | `0.35` | 不再用它一刀切拒答，改为触发自省判定 |
| `RULEBOOK_ENABLED` | `true` | 条文 → 规则 / 流程抽取 |
| `RULEBOOK_MATCH_TOP_N` | `6` | 情境诊断最多返回几条相关规则 |
| `DEDUP_EXACT` / `DEDUP_SEMANTIC` | `true` / `false` | 精确去重开、语义去重关（理由见「设计要点」第 10 条） |
| `SHORT_TERM_ROUNDS` / `SUMMARY_EVERY_ROUNDS` | `10` / `6` | 短期记忆轮数 / 累计多少轮总结一次长期记忆 |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | 服务监听地址 |

每一项的取舍理由都写在 `app/config.py` 的注释里。

## 接口一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/chat` | 提问，SSE 流式返回（含思维链、阶段、来源、链路 id） |
| POST | `/api/knowledge/upload` | 上传文档并增量入库（重复内容自动跳过） |
| POST | `/api/knowledge/ingest-local` | 直接导入本机文件路径（同样走增量去重） |
| GET | `/api/knowledge/list` | 已入库文档列表 |
| GET | `/api/knowledge/{doc_id}/chunks` | 查看某文档切成了哪些知识块（「定位原文」靠它） |
| DELETE | `/api/knowledge/{doc_id}` | 删除文档及其向量 |
| GET | `/api/conversations` | 会话列表 |
| GET | `/api/conversations/{id}/messages` | 会话历史（含来源、思维链、我的评价） |
| DELETE | `/api/conversations/{id}` | 删除会话，**连带**它的消息、反馈与链路 |
| GET | `/api/memory` | 查看长期记忆（按 `user_id` 隔离） |
| GET | `/api/memory/users` | 列出有长期记忆的用户及条数 |
| DELETE | `/api/memory/{id}` | 删除单条长期记忆 |
| POST | `/api/feedback` | 点赞 / 点踩（同一条回答重复提交会覆盖） |
| DELETE | `/api/feedback/{message_id}` | 取消评价 |
| GET | `/api/feedback/summary` | 满意度概览（赞 / 踩 / 满意率） |
| GET | `/api/feedback/badcases` | 点踩记录列表 |
| GET | `/api/feedback/export` | **把点踩记录导出成评测集条目** |
| GET | `/api/trace/overview` | 平均 / P50 / P95 耗时、拒答率、各阶段平均耗时 |
| GET | `/api/trace` | 最近的问答链路列表（可只看被拒的） |
| GET | `/api/trace/{trace_id}` | 单条链路详情：改写前后、两路召回、重排顺序、工具调用 |
| GET | `/api/rulebook/stats` | 规则库统计（规则数、流程数、按种类分布、抽取器版本） |
| POST | `/api/rulebook/rebuild` | 强制重建规则库（入库新文档后可手动触发） |
| GET | `/api/rulebook/rules` | 浏览 / 搜索规则（`q` 走 BM25，可再按 `kind` 筛） |
| POST | `/api/rulebook/diagnose` | **情境诊断**：描述自己的情况 → 相关条文 + 数值比对提示 |
| GET | `/api/rulebook/procedures` | 办事流程列表 / 搜索（不给 `q` 时按置信度排序） |
| GET | `/api/rulebook/procedures/{id}` | 单条流程详情（步骤、材料、原文出处） |
| GET | `/api/stats` | 概览统计 |
| GET | `/docs` | Swagger 交互文档 |

## 项目结构

```
school-handbook-agent/
├── backend/
│   ├── app/
│   │   ├── main.py                  # 应用入口、CORS、静态资源、启动横幅
│   │   ├── config.py                # 全部可调参数（业务代码禁止硬编码）
│   │   ├── schemas.py               # Pydantic 模型
│   │   ├── db/session.py            # 异步引擎、会话、建表与轻量迁移
│   │   ├── models/                  # 六张表：文档、会话、消息、长期记忆、反馈、链路
│   │   ├── routers/                 # chat / knowledge / history / observability / rulebook
│   │   └── services/
│   │       ├── tokenizer.py         # 中文二元组分词 + 数字归一化
│   │       ├── bm25.py              # 手写 Okapi BM25（建索引 / 检索 / 落盘）
│   │       ├── hybrid.py            # 双路召回 + RRF 融合 + 索引生命周期
│   │       ├── ai_ops.py            # 改写 / 重排 / 自省判定 / 证据拼装
│   │       ├── guard.py             # 注入扫描与证据边界
│   │       ├── rulebook.py          # 从条文抽「条件→后果」规则与办事流程
│   │       ├── rag_service.py       # 主编排：一次问答的完整流程
│   │       ├── trace_service.py     # 链路采集（内存累积，收尾一次落库）
│   │       ├── observability_service.py  # 反馈读写、badcase 导出、链路聚合
│   │       └── ...                  # 解析、向量化、检索、生成、记忆
│   ├── data/                        # sqlite / chroma / uploads（运行时生成，不入库）
│   ├── static/                      # 免安装前端：index.html + styles.css + app.js
│   │   └── vendor/                  # Vue / Element Plus / marked（本地文件，刻意入库）
│   ├── tests/                       # 自测、评测、静态检查、截图、演示数据整理
│   ├── ingest.py                    # 批量导入脚本
│   ├── run.py                       # 启动入口
│   └── .env                         # 密钥与参数（不提交到 Git）
├── docs/
│   ├── DESIGN.md                    # 设计取舍与踩坑详解
│   └── screenshots/                 # README 里用到的界面截图
├── README.md                         # 本文件
├── PYCHARM.md                        # PyCharm 运行配置说明
└── .gitignore
```

## 技术栈

| 层 | 选型 | 说明 |
| --- | --- | --- |
| 后端 | FastAPI | 异步接口，自带 OpenAPI 文档 |
| ORM | SQLAlchemy 2.0 async + aiosqlite | 换数据库只需改连接串 |
| 关系库 | SQLite | 单机零配置，文件位于 `backend/data/school.db` |
| 向量库 | ChromaDB | 本地持久化，支持 metadata 过滤与删除 |
| Embedding | BGE-small-zh-v1.5（本地） | 512 维，中文优化，离线加载 |
| 关键词检索 | 自实现 Okapi BM25 | 零依赖，中文二元组分词 + 数字归一化 |
| 融合 | 自实现 RRF | 只看排名，绕开 BM25 分数无上界的问题 |
| 大模型 | DeepSeek（`deepseek-flash`） | OpenAI 兼容接口，未配置 Key 时降级为演示模式 |
| 文本切分 | langchain-text-splitters | 递归字符切分，保留章节语义 |
| 前端 | Vue 3 + Element Plus | 本地 vendor 文件加载，免 npm 构建直接运行 |

**依赖原则：零新增依赖。** 上表里 BM25、RRF、中文归一化、重排、注入扫描、
链路追踪、规则抽取全部是手写的 —— 不是造轮子，是因为这些算法的核心只有几十行，
自己写反而更好控制、更好解释、也更容易在面试里讲清楚。

## 评测与自测

### 评测

评测集 `tests/eval_dataset.json`，25 条用例，**人工逐条核对过知识库原文**，
覆盖单跳事实、精确条款号、多跳比较、指代消解、该拒答、闲聊、越界、注入攻击八类。

```bash
# 纯检索消融：不启动服务、不调大模型，秒级出结果
D:\python\python.exe tests\eval_rag.py --retrieval-only

# 端到端评测：需要服务在跑（会真实调用大模型）
D:\python\python.exe tests\eval_rag.py --save eval_result_full.json
D:\python\python.exe tests\eval_rag.py --baseline eval_result_full.json   # 与基线对比
```

25 条全跑一遍的实测结果：

| 指标 | 通过 | 说明 |
| --- | --- | --- |
| 分流准确率 | **25/25 (100%)** | 闲聊 / 越界不该检索，规章问题必须检索 |
| 检索召回率 Recall@5 | **15/15 (100%)** | 标准答案条款落在前 5 条来源里 |
| 拒答判定准确率 | **25/25 (100%)** | 该拒的拒了，不该拒的没拒 |
| 答案要点覆盖 | **29/29 (100%)** | 整题要点全中 16/16 |
| 安全约束通过率 | **1/1** | 注入攻击下未泄露系统提示 |
| 回答长度合规率 | **4/4** | 闲聊 / 越界没长篇大论 |

**端到端延迟：平均 6717 ms / P50 5320 ms / P95 18201 ms**

差异全在「走没走 Agent 循环」上：单跳类平均 4993 ms，而多跳类 17654 ms ——
这正是复杂度路由存在的理由。

### 自测

```bash
D:\python\python.exe tests\check_undefined.py          # 静态检查：有没有「调用未定义函数」
D:\python\python.exe tests\verify_observability.py     # 反馈闭环 + 链路追踪 + 引用溯源，33 项断言
D:\python\python.exe tests\smoke_test.py               # 端到端回归
D:\python\python.exe tests\test_incremental_dedup.py   # 增量去重专项
D:\python\python.exe tests\calibrate_dedup.py          # 看语义去重的相似度分布
D:\python\python.exe tests\seed_demo.py                # 演示数据整理（去重 + 统一归属，默认只报告）
node tests\check_frontend.js                          # 前端静态检查（改完界面跑一下）
node tests\vendor_diag.js                             # 第三方库加载诊断（页面白屏/出花括号时跑）

# 截图自查：静态检查发现不了 CSS 尺寸失效，改完界面必须看一眼。
# 第一条保持运行，第二条在另一个终端执行（--js 可以注入点击后再截图）
D:\python\python.exe tests\ui_shot.py --serve
D:\python\python.exe tests\ui_shot.py --url "http://127.0.0.1:8000/#trace" --out shot.png
```

| 脚本 | 管什么 |
| --- | --- |
| `check_undefined.py` | 用标准库 `symtable` 找「函数体里还留着调用、但函数定义没了」的孤儿调用。**源文件曾被编辑器抢占导致写入回滚**，这类问题 import 阶段完全不报错，直到那行执行才抛 `NameError` —— 线上表现就是「服务正常启动，一问就崩」 |
| `verify_observability.py` | 逐条对账「数据有没有真的落库、下次读出来还对不对」，含引用可验证性（拿来源里的 `chunk_index` 回查知识库，确认定位到的原文与来源一致） |
| `smoke_test.py` | 健康检查、知识库增删、流式问答、拒答逻辑、短期/长期记忆、用户隔离、会话删除连带清理、前端静态资源可达性。**跑完自动清掉自己产生的数据** |
| `test_incremental_dedup.py` | 在**独立的临时向量库**里跑，不污染正式数据 |
| `check_frontend.js` | 不需要浏览器、不需要 npm：检查标签配对、模板调用的方法与变量是否都已暴露、`#app` 之外有没有残留 `{{ }}` |
| `ui_shot.py` | 补的是上面都覆盖不到的一块：**布局和尺寸**。走 CDP 驱动系统自带的 Edge（不下载 Chromium），能注入 JS 做交互后再截图 |
| `seed_demo.py` | 给「要拿给人看」的场景做收尾：按标题去重（同一条问题会被反复问）、统一散落的 `user_id`。**默认只打印会保留什么、会删什么**，加 `--apply` 才执行且先自动备份数据库 |

## 常见问题

| 现象 | 处理 |
| --- | --- |
| 启动报端口被占用 | 改 `.env` 里的 `PORT` |
| 提示找不到 Embedding 模型 | 检查 `.env` 的 `EMBEDDING_MODEL_PATH` 是否指向本地模型目录 |
| 回答为空 / 只有 `[模型本次没有返回正文]` | `LLM_MAX_TOKENS` 太小，思维链把额度吃光了。**调到 8192** 后重启 |
| 回答里出现 `[调用大模型失败] Connection error.` | 多半是进程继承了残留的代理变量。本项目默认已忽略（`LLM_USE_ENV_PROXY=false`）；也可先 `curl https://api.deepseek.com/v1/models` 确认网络通（返回 401 就是通的） |
| 该拒答的怪问题被放行了 | 会先走自省判定；若仍放行，说明检索到的资料「看起来相关但撑不起答案」，模型会答「手册中未找到」。想更严格可下调 `SIMILARITY_THRESHOLD` |
| 新传的规章抽不出规则 / 规则诊断没变化 | 派生缓存（BM25 + 规则库）按「条数是否一致」自愈，正常情况下不必手动干预；实在没刷新可以调一次 `POST /api/rulebook/rebuild` 或重启 |
| 上传后文档列表里出现重复文档 | 精确去重是**块级**的：整份都是新内容才会建文档记录。有重叠内容时属正常，看「跳过重复」那一列 |
| 想关掉某一步 | `.env` 里有 `HYBRID_ENABLED` / `RERANK_ENABLED` / `AGENT_ENABLED` / `QUERY_REWRITE_ENABLED` / `GUARD_ENABLED` / `TRACE_ENABLED`，逐项可关 —— 正好用来做消融 |
| 页面白屏或显示原始 `{{ }}` | 跑 `node tests\vendor_diag.js` 排除第三方库没加载的问题 |

---

> 想看「为什么这么设计、每一步踩过哪些坑」？
> 完整版在 **[docs/DESIGN.md](docs/DESIGN.md)** —— 包括混合检索的消融数据、
> 复杂度路由的耗时对比、三类静默失败的排查过程、评测方法论。
