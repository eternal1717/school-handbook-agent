# 在 PyCharm 里运行本项目

> 项目路径：`R:\xiangmu\PythonProject\school-handbook-agent`
> 已经预置好工程配置，按下面 3 步走即可。

---

## 第 1 步：用 PyCharm 打开项目

**File → Open**（或启动页的 **Open**），选择这个目录：

```
R:\xiangmu\PythonProject\school-handbook-agent
```

> 注意是 `school-handbook-agent`，**不要**选它上面的 `PythonProject`，
> 否则左侧项目树里看不到本项目。

打开后左侧应该能看到：

```
school-handbook-agent
├── backend/          ← 后端（FastAPI + SQLite + ChromaDB）
│   ├── app/          服务层：routers / models / services
│   │                 services 里几个值钱的：tokenizer（中文分词）、bm25、
│   │                 hybrid（双路召回+RRF）、ai_ops（改写/重排/自省）、
│   │                 guard（注入防护）、rag_service（主编排）、trace_service（链路）
│   ├── static/       前端（Vue3 + Element Plus）
│   ├── data/         school.db + chroma + uploads（运行时生成，不入 Git）
│   ├── run.py        启动入口
│   ├── ingest.py     批量导入手册
│   └── tests/        自测、评测、静态检查脚本
├── README.md
├── PYCHARM.md        本文件
├── .gitignore
└── .gitattributes
```

---

## 第 2 步：确认 Python 解释器

**File → Settings → Project: school-handbook-agent → Python Interpreter**

要选中这个解释器：

```
D:\python\python.exe      （Python 3.14，项目所有依赖都在这里）
```

如果下拉框里没有，点右上角 **Add Interpreter → Add Local Interpreter →
Select existing → 浏览到 `D:\python\python.exe`**。

> ⚠️ **不要选 "Virtualenv Environment / 新建虚拟环境"**——
> 新建环境是空的，会重新下载 torch（2～3 GB），而且往 C 盘写。
> 用现成的 `D:\python` 就好，chromadb、sentence-transformers、fastapi 全都在里面。

> ⚠️ 如果 PyCharm 顶部弹出 **"Package requirements 'requirements.txt' are not satisfied"**，
> **点忽略 / Don't ask again**，不要点 Install。依赖已经装齐了，点了会重复下载大包。

---

## 第 3 步：点绿三角运行

右上角运行配置下拉框里已经有 3 个现成配置（我预置好了，不用自己建）：

| 配置名 | 作用 |
|---|---|
| **启动服务 (run.py)** | 启动后端 + 前端，最常用 |
| **导入手册 (ingest.py)** | 把文档灌进向量库（需在 Parameters 里填文件路径） |
| **自测 (smoke_test.py)** | 跑 34 项端到端回归测试 |

选 **"启动服务 (run.py)"** → 点绿色三角 ▶（或按 `Shift+F10`）。

Run 窗口出现下面这行就说明起来了：

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

> ⏱ **首次启动要等 30～60 秒，别以为卡死了。**
> 启动时要加载 BGE 中文词向量模型（本地 `R:\models\bge-small-zh-v1.5`）
> 并连接 ChromaDB，这段没有日志输出，窗口会"静着"。实测约 40 秒。
> 看到 `Uvicorn running on ...` 才算好。

### 然后打开浏览器

```
http://127.0.0.1:8000/
```

就能看到聊天界面了。

---

## 四个页面分别看什么

| 左侧菜单 | 内容 |
|---|---|
| **问答** | 主聊天页；左侧是会话列表，右侧是对话区。回答上方可展开「思考过程」，下方是阶段流水条、问题理解卡片和引用来源（每条来源都标了是向量命中还是关键词命中，可点「定位原文」跳回知识库原文） |
| **知识库** | 已入库文档表格，每行可点「看分块」查看该文档切成了哪些知识块 |
| **记忆** | 长期记忆（跨会话记住的内容），可切换用户查看 |
| **链路** | 可观测看板：平均/P50/P95 耗时、拒答率、各阶段平均耗时柱状图、最近问答列表，点任意一行可打开详情抽屉，看改写前后、两路召回、重排顺序、工具调用与 token 消耗 |

### 顺手可以跑的几个脚本

在 PyCharm 底部 Terminal 里（工作目录已经在 `backend/`）：

```
D:\python\python.exe tests\check_undefined.py          静态检查：有没有「调用未定义函数」
D:\python\python.exe tests\verify_observability.py     反馈闭环 + 链路追踪 + 引用溯源（31 项断言）
D:\python\python.exe tests\eval_rag.py --retrieval-only 检索消融：混合检索 vs 纯向量（不调大模型，秒级）
D:\python\python.exe tests\eval_rag.py                  端到端评测（24 条用例，会真实调模型，较慢）
```

---

## 常见问题

**Q：右上角的运行按钮是灰色的 / 提示 "SDK is not defined"**

说明项目绑定的解释器在 PyCharm 里没找到。手动绑一次就好：

**File → Settings → Project: school-handbook-agent → Python Interpreter
→ 下拉选 `Python 3.14`（路径 `D:\python\python.exe`）**

选完运行按钮立刻变绿。

**Q：Run 窗口报 `[Errno 10048] address already in use`（端口被占用）**
说明 8000 端口上已经有一个服务在跑。要么直接用它，要么先停掉旧进程：
在 PyCharm 底部 **Terminal** 里执行：

```
netstat -ano | findstr :8000
taskkill /F /PID <上面查到的PID>
```

**Q：页面上写着「演示模式」**
正常。当前 `backend\.env` 里没填 `DEEPSEEK_API_KEY`，所以回答由检索到的原文片段拼成，
链路完整但不调用大模型。填上 Key 重启即可获得完整自然语言回答：

```
# backend\.env
DEEPSEEK_API_KEY=sk-xxxxxxxx
```

**Q：回答里出现 `[调用大模型失败] Connection error.`**

代码已修复（原因是进程继承了残留的代理环境变量），**但要重启服务才会生效**：
按 `Ctrl+F5` 重新运行即可。若仍然报错，先确认网络：
在 Terminal 里执行 `curl https://api.deepseek.com/v1/models`，
返回 `401` 说明网络是通的（只是没带 Key）。

**Q：回答是空的**

`backend\.env` 里把 `LLM_MAX_TOKENS` 调到 `4096` 后重启。
原因：`deepseek-flash` 是推理模型，思维链会占用 `max_tokens` 额度，给少了正文就被挤空。

**Q：改了代码要重启吗**
要。`run.py` 里 `reload=False`，改完按 `Ctrl+F5` 重新运行即可。
（想改 py 自动重载，可把 `run.py` 里 `reload=False` 改成 `True`，但首次启动会慢一点。）

**Q：数据存在哪**
- 数据库：`backend\data\school.db`（可用 SQLiteStudio 打开看表）
- 向量库：`backend\data\chroma\`
- 上传的原始文件：`backend\data\uploads\`
