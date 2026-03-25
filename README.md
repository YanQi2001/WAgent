# WAgent — AI 面试智能体系统

基于 Harness-Centric 架构的 AI 面试官系统。通过多智能体协作（LangGraph）、混合检索 RAG（BM25 + Dense + Cross-Encoder Reranker）、MCP 协议外部数据源和动态知识库管理，实现结构化的技术面试模拟与知识问答。

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    CLI（Typer + Rich）                    │
│  wagent start / interview / qa / prepare / topics / ... │
├─────────────────────────────────────────────────────────┤
│                 智能体执行引擎（Harness）                  │
│  中间件管道 · Token 预算 · 上下文压缩 · 护栏 · 追踪       │
├──────────────┬──────────────┬───────────┬───────────────┤
│   路由智能体  │   面试官      │   评审官   │  意图分类器    │
│  （简历 →     │  （RAG 增强   │ （LLM-as-  │ （面试 /       │
│    面试计划） │    问答循环） │   Judge）  │  问答 / 闲聊） │
├──────────────┴──────────────┴───────────┴───────────────┤
│             LangGraph 状态机（内循环）                     │
├─────────────────────────────────────────────────────────┤
│                      RAG 引擎                            │
│  语义切分 · 上下文检索（Anthropic 方法）                   │
│  混合搜索：BM25 + Dense + RRF + Cross-Encoder Reranker   │
│  Qdrant 向量数据库 · 动态 Topic 体系                      │
├─────────────────────────────────────────────────────────┤
│                     MCP 数据源                            │
│  小红书搜索（含图片 OCR） · Bing 搜索 · PDF 自动入库       │
├─────────────────────────────────────────────────────────┤
│              知识更新器 + APScheduler                      │
│  GAP 分析 · 质量过滤 · 语义去重 · Topic 自动审查           │
└─────────────────────────────────────────────────────────┘
```

## 核心特性

- **Harness-Centric 架构** — 外层控制循环管理中间件、Token 预算、上下文压缩、护栏和追踪；内层推理委托给 LangGraph
- **上下文检索（Contextual Retrieval）** — 每个 chunk 在向量化前由 LLM 生成上下文描述，提升检索准确率
- **混合检索** — BM25 稀疏检索 + 稠密向量检索 → 互惠排名融合（RRF）→ Cross-Encoder 精排（`bge-reranker-v2-m3`）
- **动态 Topic 体系** — LLM 自动从未分类 chunk 中提出新 topic；支持手动添加和重分类
- **多智能体协作（LangGraph）** — Router / Interviewer / Judge 节点，类型化状态转换
- **MCP 协议** — 标准化接口对接小红书、Bing 及未来数据源
- **图片 OCR** — RapidOCR（ONNX，本地 CPU 推理）提取小红书帖子图片中的文字，与正文合并后提取 QA 对
- **双模型分级调用** — 简单任务（意图分类、元数据生成等）走廉价快速模型，复杂任务（QA 提取、面试对话等）走强推理模型，降低成本
- **CJK 终端输入** — 基于 prompt_toolkit 的输入方案，支持中文编辑、光标移动和历史记录
- **三级上下文压缩** — 原始 → 压缩 → 摘要，逐步释放 Token 空间

## 环境要求

- **操作系统**：Linux
- **Python**：>= 3.10
- **包管理器**：Conda（推荐）
- **LLM API**：任意 OpenAI 兼容 API（DeepSeek、OpenAI、硅基流动、NVIDIA、Moonshot 等）
- **Docker**：可选，用于 Qdrant Server 模式（不安装则使用本地文件模式）

## 安装

```bash
# 1. 创建 conda 环境
conda create -n wagent python=3.10 -y
conda activate wagent

# 2. 安装核心依赖（含 RapidOCR 图片文字提取）
cd WAgent
pip install -e ".[dev]"

# 3.（可选）安装小红书浏览器自动化依赖
pip install -e ".[browser]"
playwright install chromium

# 4.（可选）安装小红书工具的 Python 依赖
pip install requests websockets
# 或在 tools/xiaohongshu-skills/ 下执行 uv sync

# 5. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY、LLM_BASE_URL、LLM_MODEL
```

## 支持的 LLM 供应商

WAgent 兼容所有 OpenAI 格式的 API，在 `.env` 中切换即可：

| 供应商 | `LLM_BASE_URL` | `LLM_MODEL`（示例） |
|--------|----------------|---------------------|
| **DeepSeek** | `https://api.deepseek.com` | `deepseek-chat` |
| **OpenAI** | `https://api.openai.com/v1` | `gpt-4o` |
| **硅基流动** | `https://api.siliconflow.cn/v1` | `deepseek-ai/DeepSeek-V3` |
| **NVIDIA** | `https://inference-api.nvidia.com/v1` | `gcp/google/gemini-3-pro` |
| **Moonshot（Kimi）** | `https://api.moonshot.cn/v1` | `moonshot-v1-auto` |
| **MiniMax** | `https://api.minimax.chat/v1` | `MiniMax-Text-01` |

### 双模型分级（可选）

可以配置一个廉价快速模型处理简单任务（意图分类、元数据生成、GAP 分析等），降低 Token 成本：

```bash
# .env 中新增以下配置（未配置则所有任务走主模型）
LLM_FAST_API_KEY=your-siliconflow-key
LLM_FAST_BASE_URL=https://api.siliconflow.cn/v1
LLM_FAST_MODEL=deepseek-ai/DeepSeek-V3
```

## 环境变量说明

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_API_KEY` | （必填） | LLM 供应商 API key |
| `LLM_BASE_URL` | `https://api.deepseek.com` | OpenAI 兼容 API 地址 |
| `LLM_MODEL` | `deepseek-chat` | 主模型名称 |
| `LLM_FAST_API_KEY` | （可选） | 快速模型 API key |
| `LLM_FAST_BASE_URL` | （可选） | 快速模型 API 地址 |
| `LLM_FAST_MODEL` | （可选） | 快速模型名称 |
| `TOKEN_BUDGET` | `100000` | 单次面试 Token 预算 |
| `QDRANT_PATH` | `./data/qdrant_db` | 本地 Qdrant 存储路径 |
| `QDRANT_URL` | （可选） | Qdrant Server 地址（Docker 模式） |
| `KNOWLEDGE_UPDATE_INTERVAL` | `43200` | 自动更新间隔（秒，默认 12 小时） |

## 快速开始

```bash
# 验证 API 连通性
wagent ping

# 入库知识库文档（PDF/TXT）
wagent ingest --path data/documents

# 查看知识库 topic 分布
wagent topics

# 准备面试（完整流程）
wagent prepare resume.pdf

# 启动模拟面试
wagent interview resume.pdf

# 启动知识问答
wagent qa
```

## 详细用法

### 1. 构建知识库

将八股文档（PDF/TXT 格式）放入 `data/documents/` 目录，然后运行：

```bash
wagent ingest --path data/documents --source manual
```

后续添加新文档时，只需将新文件放入同一目录再次执行 `wagent ingest` 即可，已入库的文件会自动跳过（文件级去重），仅处理新增文件。

如果文档较大、处理时间较长，可使用后台模式：

```bash
wagent ingest --daemon
# 查看进度
tail -f logs/ingest.log
```

处理流程：PDF 解析 → 语义切分 → LLM 上下文描述 → topic / 难度分类 → 去重 → Qdrant 入库。

### 2. 面试准备

提供简历，系统自动分析知识覆盖情况：

```bash
wagent prepare resume.pdf
```

流程：
1. 解析简历，提取技能点并映射到 topic 体系
2. GAP 分析 — 对比知识库覆盖率与简历技能
3. 搜索补充 — 可选联网搜索小红书 / Bing 补充薄弱 topic
4. Topic 审查 — LLM 自动提议新 topic
5. 准备报告 — topic 分布表及简历相关度

### 3. 模拟面试

```bash
wagent interview resume.pdf
```

系统将：
- 生成面试计划（70% 简历驱动，30% 随机 topic）
- 基于 RAG 知识库上下文提问
- 实时评估每个回答
- 面试结束后生成详细评分报告（LLM-as-Judge）

面试中输入 `/end` 可提前结束。

### 4. 知识问答

```bash
wagent qa
```

自由提问技术问题。系统优先从知识库检索；如果知识库未命中，可选择联网搜索（Bing + 小红书）补充资料；即使无参考资料，模型也会用自身知识回答并声明来源。输入 `/end` 退出。

### 5. 智能入口

```bash
wagent start
```

系统使用 LLM 意图分类，自动识别你想进行面试、问答还是闲聊，然后路由到对应模式。

### 6. 知识库管理

#### Topic 体系

Topic 通过 `data/topic_taxonomy.json` 动态管理：

```bash
# 查看当前 topic 及分布
wagent topics

# LLM 自动审查：从未分类 chunk 中提议新 topic
wagent topics --review

# 手动添加 topic
wagent topics --add "knowledge_graph"

# 重分类所有 "general" 类 chunk
wagent topics --reclassify
```

#### 清除数据

```bash
# 预览将删除的内容
wagent purge --source crawled --dry-run

# 删除所有爬取数据（保留手动入库的 PDF）
wagent purge --source crawled
```

### 7. 后台自动更新

启动后台调度器，系统将定期自动更新知识库：

```bash
wagent serve              # 默认守护进程模式：自动 fork 到后台，可关闭终端
wagent serve --foreground # 前台模式：保持在当前终端，Ctrl+C 停止
```

启动后输出 PID 和日志路径：

```
WAgent 调度器已启动 (PID=12345)
日志: /path/to/WAgent/logs/serve.log
停止: wagent stop-serve
```

查看日志和停止：

```bash
tail -f logs/serve.log     # 查看调度器日志
wagent stop-serve          # 停止调度器
```

每次自动更新的完整流程：
1. 消费待搜索的知识点队列（来自问答中未命中的提问）
2. GAP 分析 → 生成搜索关键词
3. 随机抽取 2-3 个非薄弱 topic 补充搜索（保持知识库全面更新）
4. 小红书 + Bing 联网搜索
5. LLM 质量过滤（提取结构化 QA，过滤广告 / 无关内容）
6. 对无答案但有价值的问题，LLM 自行补充专业回答
7. 语义去重 → 入库
8. Topic 体系自动审查 + 重分类

#### 手动触发更新

```bash
wagent update-kb
```

## 项目结构

```
WAgent/
├── src/wagent/                  # 核心 Python 包
│   ├── agents/                  # 多智能体模块
│   │   ├── graph.py             # LangGraph 状态机
│   │   ├── intent.py            # 意图分类器
│   │   ├── interviewer.py       # 面试官智能体
│   │   ├── judge.py             # LLM-as-Judge 评审官
│   │   ├── router.py            # 简历 → 面试计划
│   │   └── schemas.py           # Pydantic 数据模型
│   ├── cli/                     # CLI 入口
│   │   ├── main.py              # Typer 应用（所有命令）
│   │   ├── interview_session.py # 面试会话
│   │   ├── qa_session.py        # 问答会话
│   │   ├── prompt_utils.py      # CJK 安全终端输入（prompt_toolkit）
│   │   ├── smart_prompt.py      # LLM 自然语言确认
│   │   └── qdrant_docker.py     # Qdrant Docker 容器管理
│   ├── harness/                 # 智能体执行引擎
│   │   ├── harness.py           # 核心引擎循环
│   │   ├── budget.py            # Token 预算追踪
│   │   ├── context.py           # 三级上下文压缩
│   │   ├── middleware.py        # 前 / 后处理钩子
│   │   ├── state.py             # 面试状态模型
│   │   ├── tools.py             # 工具注册 + 角色 ACL
│   │   └── tracer.py            # JSONL 追踪日志
│   ├── mcp_servers/             # MCP 数据源服务
│   │   ├── xiaohongshu_server.py  # 小红书搜索 + 图片 OCR（RapidOCR）
│   │   ├── bing_server.py       # Bing 搜索
│   │   ├── pdf_downloader.py    # PDF 下载器
│   │   └── updater.py           # 知识库更新器
│   ├── rag/                     # RAG 引擎
│   │   ├── chunking.py          # 语义切分 + 上下文检索
│   │   ├── embeddings.py        # 嵌入模型（bge-large-zh-v1.5）
│   │   ├── ingest.py            # 文档入库管道
│   │   ├── retriever.py         # 混合检索：BM25 + Dense + RRF + Reranker
│   │   └── store.py             # Qdrant 操作
│   ├── config.py                # 配置 + 动态 Topic 体系
│   ├── llm.py                   # 统一 LLM 实例化（双模型分级）
│   ├── scheduler.py             # APScheduler 定时任务
│   └── utils.py                 # 共享工具函数
├── tools/
│   └── xiaohongshu-skills/      # 小红书浏览器自动化工具
│       ├── scripts/             # CLI 入口和爬虫逻辑
│       └── skills/              # 技能配置
├── data/
│   ├── documents/               # 放置 PDF/TXT 知识文档
│   ├── topic_taxonomy.json      # 动态 Topic 体系
│   └── qdrant_db/               # 本地向量数据库（自动创建）
├── tests/                       # 测试用例
├── pyproject.toml               # 项目配置和依赖声明
├── .env.example                 # 环境变量模板
├── LICENSE                      # MIT 许可证
└── .gitignore
```

## 核心技术细节

### 上下文检索（Contextual Retrieval）

借鉴 Anthropic 的方法，在向量化之前为每个 chunk 生成一段简短的上下文描述并前置拼接，让检索时能更准确地匹配语义。

### 混合检索管线

1. **BM25 稀疏检索** — 关键词精确匹配
2. **Dense 向量检索** — `BAAI/bge-large-zh-v1.5`（1024 维）语义相似度
3. **互惠排名融合（RRF）** — 合并两路结果
4. **Cross-Encoder 精排** — `bge-reranker-v2-m3` 对候选重排序

### 图片 OCR 管线

小红书帖子中大量内容以图片形式呈现。系统使用 RapidOCR（基于 ONNX Runtime，纯 CPU 推理，无需 GPU）自动提取图片中的文字，与帖子正文合并后再交给 LLM 提取结构化 QA 对。

### 双模型分级

通过 `get_llm(tier="fast"|"strong")` 工厂函数实现：

- **Fast tier**（如 DeepSeek-V3 @ 硅基流动）：意图分类、元数据生成、GAP 分析等短输出任务
- **Strong tier**（如 Gemini 3 Pro @ NVIDIA）：QA 提取、面试对话、评审打分等深度推理任务

未配置 Fast tier 时自动回退到 Strong tier，保持向后兼容。

## 许可证

[MIT](LICENSE)
