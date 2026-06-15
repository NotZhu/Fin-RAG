# FinRAG

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB)](#runtime-stack)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688)](#runtime-stack)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.12+-4B5563)](#runtime-stack)
[![Milvus](https://img.shields.io/badge/Milvus-2.4-00A1EA)](#runtime-stack)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791)](#runtime-stack)
[![Redis](https://img.shields.io/badge/Redis-7-DC382D)](#runtime-stack)
[![React 18](https://img.shields.io/badge/React-18-61DAFB)](#runtime-stack)
[![Vite 5](https://img.shields.io/badge/Vite-5-646CFF)](#runtime-stack)
[![pytest](https://img.shields.io/badge/pytest-enabled-0A9EDC)](#testing)
[![vitest](https://img.shields.io/badge/vitest-enabled-6E9F18)](#testing)

FinRAG 是一个面向金融合规、业务制度、产品说明与投研资料检索的高可靠 RAG 问答系统。系统聚焦金融机构内部资料查询、制度条款定位、业务流程问答和资料溯源，支持多格式文档接入、混合检索、证据窗口回填、可信生成、来源追踪与评估闭环。

> 项目定位：资料库问答与资料检索，不提供实时行情分析、投资推荐或交易决策。

## Capabilities

| 模块       | 能力                                                                                 |
| ---------- | ------------------------------------------------------------------------------------ |
| 文档接入   | 支持 PDF / Markdown / TXT / DOCX，提供上传、列表、删除、重建索引等文档生命周期接口   |
| 文档建模   | 采用简洁可验证元数据、内容 hash 去重、稳定 document/chunk 标识与层级引用 metadata    |
| 索引构建   | leaf nodes 写入向量索引与 BM25 稀疏索引，全部层级节点写入 PostgreSQL NodeStore       |
| 检索增强   | 支持 Vector、BM25、Hybrid、Hybrid + Rerank、QueryFusion 融合、低分过滤与资料库过滤   |
| 可信生成   | 使用 Grounded Answer 约束回答基于证据生成，资料不足时拒答，并保留非投资建议边界      |
| 可观测性   | 通过 SSE 返回 token、sources、可选 trace 和关键链路事件，记录路由、来源节点与耗时    |
| 质量评估   | 支持 hit@k、MRR、keyword coverage，并预留 Ragas 指标评估回答忠实性与上下文质量       |
| Web 工作台 | 提供 FinRAG 品牌工作台、金融资料库上传、文档状态、问答、来源引用和 trace 展示页面    |

## Architecture

```text
                    ┌────────────────────────┐
                    │ React Web Workbench    │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │ FastAPI Service         │
                    │ documents / ask / stats │
                    └───────────┬────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────▼────────┐     ┌────────▼────────┐     ┌─────────▼────────┐
│ Ingestion      │     │ Indexing        │     │ Retrieval        │
│parsers/registry│     │hierarchical node│     │llamaindex_router│
└───────┬────────┘     └────────┬────────┘     └─────────┬────────┘
        │                       │                        │
        │              ┌────────▼────────┐     ┌─────────▼────────┐
        │              │ Milvus Vector DB│     │ Native Hybrid    │
        │              └─────────────────┘     └─────────┬────────┘
        │                                                │
        │                                      ┌─────────▼────────┐
        └──────────────────────────────────────► LlamaIndex Auto  │
                                               │ Merge            │
                                               └─────────┬────────┘
                                                         │
                                               ┌─────────▼────────┐
                                               │ Grounded Answer  │
                                               │ Qwen / DashScope │
                                               └──────────────────┘
```

## RAG Pipeline

1. **Document Ingestion**
   上传文件后计算 `content_hash`，重复内容直接复用文档状态；新内容进入解析、注册和索引流程。文档状态包括 `uploaded`、`parsing`、`indexed`、`failed`、`deleted`。
2. **Metadata Normalization**
   系统不假设真实金融文档一定具备规范标题或目录。解析阶段只保留检索和溯源必需字段：`knowledge_base_id`、`document_id`、`filename`、`file_type`、`page_number`；`source_path` 与 `content_hash` 仅保存在内部文档注册表。公共文档列表暴露 `upload_time`，用于工作台展示上传或索引登记时间。
3. **Hierarchical Chunking**
   文档使用 LlamaIndex `HierarchicalNodeParser.from_defaults(chunk_sizes=[1200, 600, 300])` 构建层级节点；默认 `RAG_CHUNK_SIZE=300`、`RAG_CHUNK_OVERLAP=60`。父级节点写入 docstore/storage context，`get_leaf_nodes()` 输出的 leaf nodes 进入检索索引。每个 leaf node 继续携带 `chunk_id`、`parent_chunk_id`、`root_chunk_id`、`chunk_level` 和 `chunk_idx`，用于稳定溯源。
4. **LlamaIndex Router Retrieval**
   对外只保留 `llamaindex_router`。召回阶段使用 `MilvusNativeHybridRetriever(BaseRetriever)` 包装 Milvus 原生 dense+sparse hybrid，不接入内存 BM25 或 LlamaIndex `QueryFusionRetriever`。
5. **Auto Merge Context**
   Milvus 只保存并召回 leaf vectors，随后进入 LlamaIndex `AutoMergingRetriever(simple_ratio_thresh=RAG_AUTO_MERGE_RATIO_THRESHOLD)`；父级节点从 PostgreSQL `finrag_chunks` docstore adapter 回源。Jina reranker 作为 LlamaIndex node postprocessor 可选接入，最后由 `SentenceAwareTokenBudgetPostprocessor` 按句子边界控制上下文预算。
6. **Grounded Generation**
   `knowledge -> RAG`，回答阶段要求模型仅依据检索证据作答，输出来源编号；当证据不足时明确说明无法从资料中确认。`general -> 普通 LLM`，未配置通用问答模型时返回明确提示。
7. **Trace & Evaluation**
   问答响应可返回 trace，记录 route type、固定检索策略、Milvus native hybrid、RRFRanker、top_k、candidate_k、auto-merge、来源节点和耗时；SSE 只发送真实执行过程中的 `analysis`、`route`、`source`、`token`、`done` 和 `error` 等事件，不再伪造旧 retrieval/rerank/grading/rewrite 事件。检索评估脚本直接评估 `milvus_hybrid_retriever`。

## Runtime Stack

- Python 3.11
- FastAPI、Uvicorn
- LlamaIndex 原生 Document / TextNode / NodeWithScore / RetrieverQueryEngine
- Milvus dense+sparse 向量存储与 LlamaIndex VectorStoreIndex
- BM25 sparse embedding、jieba
- Qwen / DashScope LlamaIndex 原生集成
- PostgreSQL 文档、节点、BM25 状态和索引 manifest 存储
- Redis 父级/根级节点回源缓存
- Ragas（`eval` extra）
- React 18、Vite 5、TypeScript、Vitest

## Project Structure

```text
apps/web/                 React 金融资料库工作台
src/finrag/application/   FinRAGSystem、启动、文档生命周期、问答 pipeline
src/finrag/api/           FastAPI app、routes、middleware、error handlers
src/finrag/core/          配置、LlamaIndex 类型 re-export、响应 schema
src/finrag/storage/       PostgreSQL stores、Redis cache、DB helper、protocols
src/finrag/ingestion/     文档记录、元数据工具、多格式解析和 load documents
src/finrag/indexing/      层级节点、auto merge、Milvus index、BM25 sparse embedding
src/finrag/retrieval/     llamaindex_router / native hybrid / rerank
src/finrag/generation/    DashScope LLM 初始化
data/documents/           金融示例资料库与上传后的原始文档
datasets/eval/            检索与生成评估集
scripts/                  检索评估、Ragas 评估与维护脚本
tests/                    后端、检索、生成、API 与前端契约测试
storage/uploads/          运行时临时上传文件；索引状态保存在 PostgreSQL/Milvus/Redis
```

## Sample Dataset

`data/documents/` 提供一组合成金融业务资料，用于演示文档解析、混合检索、证据窗口、来源溯源和评估流程。样例覆盖反洗钱客户尽调、大额交易和可疑交易报告、对公账户开立、理财适当性销售、公司授信、操作风险、数据安全和新能源汽车产业链研究摘要。

这些文档参考公开监管规则和行业规范的业务主题生成，但不代表任何真实金融机构制度，不构成监管解释、投资建议、收益预测或交易决策依据。所有客户、产品、金额、编号和业务场景均为演示用途。

## Quick Start

### Backend

```bash
python -m pip install -e ".[dev]"
copy .env.example .env
docker compose up -d postgres redis etcd minio milvus
finrag rebuild
uvicorn finrag.api:app --host 127.0.0.1 --port 8000
```

`.env.example` 默认使用 DashScope embedding 模型，并关闭 rerank：

```env
RAG_EMBEDDING_MODEL=text-embedding-v4
RAG_RERANKER_PROVIDER=none
RAG_CHUNK_SIZE=300
RAG_CHUNK_OVERLAP=60
```

启动索引前需要在 `.env` 中配置 API key：

```env
DASHSCOPE_API_KEY=your_api_key
RAG_EMBEDDING_MODEL=text-embedding-v4
RAG_DATA_PATH=data/documents
RAG_KNOWLEDGE_BASE_ID=kb-finance
```

未配置 `DASHSCOPE_API_KEY` 时，索引初始化会 fail fast，避免生产环境静默使用本地兜底 embedding。需要二阶段精排时配置 Jina-compatible HTTP rerank：

```env
RAG_RERANKER_PROVIDER=jina
RAG_RERANKER_MODEL=jina-reranker-v2-base-multilingual
RAG_RERANKER_ENDPOINT=https://api.jina.ai/v1/rerank
RAG_RERANKER_API_KEY=your_rerank_key
```

### Frontend

```bash
cd apps/web
npm install
npm run dev
```

默认前端开发服务会连接本地 FastAPI 服务。当前工作台以 `FinRAG` 为页面主标题，提供资料库预热、文档上传并索引、文档状态、流式问答、来源引用和可选调试 trace 展示。生产构建：

```bash
npm run build
```

## API Overview

| Method     | Path                                 | Description                            |
| ---------- | ------------------------------------ | -------------------------------------- |
| `GET`    | `/health`                          | 服务健康检查                           |
| `GET`    | `/ready`                           | 查看资料库就绪状态                     |
| `POST`   | `/warmup`                          | 加载并预热默认资料库                   |
| `GET`    | `/documents`                       | 查看文档列表、状态和错误信息           |
| `POST`   | `/documents/upload`                | 上传并索引 PDF / Markdown / TXT / DOCX |
| `DELETE` | `/documents/{document_id}`         | 删除文档及相关索引状态                 |
| `POST`   | `/documents/{document_id}/reindex` | 重新解析并索引指定文档                 |
| `POST`   | `/ask`                             | 发起 SSE 流式资料库问答                |

### Ask Request

```json
{
  "question": "客户风险等级如何与产品风险等级匹配？",
  "knowledge_base_id": "kb-finance",
  "return_sources": true,
  "return_trace": true
}
```

### Ask Response

`/ask` 返回 `text/event-stream`。当前事件包括 `analysis`、`route`、`source`、`token`、`done` 和 `error`。当 `return_sources=false` 时不会发送 `source` 事件，最终响应中的 `sources` 也为空；当 `return_trace=true` 时，最终 `done` 事件的 `response.trace` 会包含调试信息。

```text
event: token
data: {"text":"客户风险等级应与产品风险等级匹配"}

event: done
data: {"response":{...},"final_decision":"generate"}
```

`done` 事件中的 `response` 结构如下：

```json
{
  "question": "客户风险等级如何与产品风险等级匹配？",
  "route_type": "knowledge",
  "retrieval_strategy": "llamaindex_router",
  "answer": "客户风险等级应与产品风险等级匹配，销售过程需完成适当性评估并留存相关记录。[1]",
  "sources": [
    {
      "source_id": 1,
      "filename": "理财产品风险揭示书.md",
      "page_number": null,
      "score": 0.91,
      "snippet": "客户风险等级应与产品风险等级相匹配..."
    }
  ],
  "trace": {
    "route_type": "knowledge",
    "retrieval_strategy": "llamaindex_router",
    "timings_ms": {
      "analysis": 2.1,
      "total": 861.1
    },
    "events": [
      {
        "stage": "route",
        "route_type": "knowledge",
        "selected_query_engine": "knowledge_router"
      }
    ],
    "source_count": 1,
    "reranker": { "provider": "none" },
    "auto_merge": { "simple_ratio_thresh": 0.5 },
    "final_decision": "generate"
  }
}
```

## Workbench Data Contract

当前前端工作台可直接使用以下后端字段：

- `/ready`：`ready`、`status`、`total_documents`、`total_chunks`、`last_error`
- `/documents`：`document_id`、`filename`、`file_type`、`knowledge_base_id`、`status`、`chunk_count`、`upload_time`、`last_error`
- `/ask`：SSE 事件流中的 `analysis`、`route`、`source`、`token`、`done`、`error`；`done.response.trace` 仅在 `return_trace=true` 时返回

## Evaluation

检索评估：

```bash
python -m scripts.evaluate_retrieval --json
```

Ragas 评估：

```bash
python -m pip install -e ".[eval]"
python -m scripts.evaluate_ragas datasets/eval/finance_ragas_eval_set.jsonl --json
```

检索评估关注“有没有召回正确文档”和排序质量；Ragas 评估关注生成答案是否忠实于上下文、回答是否相关、证据片段是否充分。Ragas 脚本支持读取已生成的 `answer/contexts/ground_truth` JSONL，也可以通过 `--generate` 调用 FinRAG 自动生成回答和上下文后再评估。

建议评估维度：

- `hit@1 / hit@k`：目标文档是否被召回
- `MRR`：目标文档排序质量
- `keyword coverage`：关键词覆盖程度
- `faithfulness`：回答是否忠实于证据
- `answer relevancy`：回答与问题的相关性
- `context precision / recall`：上下文质量

## Testing

```bash
uv run --with ruff ruff check --no-cache src scripts tests
uv run pytest -q
uv lock --check
cd apps/web
npm test
npm run build
```

## Configuration

| Environment Variable             | Default                      | Description                                                  |
| -------------------------------- | ---------------------------- | ------------------------------------------------------------ |
| `RAG_DATA_PATH`                | `data/documents`          | 默认资料库目录                                               |
| `RAG_UPLOAD_DIR`               | `storage/uploads`         | 上传文件临时保存目录                                         |
| `RAG_KNOWLEDGE_BASE_ID`        | `kb-finance`               | 默认资料库 ID                                                |
| `RAG_DATABASE_URL`             | `postgresql://.../finrag`  | PostgreSQL 连接串                                            |
| `RAG_REDIS_URL`                | `redis://localhost:6379/0` | Redis 缓存连接串                                             |
| `RAG_MILVUS_HOST`              | `localhost`                | Milvus 服务地址                                              |
| `RAG_MILVUS_PORT`              | `19530`                    | Milvus 服务端口                                              |
| `RAG_MILVUS_COLLECTION`        | `finrag_leaf_nodes`        | Milvus leaf node collection                                  |
| `RAG_EMBEDDING_MODEL`          | `text-embedding-v4`        | DashScope embedding 模型名                                  |
| `RAG_LLM_MODEL`                | `qwen-max`                 | 生成模型                                                     |
| `RAG_TOP_K`                    | `3`                        | 最终返回证据数量                                             |
| `RAG_RETRIEVAL_CANDIDATE_K`    | `10`                       | 初始召回候选数量                                             |
| `RAG_RRF_K`                    | `60`                       | Milvus hybrid RRFRanker 的 `k` 参数                          |
| `RAG_RETRIEVAL_STRATEGY`       | `llamaindex_router`        | 固定检索编排策略                                             |
| `RAG_LLAMAINDEX_INDEX_STORE_DIR` | `storage/llamaindex`      | LlamaIndex index metadata 本地目录；旧 `RAG_LLAMAINDEX_STORAGE_DIR` 兼容一个版本 |
| `RAG_SCORE_THRESHOLD`          | `0.0`                      | 检索候选分数过滤阈值                                         |
| `RAG_CHUNK_SIZE`               | `300`                      | 叶子节点大小；层级分块为 1200/600/300                       |
| `RAG_CHUNK_OVERLAP`            | `60`                       | 分块重叠                                                     |
| `RAG_RERANKER_PROVIDER`        | `none`                     | Reranker 类型：`none / jina`                                 |
| `RAG_RERANKER_MODEL`           | `jina-reranker-v2-base-multilingual` | Jina-compatible rerank 模型                       |
| `RAG_RERANKER_ENDPOINT`        | ``                         | Jina-compatible HTTP rerank endpoint                         |
| `RAG_RERANKER_API_KEY`         | ``                         | rerank endpoint API key                                      |
| `RAG_RERANKER_TOP_N`           | `3`                        | rerank 后截断数量                                            |
| `RAG_AUTO_MERGE_RATIO_THRESHOLD` | `0.5`                    | LlamaIndex AutoMergingRetriever ratio 阈值                    |
| `RAG_CONTEXT_TOKEN_BUDGET`     | `2400`                     | LlamaIndex node postprocessor 上下文预算                     |
| `RAG_MAX_UPLOAD_BYTES`         | `20971520`                 | 单个上传文件最大字节数                                       |
| `RAG_TEMPERATURE`              | `0.1`                      | 生成温度                                                     |
| `RAG_MAX_TOKENS`               | `2048`                     | 生成长度上限                                                 |

## Operations

本地依赖栈使用 PostgreSQL、Redis、etcd、MinIO 和 Milvus：

```bash
docker compose up -d postgres redis etcd minio milvus
finrag rebuild
uvicorn finrag.api:app --host 127.0.0.1 --port 8000
```

健康检查：

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

常见恢复动作：

- schema 或 manifest 不匹配：运行 `finrag rebuild`，从 `RAG_DATA_PATH` 源文档重建 PostgreSQL `documents`、`nodes`、BM25 状态和 Milvus leaf vectors。
- Milvus collection 损坏或被清空：运行 `finrag rebuild`，collection 会 drop/recreate 并重新写入 dense+sparse leaf vectors。
- Redis 缓存异常：重启 Redis 后访问 `/ready` 或执行一次检索，节点缓存会从 PostgreSQL 回源恢复。
