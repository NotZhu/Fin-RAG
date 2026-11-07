# Contributing to FinRAG

感谢你关注 FinRAG！本文档说明如何搭建开发环境、运行测试以及提交改动。

## 行为准则

参与本项目即表示你同意保持尊重、专业和建设性的沟通氛围。讨论请聚焦技术内容，避免人身攻击或无关争论。

## 开发环境

FinRAG 使用 Python 3.11、FastAPI、LlamaIndex、Milvus、PostgreSQL 和 React/Vite 工作台。

1. 安装后端依赖（含开发依赖）：

   ```powershell
   uv sync --group dev
   Copy-Item .env.example .env
   ```

2. 启动本地依赖栈并按需初始化知识库：

   ```powershell
   docker compose up -d postgres etcd minio milvus
   uv run python -m finrag.cli rebuild --knowledge-base-id finance
   ```

3. 安装前端依赖：

   ```bash
   cd apps/web
   npm install
   ```

更多运行细节见 [README](./README.md) 的 Quick Start 与 Operations 章节。

## 提交前检查

在提交 Pull Request 之前，请确保以下检查全部通过：

```bash
uv run --with ruff ruff check --no-cache src scripts tests
uv run python -m pytest -q
uv lock --check
cd apps/web
npm test
npm run build
```

- 新增或修改功能时，请补充对应的单元测试或契约测试。
- 修改 README 时请注意 `tests/system/test_readme_contract.py` 中的契约约束。
- 保持 `uv.lock` 与 `pyproject.toml` 一致。

## 提交信息

- 使用清晰、祈使语气的提交说明（例如 `feat: add knowledge base archive endpoint`）。
- 一个提交聚焦一件事，避免混杂无关改动。

## 提交 Pull Request

1. 从 `main` 切出功能分支。
2. 完成改动并确保上述检查通过。
3. 在 PR 描述中说明改动动机、实现方式和验证方式。
4. 关联相关 issue（如适用）。

## 报告问题

提交 issue 时请包含：复现步骤、期望行为、实际行为、运行环境（Python / Node 版本、Docker 状态）以及相关日志或截图。
