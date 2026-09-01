# AI 文档分析工作台

一个本地优先的文档检索、引用问答和 CSV 概览工具。支持 `.txt`、`.md`、`.csv`；默认使用确定性的 TF-IDF + 关键词检索，不上传文档、不默认调用 API。

## 快速开始

```bash
python -m venv .venv
.venv\\Scripts\\activate       # Windows
pip install -r requirements.txt
streamlit run app.py
```

启动后可在侧栏上传文件，也可直接分析 `docs/` 示例目录。首页会显示已加载文件、片段数和 CSV 数量；工作台包含三个标签页：

- **问答检索**：返回答案和带相关度、行号/CSV 行号的证据片段，可下载答案文本；低于阈值的结果不会伪装成答案。
- **结构化摘要**：输出结论、关键点、风险与待办，摘要证据仍可回溯。
- **CSV 分析**：查看预览、行数、字段数，以及数值列的合计、均值、最小值和最大值。

验收示例：启动后直接点击“问答检索”，输入“项目预算是多少？”，应看到“100 万元”和 `company_overview.md` 来源；切换“CSV 分析”选择 `monthly_sales.csv`，应看到 3 行数据和 revenue 合计 145。

## 可选远程模型

只有主动打开“启用远程模型”并填写 key 才会调用 OpenAI-compatible `/chat/completions` 接口。也可预先设置 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`。远程提示词要求事实标注 `[1]` 等证据编号；网络失败会在界面中显示可读错误。敏感数据建议保持离线模式。

## 开发与测试

```bash
pytest -q
```

核心 API 示例：

```python
from logic import DocumentIndex, load_paths
index = DocumentIndex(load_paths(["docs/company_overview.md"]))
print(index.ask("项目预算是多少？").text)
print(index.summarize().to_markdown())
```

当前仅按 UTF-8 读取文本（CSV 支持 UTF-8 BOM）。工具结果不构成财务、法律或安全建议。
