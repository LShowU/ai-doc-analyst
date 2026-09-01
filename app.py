import os
from pathlib import Path

import streamlit as st

from logic import DocumentIndex, OpenAICompatibleProvider, analyze_csv, load_paths

st.set_page_config(page_title="Doc Analyst", page_icon="D", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root { --ink:#163238; --muted:#60777a; --line:#d9e7e5; --teal:#087f78; --teal-soft:#e8f5f2; --orange:#e4773c; --orange-soft:#fff2e9; --paper:#ffffff; --bg:#f4f8f7; }
.stApp { background: var(--bg); color: var(--ink); }
.block-container { max-width: 1240px; padding: 2.4rem 2.6rem 4rem; }
h1, h2, h3 { color: var(--ink); letter-spacing: 0; font-weight: 720; }
h1 { font-size: 2.25rem; margin-bottom: .25rem; }
h2 { font-size: 1.25rem; }
[data-testid="stMetric"] { background: var(--paper); border: 1px solid var(--line); border-radius: 10px; padding: .9rem 1.05rem; box-shadow: 0 2px 10px rgba(22,50,56,.04); }
[data-testid="stMetricLabel"] { color: var(--muted); }
[data-testid="stMetricValue"] { color: var(--teal); }
[data-testid="stSidebar"] { background: #edf5f3; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: var(--teal); }
[data-baseweb="tab-list"] { gap: .35rem; border-bottom: 1px solid var(--line); }
button[data-baseweb="tab"] { color: var(--muted); font-weight: 650; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--teal); border-bottom-color: var(--teal); }
.stButton > button[kind="primary"] { background: var(--teal); border: 0; border-radius: 7px; padding: .55rem 1.2rem; }
.stButton > button[kind="primary"]:hover { background: #066961; }
.source { border-left: 3px solid var(--orange); padding: .7rem .9rem; margin: .55rem 0 .2rem; background: var(--orange-soft); border-radius: 0 7px 7px 0; }
.workspace-header { display:flex; justify-content:space-between; align-items:flex-end; gap:2rem; padding-bottom:1.4rem; border-bottom:1px solid var(--line); margin-bottom:1.35rem; }
.eyebrow { color:var(--teal); font-size:.78rem; font-weight:750; letter-spacing:.08em; text-transform:uppercase; }
.subtitle { color:var(--muted); font-size:1rem; margin-top:.35rem; }
.status-pill { background:var(--teal-soft); color:var(--teal); border:1px solid #c5e5df; border-radius:999px; padding:.42rem .75rem; font-size:.82rem; white-space:nowrap; }
.section-note { color:var(--muted); font-size:.9rem; margin-bottom:.8rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="正在建立文档索引...")
def indexed(paths: tuple[str, ...], mtimes: tuple[float, ...]) -> DocumentIndex:
    return DocumentIndex(load_paths(paths))


with st.sidebar:
    st.header("工作区")
    uploads = st.file_uploader("导入文档或 CSV", type=["txt", "md", "csv"], accept_multiple_files=True)
    top_k = st.slider("最多引用", 1, 8, 3)
    st.divider()
    st.subheader("回答设置")
    use_remote = st.toggle("启用远程模型", value=False)
    remote_url = st.text_input("API Base URL", value=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    remote_model = st.text_input("模型", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    api_key = st.text_input("API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")

if uploads:
    temp_dir = Path(".streamlit_uploads")
    temp_dir.mkdir(exist_ok=True)
    paths = []
    for upload in uploads:
        target = temp_dir / Path(upload.name).name
        target.write_bytes(upload.getvalue())
        paths.append(str(target))
else:
    paths = [str(p) for p in sorted(Path("docs").glob("*")) if p.suffix.lower() in {".txt", ".md", ".csv"}]

mtimes = tuple(Path(p).stat().st_mtime for p in paths)
index = indexed(tuple(paths), mtimes)

st.markdown("""
<div class="workspace-header">
  <div><div class="eyebrow">LOCAL INTELLIGENCE WORKSPACE</div><h1>文档分析工作台</h1><div class="subtitle">从文件到答案、摘要与数据洞察，所有证据都可回溯。</div></div>
  <div class="status-pill">● 离线优先 · 可追溯引用</div>
</div>
""", unsafe_allow_html=True)
with st.expander(f"已加载文件（{len(paths)}）"):
    if paths:
        st.write("\n".join(f"- {Path(path).name}" for path in paths))
    else:
        st.info("请从左侧上传文件，或在 docs/ 放入 txt、md、csv。")
col1, col2, col3 = st.columns(3)
col1.metric("已索引片段", f"{len(index.chunks):,}", help="当前文档被切分并建立检索索引的片段数")
col2.metric("文档数量", f"{len(paths):,}", help="当前工作区中的 TXT、Markdown 和 CSV 文件")
col3.metric("CSV 文件", f"{sum(Path(p).suffix.lower() == '.csv' for p in paths):,}", help="可进入 CSV 分析页查看统计摘要")

ask_tab, summary_tab, data_tab = st.tabs(["问答检索", "结构化摘要", "CSV 分析"])
with ask_tab:
    left, right = st.columns([1.12, .88], gap="large")
    with left:
        st.markdown("### 提问")
        st.markdown('<div class="section-note">问题会先在本地索引中检索，再返回带编号的证据片段。</div>', unsafe_allow_html=True)
        question = st.text_area("问题", placeholder="例如：项目预算是多少？", height=130, label_visibility="collapsed")
        ask = st.button("检索并回答", type="primary", icon="🔎", use_container_width=True)
    with right:
        st.markdown("### 工作区状态")
        st.info(f"当前索引 {len(index.chunks)} 个片段，最多返回 {top_k} 条引用。")
        st.caption("开启远程模型后，检索到的片段才会发送到配置的 API。")
    if ask and question.strip():
        provider = None
        if use_remote:
            if not api_key:
                st.error("请先填写 API Key，或关闭远程模型。"); st.stop()
            provider = OpenAICompatibleProvider(remote_url, api_key, remote_model)
        try:
            result = index.ask(question, top_k=top_k, provider=provider)
            st.divider()
            answer_col, citation_col = st.columns([1.12, .88], gap="large")
            with answer_col:
                st.markdown("### 答案")
                st.write(result.text)
                st.download_button("下载答案", result.text, file_name="answer.txt", mime="text/plain", icon="⬇️")
            with citation_col:
                st.markdown(f"### 证据来源 · {len(result.citations)} 条")
                if not result.citations:
                    st.info("没有达到相关性阈值的片段。")
                for number, hit in enumerate(result.citations, 1):
                    location = f" · {hit.chunk.locator}" if hit.chunk.locator else ""
                    with st.expander(f"[{number}] {Path(hit.chunk.source).name}{location} · {hit.score:.3f}"):
                        st.code(hit.chunk.text, language="text")
        except RuntimeError as exc:
            st.error(str(exc))

with summary_tab:
    st.write("基于当前索引片段生成确定性摘要，适合快速确认重点。")
    if st.button("生成摘要", icon="📝"):
        summary = index.summarize(top_k=top_k)
        st.markdown(summary.to_markdown())
        if summary.citations:
            st.caption("摘要证据")
            for hit in summary.citations:
                st.write(f"{Path(hit.chunk.source).name} · {hit.chunk.locator}")

with data_tab:
    csv_paths = [p for p in paths if Path(p).suffix.lower() == ".csv"]
    if not csv_paths:
        st.info("当前工作区没有 CSV 文件。")
    else:
        selected = st.selectbox("选择 CSV", csv_paths, format_func=lambda p: Path(p).name)
        try:
            analysis = analyze_csv(selected)
            a, b = st.columns(2)
            a.metric("数据行数", analysis.rows)
            b.metric("字段数", len(analysis.columns))
            st.dataframe(analysis.preview, use_container_width=True, hide_index=True)
            if analysis.numeric:
                st.subheader("数值字段")
                st.dataframe(analysis.numeric, use_container_width=True)
        except (OSError, UnicodeError, ValueError) as exc:
            st.error(f"CSV 无法分析：{exc}")
