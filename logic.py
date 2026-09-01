"""Offline document retrieval, citations, structured summaries, and CSV analysis."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv"}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", text.lower())


@dataclass(frozen=True)
class Chunk:
    source: str
    text: str
    index: int
    locator: str = ""


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[SearchHit]


@dataclass(frozen=True)
class Summary:
    conclusion: str
    key_points: list[str]
    risks: list[str]
    actions: list[str]
    citations: list[SearchHit]

    def to_markdown(self) -> str:
        sections = [f"### 结论\n{self.conclusion}"]
        for title, values in (("关键点", self.key_points), ("风险与未知", self.risks), ("待办建议", self.actions)):
            if values:
                sections.append("### " + title + "\n" + "\n".join(f"- {value}" for value in values))
        return "\n\n".join(sections)


@dataclass(frozen=True)
class CSVAnalysis:
    rows: int
    columns: list[str]
    numeric: dict[str, dict[str, float]]
    date_column: str | None
    preview: list[dict[str, str]]


class AnswerProvider(Protocol):
    def answer(self, question: str, context: str) -> str: ...


class OfflineProvider:
    def answer(self, question: str, context: str) -> str:
        if not context.strip():
            return "未找到相关内容。请换一种问法，或导入更多文档。"
        return "根据本地检索到的片段：\n" + context


class OpenAICompatibleProvider:
    def __init__(self, base_url: str, api_key: str, model: str = "gpt-4o-mini") -> None:
        self.base_url, self.api_key, self.model = base_url.rstrip("/"), api_key, model

    def answer(self, question: str, context: str) -> str:
        import json
        from urllib.error import HTTPError, URLError
        from urllib.request import Request, urlopen

        payload = json.dumps({"model": self.model, "temperature": 0, "messages": [
            {"role": "system", "content": "仅根据证据回答。每个事实后标注[1]等证据编号；证据不足时明确说不知道。"},
            {"role": "user", "content": f"证据：\n{context}\n\n问题：{question}"},
        ]}).encode("utf-8")
        request = Request(self.base_url + "/chat/completions", data=payload, headers={
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
        })
        try:
            with urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
            return str(data["choices"][0]["message"]["content"])
        except (HTTPError, URLError, TimeoutError, KeyError, IndexError, ValueError) as exc:
            raise RuntimeError(f"远程 provider 请求失败: {exc}") from exc


def load_file(path: str | Path, chunk_size: int = 1200) -> list[Chunk]:
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"不支持的文件类型: {path.suffix}")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        return [Chunk(str(path), " | ".join(f"{k}: {v}" for k, v in row.items()), i, f"第 {i + 2} 行")
                for i, row in enumerate(rows)]
    text = path.read_text(encoding="utf-8")
    chunks: list[Chunk] = []
    for piece in (p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()):
        for start in range(0, len(piece), chunk_size):
            value = piece[start:start + chunk_size].strip()
            if value:
                line = text[:text.find(value, start if start else 0)].count("\n") + 1
                chunks.append(Chunk(str(path), value, len(chunks), f"约第 {line} 行"))
    return chunks


def load_paths(paths: Iterable[str | Path]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in paths:
        chunks.extend(load_file(path))
    return chunks


def analyze_csv(path: str | Path, preview_rows: int = 8) -> CSVAnalysis:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        rows = list(reader)
    numeric: dict[str, dict[str, float]] = {}
    for column in columns:
        values = []
        for row in rows:
            try:
                values.append(float((row.get(column) or "").replace(",", "")))
            except ValueError:
                pass
        if values:
            numeric[column] = {"sum": float(sum(values)), "mean": float(np.mean(values)),
                               "min": float(min(values)), "max": float(max(values))}
    date_column = next((c for c in columns if any(re.match(r"^\d{4}[-/]\d{1,2}", r.get(c, "")) for r in rows)), None)
    return CSVAnalysis(len(rows), columns, numeric, date_column, rows[:preview_rows])


class DocumentIndex:
    def __init__(self, chunks: Sequence[Chunk] = ()) -> None:
        self.chunks = list(chunks)
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, preprocessor=None, token_pattern=None)
        texts = [c.text for c in self.chunks]
        self.matrix = self.vectorizer.fit_transform(texts) if texts and any(tokenize(t) for t in texts) else None

    def search(self, query: str, top_k: int = 3, min_score: float = 0.01) -> list[SearchHit]:
        if not self.chunks or not query.strip() or self.matrix is None or top_k < 1:
            return []
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()
        terms = set(tokenize(query))
        for i, chunk in enumerate(self.chunks):
            scores[i] += 0.03 * len(terms & set(tokenize(chunk.text)))
        order = np.argsort(-scores)[:top_k]
        return [SearchHit(self.chunks[i], float(scores[i])) for i in order if scores[i] >= min_score]

    def ask(self, question: str, top_k: int = 3, provider: AnswerProvider | None = None) -> Answer:
        hits = self.search(question, top_k)
        context = "\n".join(f"[{i + 1}] {hit.chunk.text}" for i, hit in enumerate(hits))
        return Answer((provider or OfflineProvider()).answer(question, context), hits)

    def summarize(self, top_k: int = 5) -> Summary:
        hits = [SearchHit(c, 1.0) for c in self.chunks[:top_k]]
        if not hits:
            return Summary("暂无可总结内容。", [], ["没有已索引文档"], [], [])
        points = [h.chunk.text for h in hits]
        return Summary(points[0], points, [], ["核对关键数字与来源"], hits)
