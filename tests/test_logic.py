import tempfile
from pathlib import Path

import pytest

from logic import DocumentIndex, OfflineProvider, analyze_csv, load_file


def test_load_and_search_txt_has_locator():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "note.md"
        path.write_text("项目预算为 100 万元。\n\n团队将在六月发布。", encoding="utf-8")
        hits = DocumentIndex(load_file(path)).search("预算是多少", top_k=1)
        assert hits and "100 万元" in hits[0].chunk.text
        assert hits[0].chunk.source.endswith("note.md")
        assert hits[0].chunk.locator


def test_csv_import_analysis_and_row_citation():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "sales.csv"
        path.write_text("month,revenue\n2024-01,42\n2024-02,48\n", encoding="utf-8")
        index = DocumentIndex(load_file(path))
        result = index.ask("revenue", provider=OfflineProvider())
        analysis = analyze_csv(path)
        assert result.citations and "42" in result.text
        assert result.citations[0].chunk.locator == "第 2 行"
        assert analysis.rows == 2
        assert analysis.numeric["revenue"]["sum"] == 90
        assert analysis.date_column == "month"


def test_bom_csv_and_empty_index_are_explicit():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "bom.csv"
        path.write_text("name,value\na,1\n", encoding="utf-8-sig")
        assert analyze_csv(path).columns == ["name", "value"]
    result = DocumentIndex().ask("anything")
    assert result.citations == []
    assert "未找到" in result.text


def test_search_threshold_and_structured_summary():
    index = DocumentIndex(load_file(Path("docs/company_overview.md")))
    assert index.search("完全不存在的词", min_score=0.5) == []
    summary = index.summarize(top_k=2)
    assert summary.conclusion
    assert len(summary.key_points) == 2
    assert "### 关键点" in summary.to_markdown()


def test_unsupported_file_type():
    with pytest.raises(ValueError):
        load_file("bad.pdf")
