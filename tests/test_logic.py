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




def test_hybrid_search_exposes_component_scores_and_confidence():
    index = DocumentIndex([
        load_file(Path("docs/company_overview.md"))[1],
        load_file(Path("docs/company_overview.md"))[2],
    ])
    hits = index.search("预算", top_k=1)
    assert hits
    hit = hits[0]
    assert 0 <= hit.tfidf_score <= 1
    assert 0 <= hit.keyword_score <= 1
    assert hit.score == pytest.approx(0.75 * hit.tfidf_score + 0.25 * hit.keyword_score)
    assert hit.confidence == hit.score


def test_answer_quality_and_citation_coverage_states():
    index = DocumentIndex(load_file(Path("docs/company_overview.md")))
    grounded = index.ask("预算")
    assert grounded.quality_status == "grounded"
    assert grounded.citation_coverage == pytest.approx(1.0)
    assert grounded.confidence > 0

    low = index.ask("预算 完全不存在的词")
    assert low.quality_status == "low_evidence"
    assert 0 < low.citation_coverage < 1

    empty = DocumentIndex().ask("任何问题")
    assert empty.quality_status == "no_evidence"
    assert empty.citation_coverage == 0
    assert empty.confidence == 0


def test_corpus_insights_include_counts_keywords_and_numbers():
    index = DocumentIndex(load_file(Path("docs/company_overview.md")))
    assert index.insights.file_count == 1
    assert index.insights.chunk_count == 5
    assert any(word == "项" for word, _ in index.insights.top_keywords)
    assert "100" in index.insights.numeric_discoveries
    assert "2024" in index.insights.numeric_discoveries


def test_unsupported_file_type():
    with pytest.raises(ValueError):
        load_file("bad.pdf")
