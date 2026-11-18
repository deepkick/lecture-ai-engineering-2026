"""第5回 宿題（基礎）: 記録と集計のテスト（LLM も API キーも使わない）。"""

import json

import pytest
from pydantic_ai.models.test import TestModel

from app.retrieval import load_pages
from scripts import eval_topk, summarize_topk
from scripts.eval_topk import CUTOFF, done_qids, load_questions, parse_cited, required_hit


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("再送が原因です（第3回・PDF p.27）。", ["lec03-main-p027"]),
        ("第２回・ＰＤＦ ｐ．５２ にあります。", ["lec02-main-p052"]),          # 全角
        ("第2回 p.52 と 第3回 P13", ["lec02-main-p052", "lec03-main-p013"]),  # PDF なし、大文字、ピリオドなし
        ("（第3回・PDF p.13、p.39）", ["lec03-main-p013", "lec03-main-p039"]),  # 回を省いた続き
        ("第3回・PDF p.13、39・40", ["lec03-main-p013", "lec03-main-p039", "lec03-main-p040"]),
        ("第3回 pp.39〜41", ["lec03-main-p039", "lec03-main-p040", "lec03-main-p041"]),  # 範囲
        ("第 3 回 の PDF p. 5", ["lec03-main-p005"]),
        ("第3回・PDF p.27 と、第2回・PDF p.12、第3回・PDF p.27", ["lec03-main-p027", "lec02-main-p012"]),  # 重複なし
        ("第3回・PDF p.13、3つの方法がある", ["lec03-main-p013"]),  # 続きの数は助数詞が付けば数えない
        ("PDF p.52 にあります", []),  # 回が分からないページは数えない
        ("根拠は確認できませんでした。", []),
        ("", []),
    ],
)
def test_parse_cited(answer, expected):
    assert parse_cited(answer) == expected


def test_required_hit_needs_any_page_of_each_group():
    required = [["p013"], ["p039", "p041"]]
    assert required_hit(required, ["p041", "p001"]) == [False, True]
    assert required_hit(required, ["p013", "p039"]) == [True, True]
    assert required_hit(required, []) == [False, False]


def test_questions_point_to_existing_pages():
    pages = {page["page_id"] for page in load_pages()}
    questions = load_questions()
    assert [q["qid"] for q in questions] == ["V1", "V2", "V3", "V6", "X2", "L1"]
    for q in questions:
        assert q["required"] and all(group for group in q["required"])
        assert {p for group in q["required"] for p in group} <= pages
        assert {p for point in q["key_points"] for p in point["pages"]} <= pages
        assert {point["id"] for point in q["key_points"]} <= set(summarize_topk.KEY_POINT_IDS)


def test_done_qids_skips_success_and_cutoff_only():
    rows = [
        {"qid": "V1", "ok": True, "error": None},
        {"qid": "V2", "ok": False, "error": CUTOFF},
        {"qid": "V3", "ok": False, "error": "レート制限に達しました。"},
    ]
    assert done_qids(rows) == {"V1", "V2"}


QUESTION = {"qid": "V2", "question": "メモリとは？",
            "required": [["lec03-main-p013"], ["lec03-main-p039", "lec03-main-p041"]]}


def _hit(page_id):
    return {"page_id": page_id, "title": "t", "content": "c"}


def test_attempt_records_queries_pages_and_trace(monkeypatch):
    monkeypatch.setattr(eval_topk, "build_model", lambda: TestModel())
    monkeypatch.setattr("app.retrieval.search", lambda query: [_hit("lec03-main-p013"), _hit("lec03-main-p020")])
    record, trace, status = eval_topk.attempt(QUESTION, 5)

    assert record["ok"] and record["error"] is None and status is None
    assert record["top_k"] == 5 and record["qid"] == "V2"
    assert len(record["search_queries"]) == 1
    assert record["pages_full"] == ["lec03-main-p013", "lec03-main-p020"]
    assert record["required_hit"] == [True, False]
    assert record["requests"] == 2 and record["input_tokens"] > 0
    assert "[tool call]   search_materials" in trace and "[usage]" in trace and "--- answer ---" in trace
    json.dumps(record, ensure_ascii=False)  # JSONLに書ける


def test_attempt_treats_tool_error_as_failure(monkeypatch):
    def fail(query):
        raise ValueError("検索語が空です。")

    monkeypatch.setattr(eval_topk, "build_model", lambda: TestModel())
    monkeypatch.setattr("app.retrieval.search", fail)
    record, _, status = eval_topk.attempt(QUESTION, 5)
    assert not record["ok"] and record["error"].startswith("検索ツールのエラー") and status is None
    assert record["pages_full"] == [] and record["answer"]


def test_tool_error_status_tells_rate_limit_from_other_errors():
    assert eval_topk._tool_error_status(["レート制限に達しました。1分ほど待ってから再実行してください。"]) == 429
    assert eval_topk._tool_error_status(["モデルが混み合っています（503）。少し待ってから再実行してください。"]) == 503
    assert eval_topk._tool_error_status(["検索語が空です。"]) is None


def test_attempt_records_cutoff_when_request_limit_is_reached(monkeypatch):
    monkeypatch.setattr(eval_topk, "build_model", lambda: TestModel())
    monkeypatch.setattr(eval_topk, "REQUEST_LIMIT", 1)
    monkeypatch.setattr("app.retrieval.search", lambda query: [_hit("lec03-main-p013")])
    record, trace, _ = eval_topk.attempt(QUESTION, 5)
    assert not record["ok"] and record["error"] == CUTOFF and record["answer"] == ""
    assert record["pages_full"] == ["lec03-main-p013"]  # 打ち切りまでに取れたページは残す
    assert "[tool call]   search_materials" in trace


def _row(qid, k, pages, cited, ok=True, error=None, ts="2026-11-18T00:00:00+09:00", tokens=1000):
    return {"timestamp": ts, "model": "m", "top_k": k, "qid": qid, "ok": ok, "error": error, "answer": "回答",
            "search_queries": ["q"], "pages_full": pages, "required_hit": [], "cited": cited,
            "input_tokens": tokens if ok else None, "output_tokens": 100 if ok else None,
            "requests": 2 if ok else None, "cost_usd": None, "seconds": 3.0}


def test_summary_uses_ok_rows_only_and_keeps_missing_values_blank():
    questions = {"V2": QUESTION | {"key_points": [{"id": "a"}, {"id": "b"}]}}
    rows = [
        _row("V2", 5, ["lec03-main-p013", "lec03-main-p020"], ["lec03-main-p013", "lec03-main-p099"], tokens=1000),
        _row("V2", 5, [], [], ok=False, error="レート制限に達しました。"),
        _row("V2", 10, ["lec03-main-p013", "lec03-main-p020", "lec03-main-p041"], [], tokens=3000),
    ]
    runs = [summarize_topk.row_metrics(row, questions[row["qid"]]) for row in rows]
    assert runs[0]["required_rate"] == 0.5 and runs[0]["cited_from_retrieved"] == 0.5
    assert runs[0]["required_cited_rate"] == 0.5
    assert runs[2]["required_rate"] == 1.0 and runs[2]["cited_from_retrieved"] is None  # 引用0件は空欄

    summary = summarize_topk.summarize_k(runs)
    assert (summary[5]["n_ok"], summary[5]["n_failed"]) == (1, 1)
    assert summary[5]["stats"]["input_tokens"]["mean"] == 1000  # 失敗行は平均に入らない
    assert summary[10]["stats"]["cited_from_retrieved"]["n"] == 0
    assert summary[5]["stats"]["cost_usd"]["mean"] is None  # 算定不可を0にしない

    (diff,) = summarize_topk.page_diff(rows, questions)
    assert diff["only_low"] == [] and diff["only_high"] == ["lec03-main-p041"] and diff["n_common"] == 2

    markdown = summarize_topk.build_markdown(runs, summary, [diff], ["a.jsonl"], has_labels=False)
    assert "★lec03-main-p041" in markdown and "失敗: レート制限" in markdown


def test_label_template_and_labels_round_trip(tmp_path):
    questions = {"V2": QUESTION | {"key_points": [{"id": "a"}, {"id": "b"}]}}
    rows = [_row("V2", 5, [], []), _row("V2", 10, [], [])]
    template = tmp_path / "labels_template.csv"
    assert summarize_topk.write_label_template(template, rows, questions)

    text = template.read_text(encoding="utf-8-sig")
    assert text.splitlines()[0] == "qid,top_k,answer_head,kp_a,kp_b,kp_c,kp_d,overall,note,timestamp"
    assert "V2,5,回答,,,-,-,," in text

    filled = text.replace("V2,5,回答,,,-,-,,", "V2,5,回答,○,△,-,-,一部正答,").replace(
        "V2,10,回答,,,-,-,,", "V2,10,回答,○,○,-,-,正答,")
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text(filled, encoding="utf-8")
    runs = [summarize_topk.row_metrics(row, questions["V2"]) for row in rows]
    assert summarize_topk.apply_labels(runs, summarize_topk.load_labels(labels_path)) == []
    summary = summarize_topk.summarize_k(runs)
    assert summary[5]["overall"]["一部正答"] == 1 and summary[10]["overall"]["正答"] == 1

    # 記入が始まったひな形は上書きしない
    template.write_text(filled, encoding="utf-8-sig")
    assert not summarize_topk.write_label_template(template, rows, questions)


def test_summary_rejudges_required_pages_from_questions_not_from_jsonl():
    # 本番の後で根拠ページを直しても、集計し直せば反映される
    row = _row("V2", 5, ["lec03-main-p020"], ["lec03-main-p020"]) | {"required_hit": [True, True]}
    before = summarize_topk.row_metrics(row, QUESTION)
    assert (before["required_hit"], before["required_rate"], before["required_cited_rate"]) == ("0/2", 0.0, 0.0)

    revised = QUESTION | {"required": [["lec03-main-p020"], ["lec03-main-p039"]]}
    after = summarize_topk.row_metrics(row, revised)
    assert (after["required_hit"], after["required_rate"], after["required_cited_rate"]) == ("1/2", 0.5, 0.5)
