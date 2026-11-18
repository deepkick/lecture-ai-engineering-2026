"""第5回 宿題（基礎）: 検索語の再検索の判定のテスト（API キーを使わない）。"""

from scripts import replay_queries
from scripts.replay_queries import best_rank, collect_runs, judge, run_summary, targets

QUESTION = {"qid": "V2", "required": [["p013"], ["p039", "p041"]],
            "key_points": [{"id": "a", "pages": ["p013"]}, {"id": "b", "pages": ["p039", "p041", "p040"]}]}


def _ranking(*page_ids):
    return [{"page_id": page_id, "distance": 0.1 * i} for i, page_id in enumerate(page_ids, 1)]


def _row(k, queries, pages, ok=True):
    return {"qid": "V2", "top_k": k, "ok": ok, "search_queries": queries, "pages_full": pages}


def test_best_rank_takes_the_highest_page_of_the_group():
    ranking = _ranking("p001", "p041", "p002", "p039")
    assert best_rank(["p039", "p041"], ranking) == (2, "p041")
    assert best_rank(["p099"], ranking) == (None, None)


def test_collect_runs_uses_ok_rows_and_drops_repeated_queries():
    rows = [_row(5, ["a", "b", "a"], []), _row(5, ["x"], [], ok=False)]
    assert collect_runs(rows) == [{"qid": "V2", "top_k": 5, "pages_full": [], "queries": ["a", "b"]}]


def test_targets_adds_key_point_pages_outside_required_as_supplementary():
    kinds = [(t["kind"], t["pages"]) for t in targets(QUESTION)]
    assert kinds == [("required", ["p013"]), ("required", ["p039", "p041"]), ("supplementary", ["p040"])]


def test_judge_and_summary_split_top5_from_top10():
    filler = [f"x{i}" for i in range(5)]
    ranks = {"q1": _ranking("p013", *filler, "p039", "p040")}  # p013 が1位、p039 が7位、p040 が8位
    (run,) = collect_runs([_row(5, ["q1"], ["p013", *filler[:4]])])

    lines = judge([run], {"V2": QUESTION}, ranks)
    assert [(line["kind"], line["best_rank"], line["in_top5"], line["in_top10"]) for line in lines] == [
        ("required", 1, True, True), ("required", 7, False, True), ("supplementary", 8, False, True)]
    assert lines[1]["page_ranks"] == "p039:7;p041:>20"

    summary = run_summary(run, QUESTION, ranks)
    assert summary["hit_at"] == {5: 1, 10: 2, 20: 2}
    assert summary["actual"] == 1 and summary["reproduced"]  # 元のK=5で検索し直すと本番と同じ5ページ

    markdown = replay_queries.build_markdown([run], {"V2": QUESTION}, ranks, {})
    assert "| V2 | K=5 の本番 | 1 | 1/2 | 2/2 | 2/2 | 1/2 | 一致 |" in markdown


def test_summary_marks_runs_with_a_query_that_could_not_be_replayed():
    (run,) = collect_runs([_row(10, ["q1", "q2"], ["p013"])])
    summary = run_summary(run, QUESTION, {"q1": _ranking("p013")})
    assert not summary["complete"] and not summary["reproduced"]
    assert judge([run], {"V2": QUESTION}, {"q1": _ranking("p013")})[0]["query"] == "q1"
