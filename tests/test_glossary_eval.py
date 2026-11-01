"""第2回 宿題（発展）: 採点関数のテスト（LLM も API キーも GPU も使わない）。"""

import pytest

from scripts.glossary_eval import (
    CATEGORIES,
    category_of,
    load_eval_set,
    rouge_l,
    run_search,
    score,
    summarize,
)


@pytest.fixture(scope="module")
def eval_set():
    return load_eval_set()


@pytest.fixture(scope="module")
def items(eval_set):
    return {i["id"]: i for i in eval_set["items"]}


def _row(item_id, outcome, ok, rouge=None, sub="s", scope=None):
    return {"id": item_id, "sub": sub, "scope": scope, "outcome": outcome, "ok": ok, "rougeL": rouge}


@pytest.mark.parametrize(
    "text",
    [
        "用語集に「RAG とは何ですか？」は見つかりませんでした。",  # 検索ツールの文言をそのまま返した場合
        "用語集には「RAG」は載っていません。",
        "用語集に該当する用語はありません",
        "その用語は用語集にありません。",
        "申し訳ありませんが、わかりません。",
    ],
)
def test_abstain_expressions_count_as_abstained(items, text):
    result = score(items["C1-1"], text)
    assert result == {"outcome": "abstained", "ok": True, "rougeL": None}


@pytest.mark.parametrize(
    "text",
    [
        "RAG は検索結果を入力に加えて答えを生成する手法です。",
        "用語集によると、RAG は検索と生成を組み合わせる手法です。",
        "",
    ],
)
def test_assertion_is_not_abstained(items, text):
    assert score(items["C1-1"], text) == {"outcome": "asserted", "ok": False, "rougeL": None}


PATTERN_TESTS = load_eval_set()["abstain_pattern_tests"]


@pytest.mark.parametrize("text", PATTERN_TESTS["should_match"])
def test_pattern_tests_should_match_are_abstain(items, text):
    assert score(items["C1-1"], text)["outcome"] == "abstained"
    assert score(items["A1"], text)["outcome"] == "over_refusal"


@pytest.mark.parametrize("text", PATTERN_TESTS["should_not_match"])
def test_pattern_tests_should_not_match_are_not_abstain(items, text):
    assert score(items["C1-1"], text)["outcome"] == "asserted"
    assert score(items["A1"], text)["outcome"] != "over_refusal"


def test_any_abstain_pattern_is_enough():
    item = {"kind": "abstain", "abstain_patterns": ["一致しない語", "用語集にない"], "must_include": []}
    assert score(item, "用語集にない語です")["outcome"] == "abstained"


def test_answer_correct_requires_every_group_and_any_word_in_group():
    item = {
        "kind": "answer",
        "abstain_patterns": ["断る"],
        "must_include": [["固定"], ["小さな行列", "低ランク"]],
        "definition": "重みを固定し、小さな行列だけを学習する手法。",
    }
    assert score(item, "重みを固定して低ランクの差分を学習する")["outcome"] == "correct"
    assert score(item, "重みを固定して小さな行列を学習する")["ok"] is True
    # 2 グループ目がないので不正解
    assert score(item, "重みを固定する手法")["outcome"] == "wrong"
    assert score(item, "重みを固定する手法")["ok"] is False


def test_must_include_ignores_case():
    item = {
        "kind": "answer",
        "abstain_patterns": [],
        "must_include": [["LoRA"]],
        "definition": "LoRA",
    }
    assert score(item, "lora です")["outcome"] == "correct"


def test_answer_abstain_is_over_refusal_even_with_required_words(items):
    # 必須語句を含んでいても、断る表現に当たれば over_refusal を先に判定する
    text = "固定や低ランクの話かもしれませんが、その用語は用語集にありません。"
    result = score(items["A1"], text)
    assert result["outcome"] == "over_refusal"
    assert result["ok"] is False
    assert isinstance(result["rougeL"], float)


def test_general_uses_must_include_and_ignores_abstain(items):
    assert score(items["D1"], "東京です。") == {"outcome": "correct", "ok": True, "rougeL": None}
    assert score(items["D2"], "THANK YOU")["outcome"] == "correct"
    # general 型には over_refusal がなく、断っただけなら wrong になる
    assert score(items["D1"], "わかりません。")["outcome"] == "wrong"


def test_rouge_l_known_values():
    assert rouge_l("abcd", "abcd") == 1.0
    assert rouge_l("", "abcd") == 0.0
    assert rouge_l("abcd", "") == 0.0
    assert rouge_l("xyz", "abcd") == 0.0
    assert rouge_l("axc", "abc") == pytest.approx(2 / 3)  # LCS 2、適合率・再現率とも 2/3
    assert rouge_l("ab", "abcd") == pytest.approx(2 / 3)  # LCS 2、適合率 1、再現率 1/2
    assert rouge_l("A b C", "a　Bc") == 1.0  # 空白（全角も）と大文字小文字は無視する


def test_rouge_l_is_not_fooled_by_word_order():
    # LCS は順番を保つ部分列なので、並べ替えると下がる
    assert rouge_l("dcba", "abcd") == pytest.approx(1 / 4)


def test_category_of():
    assert category_of("A1") == "A"
    assert category_of("B1-3") == "B1"
    assert category_of("B4-58") == "B4"
    assert category_of("C4-20") == "C4"
    assert category_of("D4") == "D"


def test_summarize_counts_and_rates():
    rows = [
        _row("A1", "correct", True, 0.8, sub="seen", scope="general"),
        _row("A2", "wrong", False, 0.2, sub="seen", scope="course"),
        _row("B1-1", "over_refusal", False, 0.4, sub="new", scope="general"),
        _row("C1-1", "abstained", True, sub="real"),
        _row("D1", "correct", True, sub="general"),
    ]
    s = summarize(rows)
    assert (s["n"], s["ok"], s["rate"]) == (5, 3, 3 / 5)
    assert s["rougeL_mean"] == pytest.approx((0.8 + 0.2 + 0.4) / 3)
    assert s["rougeL_median"] == pytest.approx(0.4)
    assert s["outcomes"] == {"correct": 2, "wrong": 1, "over_refusal": 1, "abstained": 1}
    assert list(s["by_category"]) == ["A", "B1", "C1", "D"]  # CATEGORIES の順に並ぶ
    assert s["by_category"]["A"]["rate"] == 0.5
    assert s["by_category"]["C1"]["rougeL_mean"] is None
    assert s["by_sub"]["seen"]["n"] == 2
    assert set(s["by_scope"]) == {"general", "course"}  # scope のない C・D は数えない
    assert s["by_scope"]["general"]["n"] == 2


def test_eval_set_shape(eval_set):
    assert eval_set["version"] == "v1"
    assert len(eval_set["items"]) == 116
    assert {category_of(i["id"]) for i in eval_set["items"]} == set(CATEGORIES)


def test_own_definition_is_scored_correct(eval_set):
    """定義文をそのまま返せば正解になる（必須語句や断る表現の判定が定義文を誤って落とさない）。"""
    failed = [
        i["id"]
        for i in eval_set["items"]
        if i["kind"] == "answer" and not score(i, i["definition"])["ok"]
    ]
    # B3-4 だけは正解が用語名で、定義文に用語名が含まれない
    assert failed == ["B3-4"]


def test_question_echo_is_not_scored_correct(eval_set):
    """問題文を繰り返しただけでは正解にならない。"""
    echoed = [
        i["id"]
        for i in eval_set["items"]
        if i["kind"] == "answer" and score(i, i["question"])["ok"]
    ]
    assert echoed == []


def test_abstain_answer_is_abstained_and_over_refusal(eval_set, items):
    assert score(items["C1-1"], eval_set["abstain_answer"])["outcome"] == "abstained"
    assert score(items["A1"], eval_set["abstain_answer"])["outcome"] == "over_refusal"


def test_run_search_columns_and_hits(eval_set):
    rows = {r["id"]: r for r in run_search(eval_set)}
    assert len(rows) == 112  # general 型の 4 問は除く
    assert rows["A1"]["hit_at_1"] is True
    assert rows["A1"]["not_found"] is False
    assert rows["C1-1"]["not_found"] is True
    assert rows["C1-1"]["hit_at_1"] is False
    # 用語集にない語でも「LoRA」を含むので当たってしまう
    assert rows["C3-2"]["not_found"] is False
    assert rows["C3-2"]["hit_terms"].startswith("LoRA")
