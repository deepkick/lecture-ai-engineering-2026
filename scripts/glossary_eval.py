"""第2回 宿題（発展）: 用語集の評価セットの採点と、② 検索の評価（API は呼ばない）。

使い方:
    uv run python -m scripts.glossary_eval search

評価セットの質問文をそのまま search_glossary に渡し、当たり方を
lectures/02/results/search_eval_<日時>.csv に保存して要約を表示する。
採点の関数は Colab のノートブックからも import するので、torch などには依存させない。
"""

import argparse
import csv
import json
import re
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

from app.tools import load_glossary, search_glossary

LECTURE_DIR = Path(__file__).resolve().parent.parent / "lectures" / "02"
EVAL_SET = LECTURE_DIR / "homework" / "eval_set.json"
RESULTS_DIR = LECTURE_DIR / "results"

CATEGORIES = ["A", "B1", "B2", "B3", "B4", "C1", "C2", "C3", "C4", "D"]
OK_OUTCOMES = {"correct", "abstained"}
SEARCH_COLUMNS = [
    "id", "category", "sub", "kind", "scope", "question", "expected_term",
    "hit_terms", "hit_at_1", "hit_at_3", "not_found",
]  # fmt: skip


def load_eval_set(path: Path = EVAL_SET) -> dict:
    """評価セットを読み、各 item に採点で使う正規表現と定義文を入れて返す。

    score を「item と出力だけで決まる」純粋関数にするため、ここで必要なものを item に持たせる。
    """
    eval_set = json.loads(Path(path).read_text(encoding="utf-8"))
    definitions = {t["id"]: t["definition"] for t in load_glossary()}
    for item in eval_set["items"]:
        item["abstain_patterns"] = eval_set["abstain_patterns"]
        if item["term_id"]:
            item["definition"] = definitions[item["term_id"]]
    return eval_set


def category_of(item_id: str) -> str:
    """id の接頭辞（A・B1〜B4・C1〜C4・D）を区分として返す。"""
    return re.match(r"A|B\d|C\d|D", item_id).group()


def _abstains(item: dict, text: str) -> bool:
    return any(re.search(pattern, text) for pattern in item["abstain_patterns"])


def _includes_all(item: dict, text: str) -> bool:
    """must_include の全グループで、グループ内のどれか 1 つが含まれるか（大文字小文字は区別しない）。"""
    folded = text.casefold()
    return all(any(word.casefold() in folded for word in group) for group in item["must_include"])


def rouge_l(text: str, reference: str) -> float:
    """文字単位の ROUGE-L（LCS の F 値）。空白と大文字小文字の違いは無視する。"""
    a = "".join(text.split()).casefold()
    b = "".join(reference.split()).casefold()
    if not a or not b:
        return 0.0
    previous = [0] * (len(b) + 1)
    for ch in a:
        current = [0]
        for j, other in enumerate(b):
            current.append(previous[j] + 1 if ch == other else max(previous[j + 1], current[j]))
        previous = current
    lcs = previous[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(a), lcs / len(b)
    return 2 * precision * recall / (precision + recall)


def score(item: dict, text: str) -> dict:
    """1 問を採点する。item は load_eval_set が返したもの。"""
    kind = item["kind"]
    if kind == "abstain":
        outcome = "abstained" if _abstains(item, text) else "asserted"
    elif kind == "general":
        outcome = "correct" if _includes_all(item, text) else "wrong"
    elif kind == "answer":
        if _abstains(item, text):
            outcome = "over_refusal"
        else:
            outcome = "correct" if _includes_all(item, text) else "wrong"
    else:
        raise ValueError(f"未知の kind: {kind}")
    rouge = rouge_l(text, item["definition"]) if kind == "answer" else None
    return {"outcome": outcome, "ok": outcome in OK_OUTCOMES, "rougeL": rouge}


def _group_stats(rows: list[dict]) -> dict:
    n = len(rows)
    ok = sum(r["ok"] for r in rows)
    rouges = [r["rougeL"] for r in rows if r.get("rougeL") is not None]
    return {
        "n": n,
        "ok": ok,
        "rate": ok / n if n else None,
        "rougeL_mean": statistics.mean(rouges) if rouges else None,
        "rougeL_median": statistics.median(rouges) if rouges else None,
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
    }


def _group_by(rows: list[dict], key) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        value = key(row)
        if value is not None:  # scope は answer 型にしかない
            groups.setdefault(value, []).append(row)
    return {value: _group_stats(members) for value, members in groups.items()}


def summarize(rows: list[dict]) -> dict:
    """全体と、区分・sub・scope ごとの件数、正解率、rougeL の平均と中央値を返す。"""
    by_category = _group_by(rows, lambda r: category_of(r["id"]))
    return {
        **_group_stats(rows),
        "by_category": {c: by_category[c] for c in CATEGORIES if c in by_category},
        "by_sub": _group_by(rows, lambda r: r["sub"]),
        "by_scope": _group_by(rows, lambda r: r["scope"]),
    }


def search_item(item: dict, terms_by_id: dict[str, str]) -> dict:
    """質問文をそのまま search_glossary に渡し、当たり方を 1 行にして返す。"""
    result = search_glossary(item["question"])
    # 見つかれば JSON の配列、見つからなければ「見つかりませんでした」の文が返る
    hits = [h["term"] for h in json.loads(result)] if result.startswith("[") else []
    expected = terms_by_id.get(item["term_id"], "")
    return {
        "id": item["id"],
        "category": category_of(item["id"]),
        "sub": item["sub"],
        "kind": item["kind"],
        "scope": item["scope"] or "",
        "question": item["question"],
        "expected_term": expected,
        "hit_terms": " | ".join(hits),
        "hit_at_1": bool(expected) and hits[:1] == [expected],
        "hit_at_3": bool(expected) and expected in hits[:3],
        "not_found": not hits,
    }


def run_search(eval_set: dict) -> list[dict]:
    """answer 型と abstain 型を検索にかける（general 型は用語集の質問ではないので除く）。"""
    terms_by_id = {t["id"]: t["term"] for t in load_glossary()}
    return [search_item(i, terms_by_id) for i in eval_set["items"] if i["kind"] != "general"]


def print_search_summary(rows: list[dict]) -> None:
    def rate(count: int, n: int) -> str:
        return f"{count}/{n} ({count / n:.0%})"

    print(f"{'区分':<4} {'n':>3}  {'hit@1 / 断れた':<16} hit@3")
    for category in CATEGORIES:
        members = [r for r in rows if r["category"] == category]
        if not members:
            continue
        if members[0]["kind"] == "answer":
            hit1 = rate(sum(r["hit_at_1"] for r in members), len(members))
            hit3 = rate(sum(r["hit_at_3"] for r in members), len(members))
        else:
            hit1, hit3 = rate(sum(r["not_found"] for r in members), len(members)), "-"
        print(f"{category:<4} {len(members):>3}  {hit1:<16} {hit3}")

    answers = [r for r in rows if r["kind"] == "answer"]
    abstains = [r for r in rows if r["kind"] == "abstain"]
    print(f"\nanswer 型: hit@1 {rate(sum(r['hit_at_1'] for r in answers), len(answers))}", end="")
    print(f" / hit@3 {rate(sum(r['hit_at_3'] for r in answers), len(answers))}")
    print(f"abstain 型: 見つかりませんでした {rate(sum(r['not_found'] for r in abstains), len(abstains))}")

    print("\nabstain 型なのに当たった語:")
    for r in abstains:
        if not r["not_found"]:
            print(f"  {r['id']}: {r['question']} -> {r['hit_terms']}")
    print("\nanswer 型で 3 件以内に正解がない問題:")
    for r in answers:
        if not r["hit_at_3"]:
            print(f"  {r['id']}: {r['question']} -> {r['hit_terms'] or '（見つからない）'}")


def cmd_search() -> None:
    rows = run_search(load_eval_set())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"search_eval_{datetime.now():%Y%m%d-%H%M%S}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SEARCH_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print_search_summary(rows)
    print(f"\n保存しました: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("search", help="評価セットの質問文で search_glossary の当たり方を調べる")
    parser.parse_args()
    cmd_search()


if __name__ == "__main__":
    main()
