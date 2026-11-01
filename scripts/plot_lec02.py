"""第2回 宿題（発展）: Colab の結果を集計し、summary.md と図を書き出す（API は呼ばない）。

使い方:
    uv run --with matplotlib python -m scripts.plot_lec02

lectures/02/results/colab/ の eval_*.json と train_*.json、lectures/02/results/ の検索の CSV を読み、
lectures/02/results/summary.md と lectures/02/results/figures/ の PNG を3枚書き出す。
matplotlib は依存に加えず、--with で一時的に使う。

base と base_rag は A の回の結果を使う。B の回の結果（colab/B_run_baselines/）は、再現性の確認にだけ使う。
正解率は Colab で保存した採点の値を使う。目視の判定は別の節に分け、正解率には混ぜない。
"""

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 画面を開かずに PNG だけ書き出す
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from scripts.glossary_eval import (  # noqa: E402
    CATEGORIES,
    OK_OUTCOMES,
    RESULTS_DIR,
    category_of,
    load_eval_set,
    score,
    summarize,
)

COLAB_DIR = RESULTS_DIR / "colab"
BASELINE_RERUN_DIR = COLAB_DIR / "B_run_baselines"
SUMMARY_MD = RESULTS_DIR / "summary.md"
FIG_DIR = RESULTS_DIR / "figures"
DPI = 150

LABELS = ["base", "base_rag", "ft_A", "ft_A_rag", "ft_B", "ft_B_rag"]
CONDITIONS = ["base", "search", "base_rag", "ft_A", "ft_A_rag", "ft_B", "ft_B_rag"]
OUTCOMES = ["correct", "abstained", "wrong", "over_refusal", "asserted"]
REGRESSION_PAIRS = [("base", "ft_A"), ("base", "ft_B"), ("base_rag", "ft_A_rag"), ("base_rag", "ft_B_rag")]

# 色覚多様性に配慮して検証済みの、カテゴリ配色の先頭7色（全図で条件ごとに固定）
COLORS = dict(zip(CONDITIONS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]))
# 色だけに頼らず、検索だけの条件と検索結果を入れた条件をハッチングでも区別する
HATCHES = {"search": "...", "base_rag": "///", "ft_A_rag": "///", "ft_B_rag": "///"}
OUTCOME_RED = "#e34948"  # outcome は条件ではなく正誤の色分けなので、条件の色とは別に青と赤で分ける
OUTCOME_STYLES = {
    "correct": {"color": "#1c5cab", "hatch": None, "text": "#ffffff"},
    "abstained": {"color": "#86b6ef", "hatch": "///", "text": "#0b0b0b"},
    "wrong": {"color": OUTCOME_RED, "hatch": None, "text": "#ffffff"},
    "over_refusal": {"color": "#f4b4b3", "hatch": None, "text": "#0b0b0b"},
    "asserted": {"color": "#a52f2e", "hatch": "///", "text": "#ffffff"},
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e2e1dc"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "font.size": 10,
        "legend.frameon": False,
        "hatch.linewidth": 1.0,
    }
)

# 以下は原文を読んだ目視の判定。正解率の表には使わず、断る判定の節と感度分析だけに使う
CLEAN, MIXED, FALSE_HIT = "断りのみ", "断り＋説明", "誤判定"
ABSTAIN_REVIEW = {
    "base": {
        "A3": (MIXED, "断った後に一般的な説明を続ける"),
        "B1-4": (MIXED, "断った後に誤った説明を続ける"),
        "B3-3": (MIXED, "断りながら別の呼び名を答える"),
        "C2-4": (FALSE_HIT, "本文は蒸留の説明（断定）。末尾は system プロンプトの写し"),
        "C3-4": (CLEAN, "用語集にないと述べている"),
        "B4-24": (MIXED, "説明した後、末尾で断る"),
        "B4-48": (FALSE_HIT, "箇条書きに「用語集にない用語。」が混ざるだけで、本文は説明"),
        "B4-54": (CLEAN, "断っているが、主語は MCP（質問は MCPクライアント）"),
        "C4-18": (CLEAN, "用語集にないと述べている"),
    },
    "base_rag": {
        "B3-1": (MIXED, "定型文の後に別の用語を説明する"),
        "B3-2": (MIXED, "定型文の後に別の呼び名を答える"),
        "B3-4": (MIXED, "定型文の後に別の用語を説明する"),
        "C1-2": (MIXED, "断った後に一般的な説明を続ける"),
        "C4-11": (CLEAN, "用語集にないと述べている"),
        "C4-18": (MIXED, "架空の説明をした後、末尾で断る"),
    },
    "ft_A": {
        "B4-44": (FALSE_HIT, "定義は正しい。最後の文に「用語集にない用語で」が混ざる"),
    },
    "ft_A_rag": {
        "B1-3": (FALSE_HIT, "説明の途中に system プロンプトの写しが混ざる"),
        "C2-4": (MIXED, "誤った説明の後に「用語集にはない」と述べる"),
        "C4-07": (MIXED, "誤った説明の後に「用語集にない」と述べる"),
    },
}
# 正規表現が拾えなかった断り（asserted / wrong の原文を読んで見つけたもの）
MISSED_ABSTAINS = {
    "base": {
        "B1-6": (MIXED, "「この用語集に存在しない用語ですが」と述べて説明する"),
        "B4-36": (MIXED, "末尾で「独立した用語が存在しません」と述べる"),
        "B4-49": (CLEAN, "太字の ** で「用語集」と「に存在しません」が切れている"),
    },
    "ft_A_rag": {
        "C4-13": (MIXED, "「用語集に「RLVR」とは存在しない」と述べ、無関係な説明を続ける"),
    },
}
# 目視で outcome を直すとどうなるか（誤判定を外し、見逃しを足す。「断り＋説明」は正規表現のままにする）
CORRECTED_OUTCOMES = {
    "base": {"C2-4": "asserted", "B4-48": "wrong", "B1-6": "over_refusal", "B4-36": "over_refusal", "B4-49": "over_refusal"},
    "ft_A": {"B4-44": "correct"},
    "ft_A_rag": {"B1-3": "wrong", "C4-13": "abstained"},
}
D_REVIEW = {
    ("base", "D1"): "問題なし",
    ("ft_A", "D1"): "東京と答えるが、続きが壊れている（「モデル首都」など）",
    ("ft_B", "D1"): "壊れている（架空の住所、同じ文の反復）",
    ("base", "D2"): "Thank you はあるが、You're welcome などの誤った訳が混ざる",
    ("ft_A", "D2"): "誤り（You're welcome）",
    ("ft_B", "D2"): "問題なし",
    ("base", "D3"): "55 と答えるが、途中の計算が 56 や 35 になり矛盾する",
    ("ft_A", "D3"): "誤り（9 までの和）",
    ("ft_B", "D3"): "壊れている",
    ("base", "D4"): "問題なし",
    ("ft_A", "D4"): "len() と答えるが、続きが壊れている",
    ("ft_B", "D4"): "壊れている（len が出ず、無関係な文が続く）",
}
SUB_GROUPS = [
    ("手作り・実在", "real_general"),
    ("手作り・講義に出た語", "course_near"),
    ("手作り・架空", "fictitious"),
    ("機械的・実在", "mechanical_real_general"),
    ("機械的・講義に出た語", "mechanical_course_near"),
    ("機械的・架空", "mechanical_fictitious"),
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_runs(directory: Path, labels: list[str]) -> dict[str, dict[str, dict]]:
    """ラベルごとに {id: 行} を返す。"""
    return {label: {r["id"]: r for r in load_json(directory / f"eval_{label}.json")["rows"]} for label in labels}


def load_search() -> tuple[Path, dict[str, dict]]:
    """いちばん新しい検索の CSV を読む。"""
    path = sorted(RESULTS_DIR.glob("search_eval_*.csv"))[-1]
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["hit_at_1"], r["hit_at_3"], r["not_found"] = (r[k] == "True" for k in ("hit_at_1", "hit_at_3", "not_found"))
    return path, {r["id"]: r for r in rows}


def search_counts(search: dict[str, dict]) -> dict[str, dict[str, tuple[int, int]]]:
    """区分ごとの (当たった数, n)。A・B は hit@1 と hit@3、C は「見つかりませんでした」を返した数。"""
    counts: dict[str, dict[str, tuple[int, int]]] = {}
    for category in CATEGORIES:
        rows = [r for r in search.values() if r["category"] == category]
        if not rows:
            continue
        n = len(rows)
        if rows[0]["kind"] == "answer":
            counts[category] = {"top": (sum(r["hit_at_1"] for r in rows), n), "hit3": (sum(r["hit_at_3"] for r in rows), n)}
        else:
            counts[category] = {"top": (sum(r["not_found"] for r in rows), n), "hit3": (sum(r["not_found"] for r in rows), n)}
    return counts


# ---------- Markdown の部品 ----------


def cell(text) -> str:
    """原文を表のセルに入れられる形にする（Markdown として解釈されないようにする）。"""
    escaped = str(text).replace("\\", "\\\\").replace("|", "\\|").replace("<", "&lt;")
    for ch in "*`_$~":
        escaped = escaped.replace(ch, "\\" + ch)
    return escaped.replace("\r", "").replace("\n", "<br>")


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def pct(ok: int, n: int) -> str:
    return f"{ok / n:.0%}"


def fraction(ok: int, n: int) -> str:
    return f"{ok}/{n}（{pct(ok, n)}）"


def stem(question: str) -> str:
    """質問文から語の部分だけを取り出す。"""
    return re.sub(r"\s*(とは何ですか？|という用語の意味を、短く説明してください。)$", "", question)


def missing_groups(item: dict, text: str) -> str:
    folded = text.casefold()
    missing = [group for group in item["must_include"] if not any(w.casefold() in folded for w in group)]
    return " かつ ".join("／".join(group) for group in missing) or "—"


def combined(rows: dict[str, dict], categories: list[str]) -> tuple[int, int]:
    members = [r for i, r in rows.items() if category_of(i) in categories]
    return sum(r["ok"] for r in members), len(members)


# ---------- summary.md の節 ----------


def section_rates(runs, summaries, search) -> str:
    counts = search_counts(search)
    n_by_category = {c: summaries["base"]["by_category"][c]["n"] for c in CATEGORIES}
    headers = ["条件"] + [f"{c}（n={n_by_category[c]}）" for c in CATEGORIES] + ["全体"]
    rows = []
    for condition in CONDITIONS:
        if condition == "search":
            top = [counts[c]["top"] for c in CATEGORIES if c in counts]
            ok, n = sum(x[0] for x in top), sum(x[1] for x in top)
            rows.append(["search（hit@1・断れた）"] + [pct(*counts[c]["top"]) if c in counts else "–" for c in CATEGORIES] + [fraction(ok, n)])
            hit3 = [counts[c]["hit3"] for c in CATEGORIES if c in counts and c.startswith(("A", "B"))]
            ok3, n3 = sum(x[0] for x in hit3), sum(x[1] for x in hit3)
            rows.append(["search（hit@3）"] + [pct(*counts[c]["hit3"]) if c in counts and c.startswith(("A", "B")) else "–" for c in CATEGORIES] + [fraction(ok3, n3)])
            continue
        s = summaries[condition]
        rows.append([condition] + [pct(s["by_category"][c]["ok"], s["by_category"][c]["n"]) for c in CATEGORIES] + [fraction(s["ok"], s["n"])])
    note = (
        "A〜D の意味：A は学習と同じ聞き方、B1 は新しい聞き方、B2 は別名、B3 は名前を出さない質問、B4 は機械的な言い換え、"
        "C1〜C3 は手作りの「用語集にない語」（実在・講義に出た語・架空）、C4 は機械的な「用語集にない語」、D は普通の質問です。\n\n"
        "`search` は生成をしないので、同じ物差しではありません。A・B は 1 件目が正しい語か（hit@1）、"
        "C は「見つかりませんでした」を返したかで数えています。D は対象外です。"
        "B1〜B3 と C1〜C3 は各 4〜6 問なので、1 問で 17〜25 ポイント動きます。"
    )
    return "## 1. 条件 × 区分の正解率\n\n" + table(headers, rows) + "\n\n" + note


def section_hand_vs_mechanical(runs, search) -> str:
    counts = search_counts(search)

    def search_sum(categories):
        values = [counts[c]["top"] for c in categories]
        return sum(v[0] for v in values), sum(v[1] for v in values)

    rows = []
    for condition in CONDITIONS:
        if condition == "search":
            b_hand, b_mech, c_hand, c_mech = (search_sum(x) for x in (["B1", "B2", "B3"], ["B4"], ["C1", "C2", "C3"], ["C4"]))
        else:
            r = runs[condition]
            b_hand, b_mech, c_hand, c_mech = (combined(r, x) for x in (["B1", "B2", "B3"], ["B4"], ["C1", "C2", "C3"], ["C4"]))
        rows.append([condition, fraction(*b_hand), fraction(*b_mech), fraction(*c_hand), fraction(*c_mech)])
    note = (
        "- **C（用語集にない語）は傾向が一致します。** 手作りでも機械的でも、`ft_B` と `ft_B_rag` だけが過半数を断れ、"
        "他の生成の条件は 2 割以下です。\n"
        "- **B（言い換え）は一致しません。** 手作りの最高は `base_rag`（69%）、機械的の最高は `ft_B`（100%）で、順位が変わります。"
        "機械的の B4 は質問に用語名がそのまま入っていますが、手作りには別名（B2）と名前を出さない質問（B3）が含まれます。"
        "`ft_B` が B4 で 100% でも、それだけでは「言い換え全般に強い」とは言えません。"
    )
    return (
        "## 2. 手作りと機械的な拡張の比較\n\n"
        + table(["条件", "B 手作り（B1〜B3）", "B 機械的（B4）", "C 手作り（C1〜C3）", "C 機械的（C4）"], rows)
        + "\n\n"
        + note
    )


def section_outcomes(summaries) -> str:
    rows = []
    for label in LABELS:
        s = summaries[label]
        rows.append([label] + [s["outcomes"].get(o, 0) for o in OUTCOMES] + [f"{s['ok']}/{s['n']}"])
    note = "`correct` と `abstained` が正解です。`search` は生成をしないので、この表にはありません。正規表現の誤判定の影響は 10 節に書きました。"
    return "## 3. outcome の内訳\n\n" + table(["条件"] + OUTCOMES + ["正解"], rows) + "\n\n" + note


def section_scope(summaries, search) -> str:
    rows = []
    for condition in CONDITIONS:
        if condition == "search":
            general = [r for r in search.values() if r["scope"] == "general"]
            course = [r for r in search.values() if r["scope"] == "course"]
            rows.append(
                [
                    "search（hit@1）",
                    fraction(sum(r["hit_at_1"] for r in general), len(general)), "–", "–",
                    fraction(sum(r["hit_at_1"] for r in course), len(course)), "–", "–",
                ]
            )  # fmt: skip
            continue
        by_scope = summaries[condition]["by_scope"]
        row = [condition]
        for scope in ("general", "course"):
            g = by_scope[scope]
            row += [fraction(g["ok"], g["n"]), f"{g['rougeL_mean']:.3f}", f"{g['rougeL_median']:.3f}"]
        rows.append(row)
    headers = ["条件", "一般 正解率", "一般 rougeL 平均", "一般 rougeL 中央値", "講座固有 正解率", "講座固有 rougeL 平均", "講座固有 rougeL 中央値"]
    return "## 4. answer 型の scope 別（一般・講座固有）\n\n" + table(headers, rows)


def section_regressions(runs, items) -> str:
    parts = ["## 5. 学習前後で ok→ng になった質問\n"]
    for before, after in REGRESSION_PAIRS:
        down = [i for i in items if runs[before][i]["ok"] and not runs[after][i]["ok"]]
        up = sum(1 for i in items if not runs[before][i]["ok"] and runs[after][i]["ok"])
        parts.append(f"### {before} → {after}（ok→ng {len(down)} 問、ng→ok {up} 問）\n")
        rows = []
        for i in down:
            mark = "（前の ok は目視で誤判定）" if ABSTAIN_REVIEW.get(before, {}).get(i, ("",))[0] == FALSE_HIT else ""
            rows.append([i, cell(items[i]["question"]), f"{runs[before][i]['outcome']} → {runs[after][i]['outcome']}{mark}", cell(clip(runs[after][i]["raw"], 90))])
        parts.append(table(["id", "質問", "outcome", "学習後の原文（先頭）"], rows))
        parts.append("")
    return "\n".join(parts).rstrip()


def section_training(train, summaries) -> str:
    a, b = train["A"], train["B"]
    rows = [
        ["学習データの件数", a["n_train"], b["n_train"]],
        ["エポック", f"{a['epoch']:.0f}", f"{b['epoch']:.0f}"],
        ["ステップ（実行 / 予定）", f"{a['steps']} / {a['expected_steps']}", f"{b['steps']} / {b['expected_steps']}"],
        ["warmup / 保存の間隔", f"{a['warmup_steps']} / {a['save_steps']}", f"{b['warmup_steps']} / {b['save_steps']}"],
        ["学習時間（秒）", a["seconds"], b["seconds"]],
        ["最終 loss（最後に記録した値）", f"{a['final_loss']:.4f}", f"{b['final_loss']:.4f}"],
        ["平均 loss（学習全体）", f"{a['mean_loss']:.4f}", f"{b['mean_loss']:.4f}"],
        ["trainable%", f"{a['trainable_pct']:.3f}", f"{b['trainable_pct']:.3f}"],
        ["学習するパラメータ / 全体", f"{a['trainable_params']:,} / {a['all_params']:,}", f"{b['trainable_params']:,} / {b['all_params']:,}"],
        ["fp16 / bf16", f"{a['fp16']} / {a['bf16']}", f"{b['fp16']} / {b['bf16']}"],
        ["生成の時間（検索なし、116 問、秒）", summaries["ft_A"]["seconds"], summaries["ft_B"]["seconds"]],
        ["生成の時間（検索あり、116 問、秒）", summaries["ft_A_rag"]["seconds"], summaries["ft_B_rag"]["seconds"]],
    ]
    note = f"学習前の生成時間（116 問）は、`base` が {summaries['base']['seconds']} 秒、`base_rag` が {summaries['base_rag']['seconds']} 秒です（A の回）。"
    return "## 6. A と B の学習の数字\n\n" + table(["項目", "A", "B"], rows) + "\n\n" + note


def section_reproducibility(runs, rerun, items, results, search_path) -> str:
    rows = []
    for label in ("base", "base_rag"):
        same = sum(runs[label][i]["raw"] == rerun[label][i]["raw"] for i in items)
        outcome_same = sum(runs[label][i]["outcome"] == rerun[label][i]["outcome"] for i in items)
        rows.append([label, f"{same}/{len(items)}", f"{outcome_same}/{len(items)}"])
    rescored = {
        label: sum(
            {k: r[k] for k in ("outcome", "ok", "rougeL")} != score(items[i], r["raw"]) for i, r in runs[label].items()
        )
        for label in LABELS
    }
    assert not any(rescored.values()), rescored
    gpu = {results[v]["gpu"] for v in "AB"}
    commits = {results[v]["repo_commit"] for v in "AB"}
    d_same = all(
        runs[x][i]["raw"] == runs[y][i]["raw"]
        for x, y in (("base", "base_rag"), ("ft_A", "ft_A_rag"), ("ft_B", "ft_B_rag"))
        for i in ("D1", "D2", "D3", "D4")
    )
    return "\n".join(
        [
            "## 7. 再現性",
            "",
            "ノートブックの `RUN_BASELINES` の行を `VARIANT == \"B\"` に書き換えて実行したため、B の回でも学習前の評価が走りました。"
            "その出力は `colab/B_run_baselines/` にあります。集計の `base` と `base_rag` には A の回の結果を使っています。",
            "",
            table(["ラベル", "raw が一致した問題", "outcome が一致した問題"], rows),
            "",
            f"- **学習前の生成は、別のセッションでも 116 問すべて同じ文字列でした。** GPU は両方とも {', '.join(sorted(gpu))}、コミットは {', '.join(sorted(commits))} です。",
            "- 確かめられたのは、同じ機種での学習前の貪欲生成だけです。学習は A・B とも 1 回ずつなので、学習の再現性は確かめていません。",
            "- 保存された `outcome`・`ok`・`rougeL` は、いまの `scripts/glossary_eval.py` と `eval_set.json` で採点し直しても、6 ラベルの全行で一致しました。",
            f"- D は検索を使わないので、検索の有無が違うだけのラベル同士で原文が一致します（確認結果：{'全問一致' if d_same else '不一致あり'}）。",
            f"- 検索の結果は `{search_path.name}` を使いました。",
        ]
    )


def section_d(runs, items) -> str:
    parts = ["## 8. D（普通の質問）の原文\n"]
    parts.append(
        "「目視」は原文を読んだ判定で、正解率には入れていません。検索の有無で原文は変わらないので、目視の判定は `ft_A` などの 3 組で共通です。\n"
    )
    for i in ("D1", "D2", "D3", "D4"):
        must = " かつ ".join("／".join(g) for g in items[i]["must_include"])
        parts.append(f"### {i}：{cell(items[i]['question'])}（必須語句：{cell(must)}）\n")
        rows = []
        for label in LABELS:
            r = runs[label][i]
            review = D_REVIEW[(label.replace("_rag", ""), i)]
            rows.append([label, r["outcome"], review, cell(r["raw"])])
        parts.append(table(["条件", "outcome", "目視", "原文"], rows))
        parts.append("")
    clean = {m: sum(D_REVIEW[(m, i)] == "問題なし" for i in ("D1", "D2", "D3", "D4")) for m in ("base", "ft_A", "ft_B")}
    parts.append(
        table(
            ["モデル", "必須語句で correct", "目視で問題なし"],
            [[m, f"{sum(runs[m][i]['ok'] for i in ('D1', 'D2', 'D3', 'D4'))}/4", f"{clean[m]}/4"] for m in ("base", "ft_A", "ft_B")],
        )
    )
    parts.append("")
    parts.append(
        "必須語句だけで見ると `ft_A` と `ft_B` は 2/4 ですが、`ft_A` の D1・D4 と `ft_B` の D1 は語句が合っているだけで、続きが壊れています。"
        "`base` の D2・D3 も語句は合っていますが、中身に誤りがあります。D は 4 問だけなので、目視の差も傾向の目安にとどまります。"
    )
    return "\n".join(parts)


def section_rouge(runs, items) -> str:
    parts = [
        "## 9. 必須語句の判定と ROUGE-L の食い違い\n",
        "`correct` なのに ROUGE-L < 0.2、`wrong` か `over_refusal` なのに ROUGE-L > 0.8 の答えを、条件ごとに最大 3 件ずつ並べました。"
        "閾値に当たる答えが 3 件に満たないときは、次に近い答えを「参考」として足しています（correct は 0.5 未満、NG は 0.5 超のものだけ）。\n",
    ]
    for label in LABELS:
        answers = [r for r in runs[label].values() if r["rougeL"] is not None]
        low_all = sorted((r for r in answers if r["outcome"] == "correct"), key=lambda r: r["rougeL"])
        high_all = sorted((r for r in answers if r["outcome"] in ("wrong", "over_refusal")), key=lambda r: -r["rougeL"])
        n_low = sum(r["rougeL"] < 0.2 for r in low_all)
        n_high = sum(r["rougeL"] > 0.8 for r in high_all)
        parts.append(f"### {label}（correct かつ < 0.2：{n_low} 件、wrong・over_refusal かつ > 0.8：{n_high} 件）\n")
        rows = []
        for r in (x for x in low_all[:3] if x["rougeL"] < 0.5):
            kind = "correct < 0.2" if r["rougeL"] < 0.2 else "参考"
            rows.append([r["id"], kind, f"{r['rougeL']:.2f}", r["outcome"], "—", cell(clip(r["raw"], 90))])
        for r in (x for x in high_all[:3] if x["rougeL"] > 0.5):
            kind = "NG > 0.8" if r["rougeL"] > 0.8 else "参考"
            rows.append([r["id"], kind, f"{r['rougeL']:.2f}", r["outcome"], cell(missing_groups(items[r["id"]], r["raw"])), cell(clip(r["raw"], 90))])
        parts.append(table(["id", "種類", "rougeL", "outcome", "足りない語句", "原文（先頭）"], rows))
        parts.append("")
    ft_b_correct = [r for r in runs["ft_B"].values() if r["outcome"] == "correct" and r["rougeL"] is not None]
    ft_b_verbatim = sum(r["raw"] == items[r["id"]]["definition"] for r in ft_b_correct)
    parts.append(
        "\n".join(
            [
                "読むときの要点：",
                "",
                "- **`wrong` で ROUGE-L > 0.8 の 7 件のうち 6 件は、採点が厳しすぎるのではなく、定義の一部が実際に変わっています。** "
                "`ft_A` の B4-01 は「基盤モデル」が「モデル」に、`ft_A_rag` の B4-23 は「大量の汎用テキスト」が「広く使うテキスト」に、"
                "B4-08 は「次のトークンを選ぶときのランダムさ」が「次の質問を返すときの回答の多様さ」に変わっています。"
                "`ft_A_rag` の B4-44 と `ft_B_rag` の B4-09 は定義の先頭の文が落ち、`ft_B_rag` の B4-19 は「取り込むよう」が落ちています。"
                "文字の一致が高くても意味が変わるので、ROUGE-L だけでは正誤を判断できません。",
                "- 残りの 1 件、`ft_B` の B3-4（0.97）は定義文をほぼそのまま返していますが、質問が求める用語名を答えていません。名前を問う B3 では、ROUGE-L は参考になりません。",
                "- `ft_A` の B4-44（0.91、`over_refusal`）は、定義は正しいのに断る表現に当たった誤判定です（10 節）。",
                "- `correct` で ROUGE-L < 0.2 は `base` の A1（0.13）だけです。原文は「知識の容量」を「低ランクに固定化」するという誤った説明で、語句が合っただけです。",
                f"- **`ft_B` の answer 型の `correct` {len(ft_b_correct)} 問のうち、定義文と完全に一致するのは {ft_b_verbatim} 問です。** "
                "学習していない聞き方でも正しい定義文を呼び出せたことは示しますが、自分の言葉で説明できたことは示しません。",
            ]
        )
    )
    return "\n".join(parts)


def section_abstain_review(runs, items, eval_set) -> str:
    standard = eval_set["abstain_answer"]
    parts = [
        "## 10. 断る判定になった答えの原文と、正規表現の誤判定\n",
        "正規表現で `abstained` か `over_refusal` になった答えを、条件ごとに並べました。原文が定型文（「" + standard + "」）だけの答えは、id だけを書いています。\n",
        "目視の分類：**断りのみ**（用語集にないと述べているだけ）、**断り＋説明**（断る文と説明の両方がある。判定は分かれ得る）、**誤判定**（断っていない）。\n",
    ]
    counts_rows = []
    for label in LABELS:
        judged = {i: r for i, r in runs[label].items() if r["outcome"] in ("abstained", "over_refusal")}
        exact = [i for i, r in judged.items() if r["raw"] == standard]
        others = {i: r for i, r in judged.items() if r["raw"] != standard}
        reviews = ABSTAIN_REVIEW.get(label, {})
        assert set(reviews) == set(others), (label, set(reviews) ^ set(others))
        parts.append(f"### {label}（{len(judged)} 件）\n")
        if exact:
            ab = [i for i in exact if judged[i]["outcome"] == "abstained"]
            over = [i for i in exact if judged[i]["outcome"] == "over_refusal"]
            parts.append(f"定型文だけの答え（{len(exact)} 件）：`abstained` {', '.join(ab) or 'なし'} ／ `over_refusal` {', '.join(over) or 'なし'}\n")
        if others:
            rows = []
            for i, r in others.items():
                matched = [m.group() for p in eval_set["abstain_patterns"] for m in [re.search(p, r["raw"])] if m]
                rows.append([i, r["outcome"], reviews[i][0], cell(reviews[i][1]), cell(" ／ ".join(matched)), cell(r["raw"])])
            parts.append(table(["id", "outcome", "目視", "メモ", "一致した表現", "原文"], rows))
            parts.append("")
        clean = len(exact) + sum(v[0] == CLEAN for v in reviews.values())
        mixed = sum(v[0] == MIXED for v in reviews.values())
        wrong = sum(v[0] == FALSE_HIT for v in reviews.values())
        counts_rows.append([label, len(judged), clean, mixed, wrong, len(MISSED_ABSTAINS.get(label, {}))])

    missed_rows = [
        [label, i, runs[label][i]["outcome"], review[0], cell(review[1]), cell(clip(runs[label][i]["raw"], 120))]
        for label, reviews in MISSED_ABSTAINS.items()
        for i, review in reviews.items()
    ]
    corrected_rows = []
    for label in LABELS:
        changes = CORRECTED_OUTCOMES.get(label, {})
        before = sum(r["ok"] for r in runs[label].values())
        after = before + sum((new in OK_OUTCOMES) - runs[label][i]["ok"] for i, new in changes.items())
        detail = "、".join(f"{i}：{runs[label][i]['outcome']}→{new}" for i, new in changes.items()) or "なし"
        corrected_rows.append([label, f"{before}/116", f"{after}/116", detail])

    parts += [
        "### 判定の集計\n",
        table(["条件", "断る判定", "断りのみ", "断り＋説明", "誤判定", "見逃し"], counts_rows),
        "",
        "### 正規表現の見逃し\n",
        "`asserted` と `wrong` のうち、用語集や否定の語を含む原文を読んで見つけました。\n",
        table(["条件", "id", "outcome", "目視", "メモ", "原文（先頭）"], missed_rows),
        "",
        "### 誤判定と見逃しを直した場合\n",
        "誤判定を外し、見逃しを足した場合の正解数です。「断り＋説明」は正規表現の判定のままにしています。**1〜3 節の表はこの補正を入れていません。**\n",
        table(["条件", "正規表現", "目視で補正", "変わる問題"], corrected_rows),
        "",
        "報告：",
        "",
        "- **`ft_B` と `ft_B_rag` に誤判定はありません。** 断る判定はすべて定型文そのものです。B 版の数字は正規表現の影響を受けていません。",
        "- 誤判定は 4 件です。4 件とも system プロンプトの言い回し「用語集にない用語」が本文に混ざったもので、`用語集に(は)?ない` に当たりました。",
        "- 見逃しは 4 件です。ok が変わるのは `ft_A_rag` の C4-13 だけで、残りは `wrong` が `over_refusal` に変わるだけです。",
        "- 補正で正解数が変わるのは `base`（−1）、`ft_A`（+1）、`ft_A_rag`（+1）の 1 問ずつで、条件の順位は変わりません。",
        "- **system プロンプトの指示は「とだけ答えてください」ですが、学習前の断りの多くは「断り＋説明」です。** 断る文を含むだけで `abstained` になるので、`base` と `base_rag` の C の正解率は、きちんと断れた割合より少し高めに出ています。",
    ]
    return "\n".join(parts)


def section_b_version(runs, items, search) -> str:
    rows = []
    for i in ("B2-1", "B2-5", "B3-1", "B3-3"):
        hit = search[i]["hit_terms"] or "（見つからない）"
        rows.append(
            [i, cell(items[i]["question"]), cell(search[i]["expected_term"]), cell(hit)]
            + [runs[label][i]["outcome"] for label in ("ft_A", "ft_B", "ft_B_rag")]
            + [cell(clip(runs["ft_B_rag"][i]["raw"], 70))]
        )
    parts = [
        "## 11. B 版の over_refusal と、C で断定した問題\n",
        "### B 版の over_refusal\n",
        table(["id", "質問", "正解の語", "検索の結果", "ft_A", "ft_B", "ft_B_rag", "ft_B_rag の原文（先頭）"], rows),
        "",
        "- `ft_B` の 4 問は、原文がすべて定型文です。別名の手本から外した B2 の 6 語のうち、断ったのは「Low-Rank Adaptation」「HITL」の 2 語でした（B2-3・B2-4・B2-6 は correct、B2-2 は wrong）。",
        "- `ft_B_rag` では、検索が正しい語を返した B2-5 は correct になりました。B2-1 も検索は正しい語を返し、断らなくなりましたが、必須語句（固定）が足りず wrong です。",
        "- B3-1 は検索が別の語（事前学習）を返し、B3-3 は検索が見つかりません。どちらも検索結果に正解の語がなく、`ft_B_rag` でも断っています。",
        "",
        "### C の小区分ごとの断れた数\n",
    ]
    columns = ["search"] + LABELS
    rows = []
    for name, sub in SUB_GROUPS:
        ids = [i for i, item in items.items() if item["sub"] == sub]
        row = [name, len(ids), f"{sum(search[i]['not_found'] for i in ids)}"]
        row += [f"{sum(runs[label][i]['ok'] for i in ids)}" for label in LABELS]
        rows.append(row)
    parts.append(table(["小区分", "n"] + columns, rows))
    parts.append("")
    parts.append("`search` は「見つかりませんでした」を返した数、他は `abstained` の数です。\n")

    parts.append("### C の問題ごとの結果\n")
    parts.append("○ は断れた（`abstained`）、× は断定した（`asserted`）です。`search` の列は、当たってしまった語か、○（見つからない）です。\n")
    for name, sub in SUB_GROUPS:
        ids = [i for i, item in items.items() if item["sub"] == sub]
        rows = []
        for i in ids:
            hit = "○" if search[i]["not_found"] else "× " + cell(search[i]["hit_terms"])
            rows.append([i, cell(stem(items[i]["question"])), hit] + ["○" if runs[label][i]["ok"] else "×" for label in LABELS])
        parts.append(f"#### {name}\n")
        parts.append(table(["id", "語"] + columns, rows))
        parts.append("")

    parts.append("### B 版で断定した答えの原文\n")
    rows = []
    for name, sub in SUB_GROUPS:
        for i, item in items.items():
            if item["sub"] != sub:
                continue
            for label in ("ft_B", "ft_B_rag"):
                if runs[label][i]["outcome"] == "asserted":
                    rows.append([name, i, label, cell(stem(item["question"])), cell(clip(runs[label][i]["raw"], 90))])
    parts.append(table(["小区分", "id", "条件", "語", "原文（先頭）"], rows))
    return "\n".join(parts)


def build_summary(runs, rerun, summaries, items, eval_set, train, results, search_path, search) -> str:
    sections = [
        "# 第2回 宿題（発展）集計\n\n"
        "`uv run --with matplotlib python -m scripts.plot_lec02` で生成しました。手で編集すると、次の実行で消えます。\n\n"
        "- 条件はすべて `Qwen/Qwen3.5-0.8B`（revision 固定）、貪欲生成、同じ 116 問です。評価セットの版は "
        f"{results['A']['eval_set_version']}、GPU は {results['A']['gpu']} です。\n"
        "- 1〜6 節の数字は保存された採点の値です。8 節と 10 節の「目視」は原文を読んだ判定で、正解率には混ぜていません。\n"
        "- 図は `figures/` にあり、最後の節に載せています。",
        section_rates(runs, summaries, search),
        section_hand_vs_mechanical(runs, search),
        section_outcomes(summaries),
        section_scope(summaries, search),
        section_regressions(runs, items),
        section_training(train, summaries),
        section_reproducibility(runs, rerun, items, results, search_path),
        section_d(runs, items),
        section_rouge(runs, items),
        section_abstain_review(runs, items, eval_set),
        section_b_version(runs, items, search),
        "## 図\n\n"
        "![Correct rate by condition and question group](figures/fig1_rate_by_category.png)\n\n"
        "![Outcome breakdown by condition](figures/fig2_outcomes.png)\n\n"
        "![Training loss](figures/fig3_training_loss.png)",
    ]
    return "\n\n".join(sections) + "\n"


# ---------- 図 ----------


def _bar_style(condition: str) -> dict:
    return {"color": COLORS[condition], "edgecolor": SURFACE, "hatch": HATCHES.get(condition), "linewidth": 0.8}


def _rate_panel(ax, groups: list[tuple[str, list[str]]], summaries, counts) -> None:
    """区分のまとまりごとに、すき間を空けて棒を並べる。"""
    width = 0.125
    position = 0.0
    ticks, tick_labels = [], []
    for name, categories in groups:
        start = position
        for category in categories:
            n = counts[category]["top"][1] if category in counts else summaries["base"]["by_category"][category]["n"]
            for k, condition in enumerate(CONDITIONS):
                x = position + (k - 3) * width
                if condition == "search":
                    if category not in counts:
                        ax.text(x, 2, "n/a", ha="center", va="bottom", fontsize=6.5, color=INK_2, rotation=90)
                        continue
                    ok, total = counts[category]["top"]
                else:
                    group = summaries[condition]["by_category"][category]
                    ok, total = group["ok"], group["n"]
                rate = 100 * ok / total
                ax.bar(x, rate, width=width, zorder=3, **_bar_style(condition))
                if rate == 0:  # 0% の棒は見えないので、基線に短い印を付ける
                    ax.hlines(0, x - width / 2, x + width / 2, color=COLORS[condition], linewidth=2.5, zorder=4, clip_on=False)
            ticks.append(position)
            tick_labels.append(f"{category}\nn={n}")
            position += 1.0
        if name:
            ax.text((start + position - 1) / 2, 108, name, ha="center", va="bottom", fontsize=9, color=INK_2)
            ax.hlines(106, start - 0.45, position - 0.55, color=INK_2, linewidth=0.8)
        position += 0.3
    ax.set_xticks(ticks, tick_labels)
    ax.set_xlim(-0.6, position - 0.7)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", zorder=0)
    ax.set_ylabel("Correct rate (%)")


def fig1_rate_by_category(summaries, search) -> plt.Figure:
    counts = search_counts(search)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.8))
    _rate_panel(axes[0], [("seen", ["A"]), ("hand-made paraphrases", ["B1", "B2", "B3"]), ("mechanical", ["B4"])], summaries, counts)
    _rate_panel(axes[1], [("hand-made terms not in glossary", ["C1", "C2", "C3"]), ("mechanical", ["C4"]), ("general", ["D"])], summaries, counts)
    axes[0].set_title("Terms in the glossary (A, B): is the answer correct?", loc="left", fontsize=10)
    axes[1].set_title("Terms not in the glossary (C): did it abstain?  General questions (D)", loc="left", fontsize=10)
    handles = [Patch(facecolor=COLORS[c], edgecolor=SURFACE, hatch=HATCHES.get(c), label=c) for c in CONDITIONS]
    fig.legend(handles=handles, loc="upper center", ncol=7, fontsize=9, columnspacing=1.0, handlelength=1.6, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Correct rate by condition and question group", x=0.02, ha="left", fontsize=11)
    fig.text(0.02, 0.012, "search generates nothing: bars show hit@1 for A/B and the 'not found' rate for C. Each hand-made group has only 4–6 questions.", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.945))
    return fig


def fig2_outcomes(summaries) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    y_positions = list(range(len(LABELS)))[::-1]
    for label, y in zip(LABELS, y_positions):
        s = summaries[label]
        left = 0
        for outcome in OUTCOMES:
            value = s["outcomes"].get(outcome, 0)
            if not value:
                continue
            style = OUTCOME_STYLES[outcome]
            ax.barh(y, value, left=left, height=0.62, color=style["color"], edgecolor=SURFACE, hatch=style["hatch"], linewidth=1.2, zorder=3)
            if value >= 3:
                ax.text(
                    left + value / 2, y, str(value), ha="center", va="center", fontsize=8.5, color=style["text"], zorder=4,
                    bbox={"facecolor": style["color"], "edgecolor": "none", "pad": 1.0} if style["hatch"] else None,
                )  # fmt: skip
            left += value
        ok = sum(s["outcomes"].get(o, 0) for o in OK_OUTCOMES)
        ax.vlines(ok, y - 0.38, y + 0.38, color=INK, linewidth=1.8, zorder=5)
        ax.text(118, y, f"correct rate {ok / s['n']:.0%}", ha="left", va="center", fontsize=9, color=INK)
    ax.set_yticks(y_positions, LABELS)
    ax.set_xlim(0, 116)
    ax.set_xticks([0, 29, 58, 87, 116])
    ax.set_xlabel("Number of questions (black tick: correct outcomes to its left, errors to its right)")
    ax.grid(axis="x", zorder=0)
    ax.set_title("Outcome breakdown by condition (116 questions)", loc="left", pad=30)
    handles = [
        Patch(facecolor=OUTCOME_STYLES[o]["color"], edgecolor=SURFACE, hatch=OUTCOME_STYLES[o]["hatch"], label=o)
        for o in OUTCOMES
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=5, fontsize=9, columnspacing=1.2, handlelength=1.6)
    fig.tight_layout()
    fig.subplots_adjust(right=0.84)
    return fig


def fig3_training_loss(logs: dict[str, list[dict]], train: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    markers = {"A": "o", "B": "s"}
    for variant, history in logs.items():
        points = [(e["step"], e["loss"]) for e in history if "loss" in e]
        steps, losses = zip(*points)
        condition = f"ft_{variant}"
        info = train[variant]
        ax.plot(
            steps, losses, color=COLORS[condition], linewidth=2, marker=markers[variant], markersize=4.5,
            markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=3,
            label=f"{condition}: {info['n_train']} rows × {info['epoch']:.0f} epochs",
        )  # fmt: skip
        ax.annotate(
            f"{condition}  {losses[-1]:.3f}", xy=(steps[-1], losses[-1]), xytext=(8, 10 if variant == "A" else -2),
            textcoords="offset points", fontsize=9, va="center", color=INK,
        )  # fmt: skip
    ax.set_xlim(0, 215)
    ax.set_ylim(0, None)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss")
    ax.grid(True, zorder=0)
    ax.set_title("Training loss (logged every 5 steps)", loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def main() -> None:
    eval_set = load_eval_set()
    items = {i["id"]: i for i in eval_set["items"]}
    runs = load_runs(COLAB_DIR, LABELS)
    rerun = load_runs(BASELINE_RERUN_DIR, ["base", "base_rag"])
    results = {v: load_json(COLAB_DIR / f"result_{v}.json") for v in "AB"}
    train = {v: load_json(COLAB_DIR / f"train_{v}.json") for v in "AB"}
    logs = {v: load_json(COLAB_DIR / f"log_{v}.json") for v in "AB"}
    search_path, search = load_search()

    # colab/ 直下の base が本当に A の回のものかを確かめる（B の回の結果を混ぜない）
    for label in ("base", "base_rag"):
        saved = load_json(COLAB_DIR / f"eval_{label}.json")["summary"]["seconds"]
        assert saved == results["A"]["summaries"][label]["seconds"], label
    summaries = {label: summarize(list(runs[label].values())) for label in LABELS}
    for label in LABELS:
        saved = load_json(COLAB_DIR / f"eval_{label}.json")["summary"]
        summaries[label]["seconds"] = saved["seconds"]

    SUMMARY_MD.write_text(
        build_summary(runs, rerun, summaries, items, eval_set, train, results, search_path, search),
        encoding="utf-8",
    )
    print(f"[保存] {SUMMARY_MD}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "fig1_rate_by_category.png": fig1_rate_by_category(summaries, search),
        "fig2_outcomes.png": fig2_outcomes(summaries),
        "fig3_training_loss.png": fig3_training_loss(logs, train),
    }
    for name, fig in figures.items():
        path = FIG_DIR / name
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        print(f"[保存] {path}")


if __name__ == "__main__":
    main()
