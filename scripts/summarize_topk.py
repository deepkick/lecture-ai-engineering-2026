"""第5回 宿題（基礎）: eval_topk.py の記録（JSONL）を、質問×K と K ごとに集計する。

使い方:
    uv run python -m scripts.summarize_topk lectures/05/results/topk5_*.jsonl lectures/05/results/topk10_*.jsonl

出力（lectures/05/results/）:
    topk_summary.csv      質問×K の行（row_type=run）と、K ごとのまとめ（row_type=mean / median / min / max）
    topk_summary.md       質問×K の表、K ごとのまとめ表、取得ページが変わった質問の一覧
    labels_template.csv   labels.csv がないときだけ作る、正しさの判定用のひな形

外れ値は除外しない。失敗した行は件数だけ数え、平均などは成功した行だけで計算する。
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

from scripts.eval_topk import QUESTIONS, RESULTS, load_questions, read_rows, required_hit

# (列名, 表の見出し, 小数の桁数)
METRICS = [
    ("required_rate", "必須グループの取得率", 2),
    ("n_pages", "取得ページ数", 1),
    ("n_searches", "検索の回数", 1),
    ("cited_from_retrieved", "引用の出どころ（取得ページにある割合）", 2),
    ("required_cited_rate", "必須ページの引用率", 2),
    ("seconds", "時間（秒）", 1),
    ("input_tokens", "入力トークン", 0),
    ("output_tokens", "出力トークン", 0),
    ("requests", "モデル応答回数", 1),
    ("cost_usd", "費用（USD）", 6),
]
KEY_POINT_IDS = ["a", "b", "c", "d"]
OVERALL = ["正答", "一部正答", "誤答"]
LABEL_COLUMNS = (["qid", "top_k", "answer_head"] + [f"kp_{k}" for k in KEY_POINT_IDS]
                 + ["overall", "note", "timestamp"])
CSV_COLUMNS = (["row_type", "qid", "top_k", "ok", "error", "required_hit"] + [m[0] for m in METRICS]
               + ["n_cited", "overall", "timestamp", "file"])


def row_metrics(row: dict, question: dict) -> dict:
    """JSONLの1行から、質問×K の表の1行を作る。必須ページの判定は questions.json から計算し直す。"""
    groups = question["required"]
    pages, cited = row["pages_full"], row["cited"]
    hit = required_hit(groups, pages)
    return {
        "row_type": "run", "qid": row["qid"], "top_k": row["top_k"], "ok": row["ok"], "error": row["error"] or "",
        "required_hit": f"{sum(hit)}/{len(hit)}",
        "required_rate": sum(hit) / len(hit),
        "n_pages": len(pages),
        "n_searches": len(row["search_queries"]),
        # 引用が1件もない回は割合を決められないので、0ではなく空欄にする
        "cited_from_retrieved": sum(c in pages for c in cited) / len(cited) if cited else None,
        "required_cited_rate": sum(required_hit(groups, cited)) / len(groups),
        "seconds": row["seconds"], "input_tokens": row["input_tokens"], "output_tokens": row["output_tokens"],
        "requests": row["requests"], "cost_usd": row["cost_usd"],
        "n_cited": len(cited), "overall": "", "timestamp": row["timestamp"], "file": row.get("_file", ""),
    }


def summarize_k(runs: list[dict]) -> dict:
    """K ごとに、成功した行だけで 平均・中央値・最小・最大 を計算する。値のない行（算定不可など）は入れない。"""
    summary = {}
    for k in sorted({run["top_k"] for run in runs}):
        ok_runs = [run for run in runs if run["top_k"] == k and run["ok"]]
        stats = {}
        for name, _, _ in METRICS:
            values = [run[name] for run in ok_runs if run[name] is not None]
            stats[name] = {
                "n": len(values),
                "mean": statistics.mean(values) if values else None,
                "median": statistics.median(values) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
        summary[k] = {
            "n_ok": len(ok_runs),
            "n_failed": sum(run["top_k"] == k and not run["ok"] for run in runs),
            "stats": stats,
            "overall": {label: sum(run["overall"] == label for run in ok_runs) for label in OVERALL},
            "n_labeled": sum(run["overall"] in OVERALL for run in ok_runs),
        }
    return summary


def page_diff(rows: list[dict], questions: dict) -> list[dict]:
    """質問ごとに、最小のKと最大Kで取得ページを比べる。同じ質問×Kに成功行が複数あれば最後の行を使う。"""
    ks = sorted({row["top_k"] for row in rows})
    if len(ks) < 2:
        return []
    low, high = ks[0], ks[-1]
    last = {(row["qid"], row["top_k"]): row for row in rows if row["ok"]}
    diffs = []
    for qid, question in questions.items():
        a, b = last.get((qid, low)), last.get((qid, high))
        if a is None or b is None:
            continue
        required = {page for group in question["required"] for page in group}
        diffs.append({
            "qid": qid, "low": low, "high": high, "required": required,
            "only_low": [p for p in a["pages_full"] if p not in b["pages_full"]],
            "only_high": [p for p in b["pages_full"] if p not in a["pages_full"]],
            "n_common": sum(p in b["pages_full"] for p in a["pages_full"]),
            "queries_low": a["search_queries"], "queries_high": b["search_queries"],
        })
    return diffs


def load_labels(path: Path) -> dict:
    """labels.csv を読む。キーは (qid, top_k, timestamp)。timestamp が空の行は (qid, top_k, "")。"""
    labels = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for line in csv.DictReader(f):
            if not (line.get("qid") or "").strip():
                continue
            key = (line["qid"].strip(), int(line["top_k"]), (line.get("timestamp") or "").strip())
            labels[key] = (line.get("overall") or "").strip()
    return labels


def apply_labels(runs: list[dict], labels: dict) -> list[str]:
    """overall を各行に付ける。時刻が一致する行を優先し、なければ (qid, K) で引く。問題があれば警告を返す。"""
    by_pair = {}
    for (qid, k, _), overall in labels.items():
        by_pair.setdefault((qid, k), []).append(overall)
    warnings = []
    for run in runs:
        if not run["ok"]:
            continue
        overall = labels.get((run["qid"], run["top_k"], run["timestamp"]))
        if overall is None:
            candidates = by_pair.get((run["qid"], run["top_k"]), [])
            overall = candidates[0] if len(candidates) == 1 else None
        if overall is None:
            warnings.append(f"{run['qid']} K={run['top_k']}: labels.csv に対応する行がありません")
        elif overall not in OVERALL:
            warnings.append(f"{run['qid']} K={run['top_k']}: overall が「{overall}」です（{'／'.join(OVERALL)} のどれか）")
        else:
            run["overall"] = overall
    return warnings


def write_label_template(path: Path, rows: list[dict], questions: dict) -> bool:
    """判定用のひな形を書く。すでに記入が始まっているひな形は上書きしない。"""
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as f:
            if any((line.get("overall") or "").strip() or (line.get("note") or "").strip()
                   for line in csv.DictReader(f)):
                return False
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        for row in rows:
            if not row["ok"]:
                continue
            points = {point["id"] for point in questions[row["qid"]]["key_points"]}
            line = {"qid": row["qid"], "top_k": row["top_k"],
                    "answer_head": " ".join(row["answer"].split())[:200],
                    "overall": "", "note": "", "timestamp": row["timestamp"]}
            line.update({f"kp_{k}": "" if k in points else "-" for k in KEY_POINT_IDS})
            writer.writerow(line)
    return True


def fmt(value, digits: int) -> str:
    if value is None:
        return "—"
    return f"{value:,.{digits}f}"


def write_csv(path: Path, runs: list[dict], summary: dict) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for run in runs:
            writer.writerow({key: ("" if value is None else value) for key, value in run.items()})
        for k, info in summary.items():
            for stat in ("mean", "median", "min", "max"):
                line = {"row_type": stat, "qid": "ALL", "top_k": k, "ok": info["n_ok"]}
                line.update({name: ("" if s[stat] is None else s[stat]) for name, s in info["stats"].items()})
                writer.writerow(line)


def build_markdown(runs: list[dict], summary: dict, diffs: list[dict], files: list[str], has_labels: bool) -> str:
    out = ["# RAG_TOP_K の変更前後の集計", ""]
    out.append("入力: " + "、".join(f"`{name}`" for name in files))
    out.append("")
    out.append("外れ値は除外していない。失敗した行は件数だけ数え、平均などは成功した行だけで計算した。"
               "「—」は値がないことを表す（引用が0件、費用が算定不可など）。")

    out += ["", "## 質問×K", ""]
    head = ["qid", "K", "結果", "必須取得", "取得ページ", "検索", "引用数", "引用の出どころ", "必須引用",
            "秒", "入力tok", "出力tok", "応答", "費用USD"] + (["正しさ"] if has_labels else [])
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "---|" * len(head))
    for run in sorted(runs, key=lambda r: (r["qid"], r["top_k"], r["timestamp"])):
        if run["ok"]:
            cells = [run["qid"], str(run["top_k"]), "成功", f"{run['required_hit']}（{run['required_rate']:.2f}）",
                     str(run["n_pages"]), str(run["n_searches"]), str(run["n_cited"]),
                     fmt(run["cited_from_retrieved"], 2), fmt(run["required_cited_rate"], 2),
                     fmt(run["seconds"], 1), fmt(run["input_tokens"], 0), fmt(run["output_tokens"], 0),
                     fmt(run["requests"], 0), fmt(run["cost_usd"], 6)]
            cells += [run["overall"] or "—"] if has_labels else []
        else:
            cells = [run["qid"], str(run["top_k"]), f"失敗: {run['error']}"] + ["—"] * (len(head) - 3)
        out.append("| " + " | ".join(cells) + " |")

    out += ["", "## K ごとのまとめ", "", "各セルは「平均（中央値、最小〜最大）」。", ""]
    ks = list(summary)
    out.append("| 指標 | " + " | ".join(f"K={k}" for k in ks) + " |")
    out.append("|---|" + "---|" * len(ks))
    out.append("| 成功した行 / 失敗した行 | "
               + " | ".join(f"{summary[k]['n_ok']} / {summary[k]['n_failed']}" for k in ks) + " |")
    for name, label, digits in METRICS:
        cells = []
        for k in ks:
            s = summary[k]["stats"][name]
            if s["n"] == 0:
                cells.append("—")
                continue
            cell = (f"{fmt(s['mean'], digits)}（{fmt(s['median'], digits)}、"
                    f"{fmt(s['min'], digits)}〜{fmt(s['max'], digits)}）")
            if s["n"] < summary[k]["n_ok"]:
                cell += f" n={s['n']}"
            cells.append(cell)
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    if has_labels:
        for label in OVERALL:
            out.append(f"| {label} | " + " | ".join(
                f"{summary[k]['overall'][label]} / {summary[k]['n_labeled']}" for k in ks) + " |")

    if diffs:
        low, high = diffs[0]["low"], diffs[0]["high"]
        out += ["", f"## K={low} と K={high} で取得ページが変わった質問", "",
                "★ は必須の根拠ページ。検索語はエージェントが毎回決めるので、"
                "ページの差には K の違いと検索語の違いの両方が含まれる。", ""]

        def show(pages, required):
            return "、".join(("★" if p in required else "") + p for p in pages) or "なし"

        same = []
        for d in diffs:
            if not d["only_low"] and not d["only_high"]:
                same.append(d["qid"])
                continue
            out.append(f"### {d['qid']}")
            out.append(f"- 共通: {d['n_common']} ページ")
            out.append(f"- K={low} だけ: {show(d['only_low'], d['required'])}")
            out.append(f"- K={high} だけ: {show(d['only_high'], d['required'])}")
            out.append(f"- K={low} の検索語: {' / '.join(d['queries_low']) or 'なし'}")
            out.append(f"- K={high} の検索語: {' / '.join(d['queries_high']) or 'なし'}")
            out.append("")
        out.append("取得ページが変わらなかった質問: " + ("、".join(same) or "なし"))
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="eval_topk.py の記録を集計する")
    parser.add_argument("jsonl", nargs="+", type=Path, help="本番の記録（topk*_*.jsonl）")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--labels", type=Path, default=RESULTS / "labels.csv")
    parser.add_argument("--outdir", type=Path, default=RESULTS)
    args = parser.parse_args()

    if smoke := [path.name for path in args.jsonl if path.name.startswith("smoke_")]:
        parser.error(f"動作確認の記録は集計に含めません: {', '.join(smoke)}")
    questions = {q["qid"]: q for q in load_questions(args.questions)}
    rows = []
    for path in args.jsonl:
        for row in read_rows(path):
            rows.append({**row, "_file": path.name})
    if unknown := sorted({row["qid"] for row in rows} - set(questions)):
        parser.error(f"questions.json にない質問ID: {', '.join(unknown)}")
    rows.sort(key=lambda row: (row["top_k"], row["timestamp"]))
    runs = [row_metrics(row, questions[row["qid"]]) for row in rows]

    has_labels = args.labels.is_file()
    warnings = apply_labels(runs, load_labels(args.labels)) if has_labels else []
    summary = summarize_k(runs)
    diffs = page_diff(rows, questions)

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "topk_summary.csv", runs, summary)
    markdown = build_markdown(runs, summary, diffs, [path.name for path in args.jsonl], has_labels)
    (args.outdir / "topk_summary.md").write_text(markdown, encoding="utf-8")

    print(markdown)
    for warning in warnings:
        print(f"[注意] {warning}")
    print(f"[出力] {args.outdir / 'topk_summary.csv'}")
    print(f"[出力] {args.outdir / 'topk_summary.md'}")
    if has_labels:
        print(f"[labels] {args.labels} の overall を集計に加えました")
    else:
        template = args.outdir / "labels_template.csv"
        if write_label_template(template, rows, questions):
            print(f"[labels] ひな形を作りました: {template}（記入して labels.csv として保存）")
        else:
            print(f"[labels] 記入済みのひな形があるので上書きしません: {template}（labels.csv として保存してください）")


if __name__ == "__main__":
    main()
