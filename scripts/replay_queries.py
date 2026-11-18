"""第5回 宿題（基礎）: 本番でエージェントが使った検索語を検索し直し、必須ページの順位を記録する。

本番（eval_topk.py）では検索語をエージェントが毎回決めるので、K=5 と K=10 の取得ページの差には
「K の違い」と「検索語の違い」が混ざる。ここでは検索語を固定し、上位20件までの順位を取って、
同じ検索語なら K=5 で取れたか・K=10 で取れたかを分けて見る。生成APIは使わない（埋め込みだけ）。

使い方:
    uv run python -m scripts.replay_queries             # 見積もりだけ（APIを呼ばない）
    uv run python -m scripts.replay_queries --confirm   # 実行。途中で止まっても、同じコマンドで続きから

入力は lectures/05/results/topk*_*.jsonl（引数で指定もできる）の成功行。出力（lectures/05/results/）:
    replay_queries_ranks.json   検索語ごとの上位20件（page_id と距離）。再実行時はここにある検索語を飛ばす
    replay_queries.csv          検索語 × 必須グループごとの最良順位、5位以内か、10位以内か
    replay_queries.md           質問ごとに「同じ検索語で K=5 なら取れたか、K=10 なら取れたか」をまとめた表
"""

import argparse
import csv
import json
import time
from pathlib import Path

import sqlite_vec
from google.genai.errors import APIError

from app import retrieval
from app.config import EMBEDDING_MODEL, RAG_DB, api_error_message
from scripts.eval_topk import QUESTIONS, RESULTS, RETRY_WAIT, load_questions, read_rows, required_hit

TOP_N = 20        # 順位を取る件数
GAP_SECONDS = 1   # 検索と検索の間隔
CSV_COLUMNS = ["qid", "top_k", "query_no", "query", "kind", "group", "pages", "best_rank", "best_page",
               "in_top5", "in_top10", "page_ranks"]


def rank_query(query: str, limit: int = TOP_N) -> list[dict]:
    """検索語を埋め込み、近い順に上位 limit 件の page_id と距離を返す。

    app.retrieval.search と同じ埋め込み（embed_text）・同じ距離（vec_distance_cosine）・同じ並び順で、
    返す件数だけが違う。
    """
    with retrieval.open_index(RAG_DB) as db:
        vector = retrieval.embed_text(query.strip(), "RETRIEVAL_QUERY")
        rows = db.execute("""
            SELECT page_id, vec_distance_cosine(embedding, ?) AS distance
            FROM chunks ORDER BY distance, page_id LIMIT ?
        """, (sqlite_vec.serialize_float32(vector), limit))
        return [{"page_id": row["page_id"], "distance": round(row["distance"], 6)} for row in rows]


def collect_runs(rows: list[dict]) -> list[dict]:
    """本番の成功行から、質問・元のK・検索語（行の中の重複を除く）・取得ページを取り出す。"""
    runs = []
    for row in rows:
        if row["ok"]:
            runs.append({"qid": row["qid"], "top_k": row["top_k"], "pages_full": row["pages_full"],
                         "queries": list(dict.fromkeys(row["search_queries"]))})
    return sorted(runs, key=lambda run: (run["qid"], run["top_k"]))


def targets(question: dict) -> list[dict]:
    """順位を調べる対象。必須グループと、補助ページ（required にない key_points・supplementary のページ）。"""
    required = [{"kind": "required", "group": i, "pages": group}
                for i, group in enumerate(question["required"], 1)]
    in_required = {page for group in question["required"] for page in group}
    extra = [page for point in question.get("key_points", []) for page in point.get("pages", [])]
    extra += question.get("supplementary", [])
    supplementary = [{"kind": "supplementary", "group": i, "pages": [page]}
                     for i, page in enumerate(dict.fromkeys(p for p in extra if p not in in_required), 1)]
    return required + supplementary


def best_rank(pages: list[str], ranking: list[dict]) -> tuple[int | None, str | None]:
    """pages のうち最も上位のページの順位（1始まり）とそのページ。上位 TOP_N 件になければ (None, None)。"""
    for rank, hit in enumerate(ranking, 1):
        if hit["page_id"] in pages:
            return rank, hit["page_id"]
    return None, None


def judge(runs: list[dict], questions: dict, ranks: dict) -> list[dict]:
    """検索語 × 対象グループごとの行を作る。検索し直せなかった検索語は行にしない。"""
    lines = []
    for run in runs:
        for number, query in enumerate(run["queries"], 1):
            ranking = ranks.get(query)
            if ranking is None:
                continue
            position = {hit["page_id"]: rank for rank, hit in enumerate(ranking, 1)}
            for target in targets(questions[run["qid"]]):
                rank, page = best_rank(target["pages"], ranking)
                lines.append({
                    "qid": run["qid"], "top_k": run["top_k"], "query_no": number, "query": query,
                    "kind": target["kind"], "group": target["group"], "pages": ";".join(target["pages"]),
                    "best_rank": rank, "best_page": page or "",
                    "in_top5": rank is not None and rank <= 5, "in_top10": rank is not None and rank <= 10,
                    "page_ranks": ";".join(f"{p}:{position.get(p, f'>{TOP_N}')}" for p in target["pages"]),
                })
    return lines


def run_summary(run: dict, question: dict, ranks: dict) -> dict:
    """1回の本番実行について、その検索語をまとめて使ったときに各必須グループが何件目までに入るか。"""
    rankings = [ranks.get(query) for query in run["queries"]]
    complete = bool(rankings) and all(ranking is not None for ranking in rankings)
    groups = []
    for group in question["required"]:
        found = [best_rank(group, ranking) for ranking in rankings if ranking is not None]
        found = [(rank, page) for rank, page in found if rank is not None]
        groups.append(min(found) if found else (None, None))
    replayed = list(dict.fromkeys(hit["page_id"] for ranking in rankings if ranking is not None
                                  for hit in ranking[:run["top_k"]]))
    return {
        "complete": complete, "groups": groups,
        "hit_at": {k: sum(rank is not None and rank <= k for rank, _ in groups) for k in (5, 10, TOP_N)},
        "actual": sum(required_hit(question["required"], run["pages_full"])),
        # 元のKで検索し直した結果が、本番の取得ページと同じか（再検索が本番を再現できているかの確認）
        "reproduced": complete and set(replayed) == set(run["pages_full"]),
    }


def short(page_id: str) -> str:
    return f"第{int(page_id[3:5])}回 p.{int(page_id[-3:])}"


def show_rank(rank: int | None) -> str:
    return f"{rank}位" if rank is not None else f"{TOP_N + 1}位以下"


def build_markdown(runs: list[dict], questions: dict, ranks: dict, failed: dict) -> str:
    out = ["# 本番の検索語で検索し直したときの、必須ページの順位", "",
           f"本番（成功行）でエージェントが使った検索語を1つずつ検索し直し、上位{TOP_N}件の順位を取った。"
           "埋め込みと距離の計算は本番と同じ。生成APIは使っていない。", "",
           "## 同じ検索語で、K=5 なら取れたか・K=10 なら取れたか", "",
           "1回の実行で検索語が複数あるときは、どれかの検索語で入れば「取れた」と数える（本番と同じ数え方）。"
           "「本番」は、その実行で実際に取れた必須グループの数。"
           "「再現」は、元の K で検索し直した結果が本番の取得ページと一致したか。", "",
           "| qid | 検索語の出どころ | 検索語の数 | K=5 なら | K=10 なら | "
           f"上位{TOP_N}件なら | 本番 | 再現 |",
           "|---|---|---|---|---|---|---|---|"]
    summaries = {}
    for run in runs:
        question = questions[run["qid"]]
        s = summaries[(run["qid"], run["top_k"])] = run_summary(run, question, ranks)
        n = len(question["required"])
        if not s["complete"]:
            out.append(f"| {run['qid']} | K={run['top_k']} の本番 | {len(run['queries'])} | "
                       "検索し直せなかった検索語がある | — | — | — | — |")
            continue
        out.append(f"| {run['qid']} | K={run['top_k']} の本番 | {len(run['queries'])} | {s['hit_at'][5]}/{n} | "
                   f"{s['hit_at'][10]}/{n} | {s['hit_at'][TOP_N]}/{n} | {s['actual']}/{n} | "
                   f"{'一致' if s['reproduced'] else '不一致'} |")

    out += ["", "## 必須グループごとの最良順位", "",
            "その実行の検索語のうち、最も上位に入ったときの順位。", "",
            "| qid | 検索語の出どころ | 必須グループ | 最良順位（ページ） | K=5 なら | K=10 なら |",
            "|---|---|---|---|---|---|"]
    for run in runs:
        s = summaries[(run["qid"], run["top_k"])]
        for group, (rank, page) in zip(questions[run["qid"]]["required"], s["groups"]):
            where = f"{show_rank(rank)}（{short(page)}）" if rank is not None else show_rank(rank)
            out.append(f"| {run['qid']} | K={run['top_k']} の本番 | {' か '.join(short(p) for p in group)} | {where} | "
                       f"{'○' if rank is not None and rank <= 5 else '×'} | "
                       f"{'○' if rank is not None and rank <= 10 else '×'} |")

    out += ["", "## 検索語ごとの順位", ""]
    for run in runs:
        for query in run["queries"]:
            out.append(f"### {run['qid']}（K={run['top_k']} の本番）「{query}」")
            ranking = ranks.get(query)
            if ranking is None:
                out += [f"- 検索し直せなかった: {failed.get(query, '未実行')}", ""]
                continue
            position = {hit["page_id"]: rank for rank, hit in enumerate(ranking, 1)}
            for target in targets(questions[run["qid"]]):
                label = "必須" if target["kind"] == "required" else "補助"
                pages = "、".join(f"{short(p)}: {show_rank(position.get(p))}" for p in target["pages"])
                out.append(f"- {label}{target['group']}: {pages}")
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def write_outputs(outdir: Path, runs: list[dict], questions: dict, ranks: dict, failed: dict) -> str:
    with (outdir / "replay_queries.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for line in judge(runs, questions, ranks):
            writer.writerow({key: ("" if value is None else value) for key, value in line.items()})
    markdown = build_markdown(runs, questions, ranks, failed)
    (outdir / "replay_queries.md").write_text(markdown, encoding="utf-8")
    return markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="本番の検索語を検索し直し、必須ページの順位を記録する")
    parser.add_argument("jsonl", nargs="*", type=Path, help="本番の記録（既定は results/topk*_*.jsonl）")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--outdir", type=Path, default=RESULTS)
    parser.add_argument("--confirm", action="store_true", help="付けると実行する。付けなければ見積もりだけ")
    args = parser.parse_args()

    paths = args.jsonl or sorted(RESULTS.glob("topk*_*.jsonl"))
    if smoke := [path.name for path in paths if path.name.startswith("smoke_")]:
        parser.error(f"動作確認の記録は使いません: {', '.join(smoke)}")
    questions = {q["qid"]: q for q in load_questions(args.questions)}
    runs = collect_runs([row for path in paths for row in read_rows(path)])
    queries = list(dict.fromkeys(query for run in runs for query in run["queries"]))

    ranks_path = args.outdir / "replay_queries_ranks.json"
    ranks = json.loads(ranks_path.read_text(encoding="utf-8")) if ranks_path.is_file() else {}
    todo = [query for query in queries if query not in ranks]

    print(f"[入力] {', '.join(path.name for path in paths) or 'なし'}")
    for k in sorted({run["top_k"] for run in runs}):
        n = len({query for run in runs if run["top_k"] == k for query in run["queries"]})
        print(f"[検索語] K={k} の本番（成功 {sum(run['top_k'] == k for run in runs)} 行）: {n} 個")
    print(f"[検索語] 重複を除いて {len(queries)} 個（記録済み {len(queries) - len(todo)} 個、これから検索 {len(todo)} 個）")
    print(f"[埋め込みAPI] {len(todo)} 回（{EMBEDDING_MODEL}。ほかにトークン数の計測が {len(todo)} 回。生成APIは 0 回）")
    seconds = len(todo) * 2 + max(0, len(todo) - 1) * GAP_SECONDS
    print(f"[推定所要時間] 約{seconds}秒（1回 約2秒 ＋ 間隔 {GAP_SECONDS}秒）")
    print(f"[順位] 検索語ごとに上位 {TOP_N} 件（検索DB: {RAG_DB.name}）")
    print(f"[出力] {args.outdir / 'replay_queries.csv'}、replay_queries.md、replay_queries_ranks.json")
    if not args.confirm:
        print("\n見積もりだけを表示しました。実行するには --confirm を付けてください。")
        return

    args.outdir.mkdir(parents=True, exist_ok=True)
    failed, stopped = {}, False
    for i, query in enumerate(todo):
        if i:
            time.sleep(GAP_SECONDS)
        for retry in (False, True):
            try:
                ranks[query] = rank_query(query)
                failed.pop(query, None)
                break
            except APIError as e:
                failed[query] = api_error_message(e.code)
                if retry or e.code not in RETRY_WAIT:
                    stopped = e.code == 429
                    break
                print(f"[{i + 1}/{len(todo)}] HTTP {e.code}。{RETRY_WAIT[e.code]}秒待って、1回だけやり直します")
                time.sleep(RETRY_WAIT[e.code])
            except (ValueError, OSError) as e:
                failed[query] = str(e)[:200]
                break
        ranks_path.write_text(json.dumps(ranks, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"[{i + 1}/{len(todo)}] {'失敗（' + failed[query] + '）' if query in failed else 'ok'}: {query}")
        if stopped:
            print("\nやり直しても 429 でした。ここで止めます（ここまでの順位は残っています）。")
            print("時間を置いてから、同じコマンドで続きから再開できます。")
            break

    markdown = write_outputs(args.outdir, runs, questions, ranks, failed)
    print()
    print(markdown)
    if failed:
        print(f"[注意] 検索し直せなかった検索語が {len(failed)} 個あります。同じコマンドでやり直せます。")
    if stopped:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
