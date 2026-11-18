"""第5回 宿題（基礎）: 検索件数 RAG_TOP_K の変更前後を、同じ質問で記録する。

基本RAGのエージェント（app/agent.py の agent）に評価用の質問を1問ずつ答えさせ、
検索語・取得ページ・回答・引用・時間・トークン数を JSONL に残す。
K はこのスクリプトでは変えない。app/config.py の RAG_TOP_K を書き換えて切り替える。

使い方:
    uv run python -m scripts.eval_topk                                # 見積もりだけ（APIを呼ばない）
    uv run python -m scripts.eval_topk --confirm                      # 本番
    uv run python -m scripts.eval_topk --smoke --qids V1 --confirm    # 動作確認（ファイル名が smoke_ で始まる）
    uv run python -m scripts.eval_topk --confirm --resume lectures/05/results/topk5_20261118-001500.jsonl
"""

import argparse
import contextlib
import io
import json
import os
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from pydantic_ai import capture_run_messages
from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.usage import RunUsage, UsageLimits

from app import retrieval
from app.agent import agent, build_model, print_trace
from app.config import MODEL, api_error_message

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "lectures/05/eval/questions.json"
RESULTS = ROOT / "lectures/05/results"

REQUEST_LIMIT = 10              # 1つの質問で許すモデル応答の回数
GAP_SECONDS = 15                # 質問と質問の間隔
EST_SECONDS = 25                # 見積もり用の、1問あたりの所要時間（動作確認の実測 約26秒から）
RETRY_WAIT = {429: 60, 503: 20}  # このHTTPステータスは、待ってから同じ試行を1回だけやり直す
CUTOFF = "上限で打ち切り"


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def page_id(lecture: int, page: int) -> str:
    return f"lec{lecture:02d}-main-p{page:03d}"


_CITE_TOKEN = re.compile(r"第\s*(\d+)\s*回|(?<![A-Za-z])p{1,2}\s*\.?\s*(\d+)", re.IGNORECASE)
_CITE_RANGE = re.compile(r"\s*[-‐–—〜~]\s*(?:p{1,2}\s*\.?\s*)?(\d+)", re.IGNORECASE)
# 「p.13、39」のように p. を省いた続き。「p.13、3つの…」の 3 は数に入れない
_CITE_MORE = re.compile(r"\s*(?:[、,・/]|と|および)\s*(\d+)(?!\d|\s*[回つ個件点章倍種%])")


def parse_cited(answer: str) -> list[str]:
    """回答中の「第N回・PDF p.N」を page_id の列にする（出てきた順、重複なし）。

    全角は半角にそろえる。「第2回 p.52」「第3回・PDF p.13、p.39」「p.39〜41」にも対応する。
    回が書かれていないページ番号は、直前に出た「第N回」のものとして扱う。
    回が一度も出ていないページ番号は、どの資料か決められないので数えない。
    """
    text = unicodedata.normalize("NFKC", answer or "")
    cited: list[str] = []
    lecture = None

    def add(page: int) -> None:
        pid = page_id(lecture, page)
        if pid not in cited:
            cited.append(pid)

    pos = 0
    while match := _CITE_TOKEN.search(text, pos):
        pos = match.end()
        if match.group(1):
            lecture = int(match.group(1))
            continue
        if lecture is None:
            continue
        last = int(match.group(2))
        add(last)
        while True:
            if (more := _CITE_RANGE.match(text, pos)) and last < int(more.group(1)) <= last + 20:
                for page in range(last + 1, int(more.group(1)) + 1):
                    add(page)
            elif more := _CITE_MORE.match(text, pos):
                add(int(more.group(1)))
            else:
                break
            last = int(more.group(1))
            pos = more.end()
    return cited


def required_hit(required: list[list[str]], pages: list[str]) -> list[bool]:
    """required の各グループについて、グループ内のどれか1ページが pages にあるか。"""
    return [any(page in pages for page in group) for group in required]


def extract_search(messages) -> tuple[list[str], list[str], list[str]]:
    """メッセージ列から、search_materials の検索語・返されたpage_id・ツールが返したエラーを取り出す。"""
    queries, pages, errors = [], [], []
    for message in messages:
        for part in message.parts:
            if getattr(part, "tool_name", None) != "search_materials":
                continue
            if isinstance(part, ToolCallPart):
                queries.append(str(part.args_as_dict().get("query", "")))
            elif isinstance(part, ToolReturnPart):
                try:
                    data = json.loads(part.content) if isinstance(part.content, str) else part.content
                    if data.get("error"):
                        errors.append(str(data["error"]))
                    for hit in data.get("results", []):
                        if hit["page_id"] not in pages:
                            pages.append(hit["page_id"])
                except (ValueError, AttributeError, KeyError, TypeError):
                    errors.append(f"返り値を読めません: {str(part.content)[:80]}")
    return queries, pages, errors


def _safe(text: str) -> str:
    """エラー文を記録用に短くする。万一APIキーが含まれていたら伏せる。"""
    key = os.getenv("GEMINI_API_KEY")
    if key:
        text = text.replace(key, "***")
    return text[:300]


def _tool_error_status(errors: list[str]) -> int | None:
    """search_materials は埋め込みAPIのエラーを文にして返す。その文から 429 / 503 を見分ける。"""
    joined = " ".join(errors)
    if "レート制限" in joined:
        return 429
    if "503" in joined:
        return 503
    return None


def attempt(question: dict, top_k: int) -> tuple[dict, str, int | None]:
    """1問を1回実行し、(JSONLの1行, traceファイルの本文, やり直しの判断に使うHTTPステータス) を返す。"""
    usage = RunUsage()
    result, error, status = None, None, None
    started = perf_counter()
    with capture_run_messages() as captured:
        try:
            result = agent.run_sync(
                question["question"], model=build_model(),
                usage_limits=UsageLimits(request_limit=REQUEST_LIMIT), usage=usage,
            )
        except UsageLimitExceeded:
            error = CUTOFF
        except ModelHTTPError as e:
            status = e.status_code
            error = _safe(api_error_message(e.status_code, str(e.body)[:200]))
        except Exception as e:  # 通信の切断など。記録して次の質問へ進む
            error = _safe(f"{type(e).__name__}: {e}")
    seconds = perf_counter() - started

    messages = result.all_messages() if result is not None else captured
    usage = result.usage if result is not None else usage
    queries, pages, tool_errors = extract_search(messages)
    if error is None and tool_errors:
        # 検索が失敗したまま出た回答は「根拠なし」の成功に見えるので、失敗として扱う
        error = _safe("検索ツールのエラー: " + " / ".join(tool_errors))
        status = _tool_error_status(tool_errors)

    answer = str(result.output) if result is not None else ""
    measured = usage.requests > 0
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": MODEL,
        "top_k": top_k,
        "qid": question["qid"],
        "ok": error is None,
        "error": error,
        "retried": False,
        "answer": answer,
        "search_queries": queries,
        "pages_full": pages,
        "required_hit": required_hit(question["required"], pages),
        "cited": parse_cited(answer),
        "input_tokens": usage.input_tokens if measured else None,
        "output_tokens": usage.output_tokens if measured else None,
        "requests": usage.requests if measured else None,
        "cost_usd": float(usage.cost) if measured and usage.cost is not None else None,
        "seconds": round(seconds, 2),
    }

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        print(f"[qid] {question['qid']}  [top_k] {top_k}  [model] {MODEL}  [timestamp] {record['timestamp']}")
        print(f"[question] {question['question']}")
        print(f"[ok] {record['ok']}" + (f"  [error] {error}" if error else ""))
        print("\n--- answer ---")
        print(answer or "（回答なし）")
        # app.agent の --trace と同じ表示。失敗した回も、そこまでのツール呼び出しを残す
        print_trace(SimpleNamespace(all_messages=lambda: messages, usage=usage), seconds)
    return record, buffer.getvalue(), status


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def done_qids(rows: list[dict]) -> set[str]:
    """再開時に飛ばす質問。成功した質問と、上限で打ち切りになった質問（これも1つの結果）。"""
    return {row["qid"] for row in rows if row["ok"] or row["error"] == CUTOFF}


def print_estimate(pending: list[dict], top_k: int, out_path: Path) -> None:
    n = len(pending)
    seconds = n * EST_SECONDS + max(0, n - 1) * GAP_SECONDS
    print(f"[model] {MODEL}")
    print(f"[RAG_TOP_K] {top_k}（app/config.py の現在の値）")
    print(f"[質問] {n} 問: {', '.join(q['qid'] for q in pending) or 'なし'}")
    print(f"[実行回数] {n} 回（1問につき1回）")
    print(f"[推定リクエスト] 生成 約{n * 2}〜{n * 3} 回（上限 {n * REQUEST_LIMIT} 回）、"
          f"検索 約{n}〜{n * 2} 回（1回の検索でトークン数の計測と埋め込みを1回ずつ）")
    print(f"[推定所要時間] 約{seconds // 60}分{seconds % 60:02d}秒（1問 約{EST_SECONDS}秒 ＋ 間隔 {GAP_SECONDS}秒）")
    print(f"[出力] {out_path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG_TOP_K の変更前後を比べるための記録を取る")
    parser.add_argument("--qids", nargs="+", help="実行する質問のID（既定は全問）")
    parser.add_argument("--smoke", action="store_true", help="動作確認。ファイル名の先頭を smoke_ にする")
    parser.add_argument("--confirm", action="store_true", help="付けると実行する。付けなければ見積もりだけ")
    parser.add_argument("--resume", type=Path, help="途中で止まった記録（jsonl）の続きから実行する")
    args = parser.parse_args()

    top_k = retrieval.RAG_TOP_K
    questions = load_questions()
    if args.qids:
        known = {q["qid"] for q in questions}
        if unknown := [qid for qid in args.qids if qid not in known]:
            parser.error(f"questions.json にない質問ID: {', '.join(unknown)}")
        questions = [q for q in questions if q["qid"] in args.qids]

    if args.resume:
        out_path = args.resume.resolve()
        rows = read_rows(out_path)
        if any(row["top_k"] != top_k or row["model"] != MODEL for row in rows):
            parser.error(f"{out_path.name} の top_k・model が現在の設定（K={top_k}, {MODEL}）と違います")
        skip = done_qids(rows)
        pending = [q for q in questions if q["qid"] not in skip]
    else:
        prefix = ("smoke_" if args.smoke else "") + f"topk{top_k}"
        out_path = RESULTS / f"{prefix}_{datetime.now():%Y%m%d-%H%M%S}.jsonl"
        pending = questions
    # traceのファイル名は、jsonlと同じ接頭辞・同じ実行時刻にする
    prefix, stamp = out_path.stem.rsplit("_", 1)
    trace_dir = out_path.parent / "trace"

    print_estimate(pending, top_k, out_path)
    if not args.confirm:
        print("\n見積もりだけを表示しました。実行するには --confirm を付けてください。")
        return
    if not pending:
        print("\n実行する質問がありません。")
        return

    trace_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for i, question in enumerate(pending):
            if i:
                time.sleep(GAP_SECONDS)
            record, trace, status = attempt(question, top_k)
            if status in RETRY_WAIT:
                print(f"[{question['qid']}] HTTP {status}。{RETRY_WAIT[status]}秒待って、1回だけやり直します")
                time.sleep(RETRY_WAIT[status])
                record, trace, status = attempt(question, top_k)
                record["retried"] = True
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            trace_path = trace_dir / f"{prefix}_{question['qid']}_{stamp}.txt"
            trace_path.write_text(trace, encoding="utf-8")

            hit = record["required_hit"]
            print(f"[{i + 1}/{len(pending)}] {question['qid']}: "
                  f"{'ok' if record['ok'] else '失敗（' + record['error'] + '）'} / "
                  f"必須 {sum(hit)}/{len(hit)} / 検索 {len(record['search_queries'])}回 / "
                  f"{len(record['pages_full'])}ページ / {record['seconds']}秒")
            if status == 429:
                print("\nやり直しても 429 でした。ここで止めます（ここまでの記録は残っています）。")
                print("時間を置いてから、次のコマンドで続きから再開できます:")
                print(f"  uv run python -m scripts.eval_topk --confirm --resume {out_path.relative_to(ROOT)}")
                raise SystemExit(2)
    print(f"\n記録: {out_path.relative_to(ROOT)}")
    print(f"trace: {trace_dir.relative_to(ROOT)}/{prefix}_<qid>_{stamp}.txt")


if __name__ == "__main__":
    main()
