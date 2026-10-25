"""第1回 宿題（発展）: 応答時間とトークン数を計測し、モデルや質問で比較する。

使い方:
    # 見積もりだけ表示する（合計2リクエスト以上は --confirm がないと API を呼ばない）
    uv run python -m scripts.measure_latency --mode grid
    uv run python -m scripts.measure_latency --mode burst

    # 実験A: 複数のモデル × 質問 × 繰り返し（無料枠の RPM を超えない間隔で送る）
    uv run python -m scripts.measure_latency --mode grid --confirm
    # 実験B: 同じ質問を間隔なしで送り、レート制限が効くタイミングを観測する
    uv run python -m scripts.measure_latency --mode burst --confirm                  # 20本同時
    uv run python -m scripts.measure_latency --mode burst --concurrency 1 --confirm  # 順次

    # 集計して lectures/01/results/summary.md に書き出す
    uv run python -m scripts.measure_latency --summarize lectures/01/results/raw_*.csv

結果は lectures/01/results/raw_<mode>_<日時>.csv に1リクエスト1行で保存する。
app.ask と同じく素の generate_content を呼ぶが、応答時間を測るために直接呼び出している。
"""

import argparse
import csv
import math
import statistics
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from google import genai
from google.genai import errors

from app.ask import DEFAULT_QUESTION
from app.config import api_error_message, get_api_key

RESULTS_DIR = Path(__file__).resolve().parent.parent / "lectures" / "01" / "results"

QUESTIONS = {
    "Q1": "LoRA とは何ですか？",
    "Q2": DEFAULT_QUESTION,
    "Q3": "Transformer の仕組みを初学者向けに1000字程度で説明してください。",
}
GRID_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash"]
BURST_MODEL = "gemini-3.5-flash-lite"
# 無料枠の上限（lite は 15 RPM、gemini-3.8-flash は 5 RPM）を超えないリクエスト間隔（秒）
INTERVALS = {"gemini-3.5-flash-lite": 5.0, "gemini-3.1-flash-lite": 5.0, "gemini-3.8-flash": 15.0}
DEFAULT_INTERVAL = 15.0  # 上の表にないモデルは安全側の間隔にする
# grid で一時的なエラーが返ったときに待つ秒数（待ってから同じ試行を1回だけやり直す）
BACKOFF = {"429": 60.0, "503": 20.0}
ASSUMED_LATENCY = 5.0  # 見積もり用の1リクエストあたりの応答時間（仮定）

COLUMNS = [
    "timestamp", "elapsed_from_start_s", "mode", "model", "question_id", "rep",
    "latency_s", "prompt_tokens", "output_tokens", "thoughts_tokens", "total_tokens",
    "output_tokens_per_s", "status", "waited_s", "error",
]  # fmt: skip


class Request(NamedTuple):
    model: str
    question_id: str
    rep: int
    interval: float  # 前のリクエストの送信開始から空ける秒数


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="応答時間とトークン数を計測する")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--mode", choices=["grid", "burst"], help="grid=実験A, burst=実験B")
    action.add_argument("--summarize", nargs="+", metavar="CSV", help="CSV を集計する")
    parser.add_argument("--models", nargs="+", help="使うモデル（burst は先頭の1つ）")
    parser.add_argument("--questions", nargs="+", choices=list(QUESTIONS), help="使う質問（burst は先頭の1つ）")
    parser.add_argument("--reps", type=int, default=3, help="grid の繰り返し回数（既定 3）")
    parser.add_argument("--interval", type=float, help="grid のリクエスト間隔（秒）。全モデル共通で上書きする")
    parser.add_argument("--n", type=int, default=20, help="burst のリクエスト数（既定 20）")
    parser.add_argument("--concurrency", type=int, default=20, help="burst の同時送信数（既定 20。1 なら順次送信）")
    parser.add_argument("--confirm", action="store_true", help="合計2リクエスト以上を実際に送る")
    args = parser.parse_args()
    if args.reps < 1 or args.n < 1 or args.concurrency < 1:
        parser.error("--reps・--n・--concurrency は1以上にしてください")
    return args


def build_plan(args: argparse.Namespace) -> list[Request]:
    """送るリクエストを順番どおりに並べる。grid はモデルごとにまとめる。"""
    if args.mode == "burst":
        model = (args.models or [BURST_MODEL])[0]
        qid = (args.questions or ["Q1"])[0]
        return [Request(model, qid, i, 0.0) for i in range(1, args.n + 1)]

    plan = []
    for model in args.models or GRID_MODELS:
        interval = args.interval if args.interval is not None else INTERVALS.get(model, DEFAULT_INTERVAL)
        for rep in range(1, args.reps + 1):
            for qid in args.questions or list(QUESTIONS):
                plan.append(Request(model, qid, rep, interval))
    return plan


def print_estimate(plan: list[Request], mode: str, concurrency: int = 1) -> None:
    by_model: dict[str, list[Request]] = defaultdict(list)
    for req in plan:
        by_model[req.model].append(req)

    print(f"[見積もり] mode={mode}")
    total_s = 0.0
    for model, reqs in by_model.items():
        if mode == "burst":
            seconds = math.ceil(len(reqs) / concurrency) * ASSUMED_LATENCY
            how = f"同時 {concurrency} 本" if concurrency > 1 else "間隔なしで順次"
        else:
            seconds = len(reqs) * max(reqs[0].interval, ASSUMED_LATENCY)
            how = f"間隔 {reqs[0].interval:g} 秒"
        total_s += seconds
        print(f"  {model}: {len(reqs)} リクエスト（{how}）約 {seconds:.0f} 秒")
    print(f"  合計: {len(plan)} リクエスト / 推定所要時間: 約 {total_s:.0f} 秒（{total_s / 60:.1f} 分）")
    print(f"  ※ 応答時間を1件 {ASSUMED_LATENCY:g} 秒と仮定した概算")
    if mode == "grid":
        print("  ※ 429 は60秒・503 は20秒待って同じ試行を1回だけやり直す（1回につき +1 リクエスト）")
    else:
        print("  ※ 直前に同じモデルを使った場合は、1分以上空けてから実行する（前の利用分が RPM に数えられるため）")


def call_once(client: genai.Client, model: str, question: str) -> dict:
    """1回だけ呼び、応答時間・トークン数・ステータスを返す。エラーでも例外は出さない。"""
    start = time.perf_counter()
    try:
        response = client.models.generate_content(model=model, contents=question)
    except errors.APIError as e:
        status = str(e.code) if e.code in (429, 503) else "other"
        error = api_error_message(e.code, e.message)
        return {"latency_s": time.perf_counter() - start, "status": status, "error": error}
    except Exception as e:  # ネットワーク断なども記録して続行する
        return {"latency_s": time.perf_counter() - start, "status": "other", "error": f"{type(e).__name__}: {e}"}
    latency = time.perf_counter() - start

    usage = response.usage_metadata  # None のことも、各フィールドが None のこともある
    output = getattr(usage, "candidates_token_count", None)
    return {
        "latency_s": latency,
        "prompt_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": output,
        "thoughts_tokens": getattr(usage, "thoughts_token_count", None),
        "total_tokens": getattr(usage, "total_token_count", None),
        "output_tokens_per_s": output / latency if output and latency > 0 else None,
        "status": "ok",
        "error": "",
    }


def run(plan: list[Request], mode: str, retry: bool, concurrency: int = 1) -> Path:
    """計画どおりに送り、1リクエスト1行で CSV に書く（途中で止めてもそこまでは残る）。

    elapsed_from_start_s は送信時刻。完了時刻は elapsed_from_start_s + latency_s で求められる。
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"raw_{mode}_{datetime.now():%Y%m%d-%H%M%S}.csv"
    api_key = get_api_key()

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()

        def record(i: int, req: Request, sent_s: float, result: dict, waited: float = 0.0, is_retry: bool = False):
            if is_retry:
                result["error"] = f"retry: {result['error']}" if result["error"] else "retry"
            writer.writerow(
                {
                    **result,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "elapsed_from_start_s": round(sent_s, 3),
                    "mode": mode,
                    "model": req.model,
                    "question_id": req.question_id,
                    "rep": req.rep,
                    "latency_s": round(result["latency_s"], 3),
                    "output_tokens_per_s": _round(result.get("output_tokens_per_s"), 1),
                    "waited_s": waited,
                    "error": " ".join(result["error"].split())[:100],
                }
            )
            f.flush()
            note = " (retry)" if is_retry else ""
            print(
                f"[{i}/{len(plan)}] {req.model} {req.question_id} rep={req.rep}{note}: "
                f"{result['status']} {result['latency_s']:.2f}s output={result.get('output_tokens')}"
                + (f" → {waited:g}秒待つ" if waited else "")
            )

        t0 = time.perf_counter()
        if mode == "burst" and concurrency > 1:
            _run_concurrent(plan, api_key, concurrency, t0, record)
        else:
            _run_sequential(plan, mode, retry, genai.Client(api_key=api_key), t0, record)
    return path


def _run_sequential(plan: list[Request], mode: str, retry: bool, client: genai.Client, t0: float, record) -> None:
    attempts = 2 if mode == "grid" and retry else 1
    last_start = None
    for i, req in enumerate(plan, 1):
        if last_start is not None:
            wait = req.interval - (time.perf_counter() - last_start)
            if wait > 0:
                time.sleep(wait)

        for attempt in range(1, attempts + 1):
            last_start = time.perf_counter()
            result = call_once(client, req.model, QUESTIONS[req.question_id])
            # grid では 429/503 のあと待つ（最後のリクエストのあとは待たない）
            has_next = attempt < attempts or i < len(plan)
            waited = BACKOFF.get(result["status"], 0.0) if mode == "grid" and has_next else 0.0
            record(i, req, last_start - t0, result, waited, is_retry=attempt > 1)
            if waited:
                time.sleep(waited)
            if result["status"] not in BACKOFF:
                break


def _run_concurrent(plan: list[Request], api_key: str, concurrency: int, t0: float, record) -> None:
    """burst 用: 複数のスレッドから同時に送る。genai.Client はスレッドごとに作る。"""
    local = threading.local()

    def send(req: Request) -> tuple[Request, float, dict]:
        if not hasattr(local, "client"):
            local.client = genai.Client(api_key=api_key)
        sent = time.perf_counter()
        return req, sent - t0, call_once(local.client, req.model, QUESTIONS[req.question_id])

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(send, req) for req in plan]
        # CSV への書き込みはメインスレッドだけで、完了した順に行う
        for i, future in enumerate(as_completed(futures), 1):
            record(i, *future.result())


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


# ---- 集計（標準ライブラリのみ） ----


def _num(text: str | None) -> float | None:
    return float(text) if text not in (None, "") else None


def _stat(values: list[float | None], digits: int = 2) -> str:
    """平均 ± 標準偏差（n<2 なら平均のみ）。"""
    vals = [v for v in values if v is not None]
    if not vals:
        return "–"
    mean = statistics.mean(vals)
    if len(vals) < 2:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {statistics.stdev(vals):.{digits}f}"


def _median(values: list[float | None], digits: int = 2) -> str:
    vals = [v for v in values if v is not None]
    return f"{statistics.median(vals):.{digits}f}" if vals else "–"


def _range(values: list[float | None], digits: int = 2) -> str:
    """最小〜最大（外れ値も除外しない）。"""
    vals = [v for v in values if v is not None]
    return f"{min(vals):.{digits}f}〜{max(vals):.{digits}f}" if vals else "–"


def fit_line(xs: list[float], ys: list[float]) -> tuple[float, float, float | None] | None:
    """y = a + b x を最小二乗で当てはめ、(a, b, R²) を返す。当てはめられなければ None。"""
    if len(xs) < 2 or len(set(xs)) < 2:
        return None
    b, a = statistics.linear_regression(xs, ys)
    r2 = statistics.correlation(xs, ys) ** 2 if len(set(ys)) > 1 else None
    return a, b, r2


def _table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def summarize_grid(rows: list[dict]) -> list[str]:
    models = list(dict.fromkeys(r["model"] for r in rows))
    ok = [r for r in rows if r["status"] == "ok"]
    out = ["## 実験A（grid）", ""]

    out += ["### ステータス件数（全行）", ""]
    status_rows = []
    for m in models:
        c = Counter(r["status"] for r in rows if r["model"] == m)
        retries = sum(1 for r in rows if r["model"] == m and r["error"].startswith("retry"))
        status_rows.append([m, c["ok"], c["429"], c["503"], c["other"], retries, sum(c.values())])
    out += _table(["model", "ok", "429", "503", "other", "うち retry 行", "合計"], status_rows) + [""]

    out += ["### モデル × 質問（status=ok、平均 ± 標準偏差。外れ値も除外しない）", ""]
    stat_rows = []
    for m in models:
        for q in sorted({r["question_id"] for r in ok if r["model"] == m}):
            g = [r for r in ok if r["model"] == m and r["question_id"] == q]
            col = lambda name, d=1: _stat([_num(r[name]) for r in g], d)  # noqa: E731
            latency = [_num(r["latency_s"]) for r in g]
            stat_rows.append(
                [m, q, len(g), col("latency_s", 2), _median(latency), _range(latency), col("prompt_tokens"),
                 col("output_tokens"), col("thoughts_tokens"), col("total_tokens"), col("output_tokens_per_s")]
            )  # fmt: skip
    out += _table(
        ["model", "質問", "n", "latency_s", "latency 中央値", "latency 最小〜最大", "prompt", "output",
         "thoughts", "total", "output tok/s"],
        stat_rows,
    ) + [""]  # fmt: skip

    out += ["### latency = a + b × x の当てはめ（status=ok）", ""]
    fit_rows = []
    for m in models:
        g = [r for r in ok if r["model"] == m and _num(r["output_tokens"]) is not None]
        ys = [_num(r["latency_s"]) for r in g]
        xs_output = [_num(r["output_tokens"]) for r in g]
        candidates = [("output", xs_output)]
        if any(_num(r["thoughts_tokens"]) for r in g):
            candidates.append(("output+thoughts", [x + (_num(r["thoughts_tokens"]) or 0) for x, r in zip(xs_output, g)]))
        for label, xs in candidates:
            fit = fit_line(xs, ys)
            if fit is None:
                fit_rows.append([m, label, len(g), "–", "–", "–"])
            else:
                a, b, r2 = fit
                fit_rows.append([m, label, len(g), f"{a:.3f}", f"{b * 1000:.2f}", "–" if r2 is None else f"{r2:.3f}"])
    out += _table(["model", "x", "n", "a（秒）", "b（ミリ秒/トークン）", "R²"], fit_rows) + [""]

    out += ["### rep=1 とそれ以外の latency 平均（status=ok）", ""]
    rep_rows = []
    for m in models:
        first = [_num(r["latency_s"]) for r in ok if r["model"] == m and r["rep"] == "1"]
        rest = [_num(r["latency_s"]) for r in ok if r["model"] == m and r["rep"] != "1"]
        diff = f"{statistics.mean(first) - statistics.mean(rest):+.2f}" if first and rest else "–"
        median_diff = f"{statistics.median(first) - statistics.median(rest):+.2f}" if first and rest else "–"
        rep_rows.append(
            [m, f"{_stat(first)} (n={len(first)})", _median(first),
             f"{_stat(rest)} (n={len(rest)})", _median(rest), diff, median_diff]
        )  # fmt: skip
    out += _table(
        ["model", "rep=1 平均", "rep=1 中央値", "rep≥2 平均", "rep≥2 中央値", "平均の差（rep=1 − rep≥2）", "中央値の差"],
        rep_rows,
    ) + [""]
    return out


def _max_in_flight(rows: list[dict]) -> int:
    """送信〜完了の区間の重なりから、同時に処理中だったリクエスト数の最大値を推定する。"""
    events = []
    for r in rows:
        sent = _num(r["elapsed_from_start_s"])
        events += [(sent, 1), (sent + _num(r["latency_s"]) - 0.01, -1)]  # 丸め誤差の分だけ完了を早める
    current = peak = 0
    for _, delta in sorted(events):  # 同時刻なら完了（-1）を先に数える
        current += delta
        peak = max(peak, current)
    return peak


def summarize_burst(files: dict[str, list[dict]]) -> list[str]:
    out = [
        "## 実験B（burst）",
        "",
        "最初の 429 は送信時刻が最も早いもの。経過秒は「送信 → 完了」（完了 = 送信 + latency）。",
        "",
    ]
    burst_rows = []
    for name, rows in files.items():
        rows = sorted(rows, key=lambda r: _num(r["elapsed_from_start_s"]))
        c = Counter(r["status"] for r in rows)
        first_429 = next((r for r in rows if r["status"] == "429"), None)
        if first_429:
            sent = _num(first_429["elapsed_from_start_s"])
            when = f"{sent:.2f} → {sent + _num(first_429['latency_s']):.2f}"
            ok_before = sum(1 for r in rows if r["status"] == "ok" and _num(r["elapsed_from_start_s"]) < sent)
        else:
            when, ok_before = "–", c["ok"]
        ok_latency = [_num(r["latency_s"]) for r in rows if r["status"] == "ok"]
        burst_rows.append(
            [name, rows[0]["model"], len(rows), _max_in_flight(rows), c["ok"], c["429"], c["503"], c["other"],
             first_429["rep"] if first_429 else "なし", when, ok_before, _stat(ok_latency)]
        )  # fmt: skip
    out += _table(
        ["ファイル", "model", "n", "最大同時数（推定）", "ok", "429", "503", "other",
         "最初の 429（番号）", "その経過秒", "それより前に送った ok 数", "ok の latency_s"],
        burst_rows,
    ) + [""]  # fmt: skip
    if len(files) >= 2:
        out += compare_bursts(files)
    return out


def compare_bursts(files: dict[str, list[dict]]) -> list[str]:
    """burst の CSV を実行順（ファイル名の日時順）に並べて比べる。"""
    names = sorted(files)
    labels = [f"{i}回目" for i in range(1, len(names) + 1)]
    out = ["### 実行回ごとの比較", "", "、".join(f"{lb} = `{n}`" for lb, n in zip(labels, names)), ""]

    def latencies(rows: list[dict], status: str) -> list[float]:
        return [_num(r["latency_s"]) for r in rows if r["status"] == status]

    def mean(values: list[float]) -> float | None:
        return statistics.mean(values) if values else None

    metrics = [
        ("ok 件数", lambda rows: sum(r["status"] == "ok" for r in rows), "{:d}", "{:+d}"),
        ("429 件数", lambda rows: sum(r["status"] == "429" for r in rows), "{:d}", "{:+d}"),
        ("ok の latency 平均（秒）", lambda rows: mean(latencies(rows, "ok")), "{:.2f}", "{:+.2f}"),
        ("ok の latency 中央値（秒）", lambda rows: (statistics.median(latencies(rows, "ok")) if latencies(rows, "ok") else None), "{:.2f}", "{:+.2f}"),
        ("429 の latency 平均（秒）", lambda rows: mean(latencies(rows, "429")), "{:.2f}", "{:+.2f}"),
    ]  # fmt: skip
    header = ["項目", *labels] + (["差（2回目 − 1回目）"] if len(names) == 2 else [])
    rows_out = []
    for label, fn, fmt, diff_fmt in metrics:
        values = [fn(files[n]) for n in names]
        row = [label] + [fmt.format(v) if v is not None else "–" for v in values]
        if len(names) == 2:
            row.append(diff_fmt.format(values[1] - values[0]) if None not in values else "–")
        rows_out.append(row)

    reps_429 = [{int(r["rep"]) for r in files[n] if r["status"] == "429"} for n in names]
    row = ["429 になった rep"] + [", ".join(map(str, sorted(s))) or "なし" for s in reps_429]
    if len(names) == 2:
        row.append("共通: " + (", ".join(map(str, sorted(reps_429[0] & reps_429[1]))) or "なし"))
    rows_out.append(row)
    out += _table(header, rows_out) + [""]

    out += ["### rep ごとのステータス", ""]
    status = [{int(r["rep"]): r["status"] for r in files[n]} for n in names]
    all_reps = sorted(set().union(*status))
    out += _table(["rep", *labels], [[rep, *(s.get(rep, "–") for s in status)] for rep in all_reps]) + [""]
    return out


def summarize(paths: list[str]) -> Path:
    grid_rows: list[dict] = []
    burst_files: dict[str, list[dict]] = {}
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        grid_rows += [r for r in rows if r["mode"] == "grid"]
        burst = [r for r in rows if r["mode"] == "burst"]
        if burst:
            burst_files[Path(p).name] = burst

    out = [
        "# 第1回 発展: 応答時間とトークン数の計測結果",
        "",
        f"集計日時: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "元データ: " + ", ".join(f"`{Path(p).name}`" for p in paths),
        "",
    ]
    if grid_rows:
        out += summarize_grid(grid_rows)
    if burst_files:
        out += summarize_burst(burst_files)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "summary.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    return path


def main() -> None:
    args = parse_args()
    if args.summarize:
        print(f"\n[保存] {summarize(args.summarize)}")
        return

    plan = build_plan(args)
    print_estimate(plan, args.mode, args.concurrency)
    if len(plan) >= 2 and not args.confirm:
        print("\n--confirm がないため、API は呼ばずに終了します。")
        return

    print()
    # --confirm なしの1リクエストでは、やり直しも含めて API 呼び出しを1回に抑える
    path = run(plan, args.mode, retry=args.confirm, concurrency=args.concurrency)
    print(f"\n[保存] {path}")


if __name__ == "__main__":
    main()
