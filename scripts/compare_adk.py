"""第3回 宿題（発展 a）: Pydantic AI 版と ADK 版のエージェントを、同じ質問で比べる。

使い方（リポジトリ直下で）:
    uv run --with google-adk python -m scripts.compare_adk              # 見積もりだけ表示（API は呼ばない）
    uv run --with google-adk python -m scripts.compare_adk --confirm    # 本番（既定 reps=2 → 20 回の実行）
    uv run --with google-adk python -m scripts.compare_adk --confirm --reps 1

同じ INSTRUCTIONS と同じツール（search_glossary / get_lecture_summary / make_quiz）を持つ
2つのエージェントに、固定の5問を投げる。
  - pydantic_ai: app.agent.agent を build_model() のモデルで実行する（演習の CLI と同じ）
  - adk:         adk_agent/glossary_agent の root_agent を InMemoryRunner で実行する
順番は rep → 質問 → フレームワーク（rep ごとに5問を一巡させる）。
1回の実行ごとに lectures/03/results/adk_compare_<日時>.jsonl へ1行書き出し（途中で止めてもそこまで残る）、
最後に質問 × フレームワークの集計を表示して adk_compare_summary.md にも保存する。

数え方:
  - LLM の呼び出し回数（llm_requests）
      pydantic_ai: result.usage.requests（app/agent.py の print_trace と同じ取り方。usage はプロパティ）
      adk:         before_model_callback が呼ばれた回数を正とする（モデルを呼ぶ直前に必ず1回呼ばれる）。
                   照合用に、usage_metadata を持つ非 partial なイベントの数も llm_events に記録する
                   （ストリーミングの部分イベントで二重に数えないため、こちらは参考値）
  - ツール呼び出し（tool_calls）
      pydantic_ai: result.all_messages() の ToolCallPart
      adk:         各イベントの get_function_calls()
  - トークン（input_tokens / output_tokens）
      pydantic_ai: result.usage.input_tokens / output_tokens。pydantic-ai 2.57.0 は Gemini の
                   candidates_token_count + thoughts_token_count を output_tokens にする（手元で確認済み。
                   思考トークンの内訳は usage.details["thoughts_tokens"]）
      adk:         各イベントの usage_metadata から、prompt_token_count の合計を input_tokens、
                   candidates_token_count + thoughts_token_count の合計を output_tokens にして
                   pydantic_ai と同じ数え方に揃える。思考トークンだけの合計は thoughts_tokens に別記録
  - 最終回答（answer）
      pydantic_ai: result.output
      adk:         is_final_response() が True のイベントの text パートを連結
"""

import argparse
import asyncio
import json
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ToolCallPart

from adk_agent.glossary_agent.agent import root_agent
from app.agent import agent as pydantic_agent
from app.agent import build_model
from app.config import MODEL, api_error_message, get_api_key

QUESTIONS = [
    ("Q1", "AIE2026の講座の宿題にある「3レベル」とは何ですか？"),  # 用語集にある
    ("Q2", "LoRAとQLoRAの違いは？"),  # 用語が2つ
    ("Q3", "第2回の内容から確認問題を1問出して"),  # make_quiz
    ("Q4", "第3回では何を学びますか？"),  # get_lecture_summary
    ("Q5", "VLAとは何ですか？"),  # 用語集にない（VLA は glossary.json に無いことを確認済み）
]
FRAMEWORKS = ("pydantic_ai", "adk")

# 1回の実行で LLM を最大4回呼ぶと見て、無料枠 RPM 15 に 1.2 倍の余裕: 4 回 × 15 秒 = 60 秒あたり 4 回
INTERVAL_S = 15
MAX_LLM_CALLS_PER_RUN = 4
MIN_LLM_CALLS_PER_RUN = 2
RETRY_WAIT_S = {429: 60, 503: 20}  # このステータスなら待って1回だけやり直す
EST_RUN_S = 10  # 1回の実行にかかる時間の目安（見積もり用）

RESULTS_DIR = Path(__file__).resolve().parent.parent / "lectures" / "03" / "results"
SUMMARY_PATH = RESULTS_DIR / "adk_compare_summary.md"
ADK_USER_ID = "compare"


# ----------------------------------------------------------------------------- 実行（フレームワークごと）
def run_pydantic_ai(question: str) -> dict:
    """演習の CLI（app.agent main）と同じく run_sync で実行し、trace と同じ場所から数える。"""
    result = pydantic_agent.run_sync(question, model=build_model())
    tool_calls = [
        {"name": part.tool_name, "args": part.args_as_dict()}
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolCallPart)
    ]
    usage = result.usage  # print_trace と同じ（2.57.0 ではプロパティ）
    return {
        "llm_requests": usage.requests,
        "llm_events": None,  # ADK 専用の照合値
        "tool_calls": tool_calls,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "thoughts_tokens": (usage.details or {}).get("thoughts_tokens"),
        "answer": str(result.output),
    }


async def _run_adk_async(question: str) -> dict:
    llm_calls = 0

    def count_llm_call(callback_context, llm_request):
        """モデルを呼ぶ直前に ADK が呼ぶ。None を返すと通常どおりモデルを呼ぶ。"""
        nonlocal llm_calls
        llm_calls += 1
        return None

    # adk_agent 側は adk run 用に素のまま残し、比較のためのコールバックはコピーに付ける
    agent = root_agent.model_copy(update={"before_model_callback": count_llm_call})
    runner = InMemoryRunner(agent=agent, app_name="glossary_compare")
    # 質問ごとに新しいセッション（履歴を持ち越さない）
    session = await runner.session_service.create_session(app_name=runner.app_name, user_id=ADK_USER_ID)

    tool_calls: list[dict] = []
    llm_events = 0
    input_tokens = output_tokens = thoughts_tokens = 0
    answer_parts: list[str] = []
    try:
        async for event in runner.run_async(
            user_id=ADK_USER_ID,
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=question)]),
        ):
            for call in event.get_function_calls():
                tool_calls.append({"name": call.name, "args": call.args})
            meta = event.usage_metadata
            if meta is not None and not event.partial:
                llm_events += 1
                input_tokens += meta.prompt_token_count or 0
                # pydantic-ai の output_tokens（回答 + 思考）と同じ数え方に揃える
                output_tokens += (meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0)
                thoughts_tokens += meta.thoughts_token_count or 0
            if event.is_final_response() and event.content and event.content.parts:
                answer_parts.extend(p.text for p in event.content.parts if p.text and not p.thought)
    finally:
        await runner.close()

    return {
        "llm_requests": llm_calls,
        "llm_events": llm_events,
        "tool_calls": tool_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thoughts_tokens": thoughts_tokens,
        "answer": "".join(answer_parts),
    }


def run_adk(question: str) -> dict:
    return asyncio.run(_run_adk_async(question))


RUNNERS = {"pydantic_ai": run_pydantic_ai, "adk": run_adk}


# ----------------------------------------------------------------------------- エラーの種別
def http_status(exc: BaseException) -> int | None:
    """例外（とその原因の連鎖）から HTTP ステータスを取り出す。

    pydantic_ai: ModelHTTPError.status_code。 ADK: google.genai.errors.APIError の code。
    """
    seen = set()
    e: BaseException | None = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if isinstance(e, ModelHTTPError):
            return e.status_code
        code = getattr(e, "code", None)
        if isinstance(code, int) and 100 <= code < 600:
            return code
        e = e.__cause__ or e.__context__
    return None


def error_kind(exc: BaseException) -> str:
    status = http_status(exc)
    return f"http_{status}" if status else type(exc).__name__


# ----------------------------------------------------------------------------- 記録
def write_row(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def attempt(rep: int, qid: str, question: str, framework: str, attempt_no: int) -> dict:
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "rep": rep,
        "question_id": qid,
        "question": question,
        "framework": framework,
        "model": MODEL,
        "attempt": attempt_no,
        "ok": False,
        "error": None,
        "latency_s": None,
        "llm_requests": None,
        "llm_events": None,
        "tool_calls": [],
        "n_tool_calls": 0,
        "input_tokens": None,
        "output_tokens": None,
        "thoughts_tokens": None,
        "answer": "",
    }
    t0 = time.perf_counter()
    try:
        measured = RUNNERS[framework](question)
    except Exception as e:  # noqa: BLE001 失敗も1行として残す
        row["latency_s"] = round(time.perf_counter() - t0, 2)
        row["error"] = error_kind(e)
        row["answer"] = str(e)[:300]
        return row
    row["latency_s"] = round(time.perf_counter() - t0, 2)
    row.update(measured)
    row["ok"] = True
    row["n_tool_calls"] = len(measured["tool_calls"])
    return row


# ----------------------------------------------------------------------------- 集計
def _fmt_stat(values: list) -> str:
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    return f"{statistics.median(values):g} ({min(values)}–{max(values)})"


def summarize(rows: list[dict]) -> str:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["question_id"], r["framework"])].append(r)
    lines = [
        "| 質問 | フレームワーク | 成功/試行 | LLM 呼び出し 中央値 (最小–最大) | ツール呼び出し 中央値 (最小–最大) | 平均所要時間 [s] |",
        "|---|---|---|---|---|---|",
    ]
    for qid, _ in QUESTIONS:
        for fw in FRAMEWORKS:
            rs = groups.get((qid, fw), [])
            ok = [r for r in rs if r["ok"]]
            lat = f"{statistics.mean(r['latency_s'] for r in ok):.1f}" if ok else "-"
            lines.append(
                f"| {qid} | {fw} | {len(ok)}/{len(rs)} | "
                f"{_fmt_stat([r['llm_requests'] for r in ok])} | "
                f"{_fmt_stat([r['n_tool_calls'] for r in ok])} | {lat} |"
            )
    return "\n".join(lines)


# ----------------------------------------------------------------------------- メイン
def main() -> None:
    parser = argparse.ArgumentParser(description="Pydantic AI 版と ADK 版のエージェントを同じ質問で比べる")
    parser.add_argument("--confirm", action="store_true", help="実際に API を呼ぶ（無いと見積もりだけ表示）")
    parser.add_argument("--reps", type=int, default=2, help="質問ごとの繰り返し回数（既定 2）")
    args = parser.parse_args()

    n_runs = len(QUESTIONS) * len(FRAMEWORKS) * args.reps
    est_requests = (n_runs * MIN_LLM_CALLS_PER_RUN, n_runs * MAX_LLM_CALLS_PER_RUN)
    est_time_s = (n_runs - 1) * INTERVAL_S + n_runs * EST_RUN_S
    print(f"[model] {MODEL}")
    print(f"実行回数: {len(QUESTIONS)} 問 × {len(FRAMEWORKS)} フレームワーク × {args.reps} 回 = {n_runs} 回")
    print(f"LLM リクエスト数の見積もり: {est_requests[0]}〜{est_requests[1]}（1回の実行で {MIN_LLM_CALLS_PER_RUN}〜{MAX_LLM_CALLS_PER_RUN} 回と仮定）")
    print(f"推定所要時間: 約 {est_time_s / 60:.0f} 分（実行の間隔 {INTERVAL_S} 秒 + 1回あたり約 {EST_RUN_S} 秒。再試行があれば延びる）")
    if not args.confirm:
        print("--confirm を付けると実行します。")
        return

    get_api_key()  # 未設定なら手順を表示して終了（キーの値は表示しない）
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"adk_compare_{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    print(f"[out] {out_path}")

    rows: list[dict] = []
    done = 0
    for rep in range(1, args.reps + 1):
        for qid, question in QUESTIONS:
            for fw in FRAMEWORKS:
                if done:
                    time.sleep(INTERVAL_S)
                row = attempt(rep, qid, question, fw, attempt_no=1)
                write_row(out_path, row)
                rows.append(row)
                status = http_status_from_row(row)
                if not row["ok"] and status in RETRY_WAIT_S:
                    wait = RETRY_WAIT_S[status]
                    print(f"  {api_error_message(status)} {wait} 秒待ってやり直します")
                    time.sleep(wait)
                    row = attempt(rep, qid, question, fw, attempt_no=2)
                    write_row(out_path, row)
                    rows.append(row)
                done += 1
                mark = "ok " if row["ok"] else f"NG({row['error']})"
                print(
                    f"[{done:2d}/{n_runs}] rep{rep} {qid} {fw:11s} {mark} "
                    f"llm={row['llm_requests']} tools={[c['name'] for c in row['tool_calls']]} "
                    f"{row['latency_s']}s"
                )

    table = summarize(rows)
    print("\n" + table)
    SUMMARY_PATH.write_text(
        f"# Pydantic AI 版と ADK 版の比較（集計）\n\n"
        f"- 元データ: `{out_path.relative_to(RESULTS_DIR.parent.parent.parent)}`\n"
        f"- モデル: {MODEL}、reps: {args.reps}、実行間隔: {INTERVAL_S} 秒\n"
        f"- 成功/試行 は再試行も1試行として数える。中央値・最小–最大・平均は成功した実行だけで計算\n\n"
        f"{table}\n",
        encoding="utf-8",
    )
    print(f"[summary] {SUMMARY_PATH}")


def http_status_from_row(row: dict) -> int | None:
    err = row.get("error") or ""
    if err.startswith("http_"):
        try:
            return int(err.removeprefix("http_"))
        except ValueError:
            return None
    return None


if __name__ == "__main__":
    main()
