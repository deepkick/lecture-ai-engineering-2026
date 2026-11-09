"""第3回 宿題（発展 a）: compare_adk.py の結果（jsonl）から、Pydantic AI 版と ADK 版の比較図を描く。

使い方（リポジトリ直下で）:
    uv run --with matplotlib python -m scripts.plot_adk_compare                      # 最新の jsonl を使う
    uv run --with matplotlib python -m scripts.plot_adk_compare lectures/03/results/adk_compare_XXXX.jsonl

出力（lectures/03/results/figures/、150 dpi、幅 8 インチ）:
    adk_compare_calls.png   質問 × フレームワークごとの LLM 呼び出し回数とツール呼び出し回数（rep の平均、点は各 rep）
    adk_compare_tokens.png  質問 × フレームワークごとの入力・出力トークンと所要時間（rep の平均、点は各 rep）

作法: ラベルは英語。系列の色は全図で同じ（Pydantic AI = 青、ADK = 橙。色覚多様性に配慮してハッチングと
直接ラベルも併用）。乱数は使わない。matplotlib は講座の依存に無いので --with で一時的に足す。
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "lectures" / "03" / "results"
FIG_DIR = RESULTS_DIR / "figures"

FRAMEWORKS = ("pydantic_ai", "adk")
LABELS = {"pydantic_ai": "Pydantic AI", "adk": "ADK"}
# 全図で同じ色・同じハッチング（色だけに頼らない）
COLORS = {"pydantic_ai": "#2a78d6", "adk": "#eb6834"}
HATCHES = {"pydantic_ai": "", "adk": "//"}
QUESTION_LABELS = {
    "Q1": "Q1\n3 levels",
    "Q2": "Q2\nLoRA/QLoRA",
    "Q3": "Q3\nquiz",
    "Q4": "Q4\nlecture 3",
    "Q5": "Q5\nVLA",
}
BAR_W = 0.38


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r["ok"]]


def group(rows: list[dict], key: str) -> dict[tuple[str, str], list[float]]:
    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if r.get(key) is not None:
            out[(r["question_id"], r["framework"])].append(r[key])
    return out


def grouped_bars(ax, data: dict, questions: list[str], ylabel: str, fmt: str = "{:g}") -> None:
    """質問ごとに2本（フレームワーク）の棒。高さは rep の平均、各 rep の値を点で重ね、値を棒の上に書く。"""
    for i, fw in enumerate(FRAMEWORKS):
        xs = [q_i + (i - 0.5) * BAR_W for q_i in range(len(questions))]
        means = [statistics.mean(data.get((q, fw), [0])) for q in questions]
        ax.bar(
            xs, means, BAR_W * 0.92, label=LABELS[fw], color=COLORS[fw], hatch=HATCHES[fw],
            edgecolor="white", linewidth=0.8, zorder=2,
        )
        for x, q, m in zip(xs, questions, means):
            values = data.get((q, fw), [])
            for v in values:
                ax.plot(x, v, marker="o", markersize=3.5, color="#333333", zorder=3)
            # 値ラベルは点と重ならないよう、各 rep の最大値の上に置く
            top = max(values) if values else m
            ax.annotate(fmt.format(m), (x, top), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, color="#333333")
    ax.set_xticks(range(len(questions)))
    ax.set_xticklabels([QUESTION_LABELS.get(q, q) for q in questions], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_ylim(0, max(max(v) for v in data.values()) * 1.3)


def main() -> None:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        candidates = sorted(RESULTS_DIR.glob("adk_compare_*.jsonl"))
        if not candidates:
            raise SystemExit("adk_compare_*.jsonl が見つかりません。先に scripts.compare_adk --confirm を実行してください。")
        path = candidates[-1]
    rows = load_rows(path)
    questions = sorted({r["question_id"] for r in rows})
    reps = sorted({r["rep"] for r in rows})
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    subtitle = f"{rows[0]['model']}, {len(reps)} reps per question (bars: mean, dots: each rep)"

    # 図1: LLM 呼び出し回数とツール呼び出し回数
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    grouped_bars(axes[0], group(rows, "llm_requests"), questions, "LLM requests per run")
    axes[0].set_title("LLM requests", fontsize=10)
    grouped_bars(axes[1], group(rows, "n_tool_calls"), questions, "Tool calls per run")
    axes[1].set_title("Tool calls", fontsize=10)
    fig.suptitle(f"Pydantic AI vs ADK: same instructions and tools\n{subtitle}", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.86))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    out1 = FIG_DIR / "adk_compare_calls.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)

    # 図2: トークンと所要時間
    fig, axes = plt.subplots(1, 3, figsize=(8, 3.8))
    grouped_bars(axes[0], group(rows, "input_tokens"), questions, "Input tokens per run (sum of requests)", fmt="{:.0f}")
    axes[0].set_title("Input tokens", fontsize=10)
    grouped_bars(axes[1], group(rows, "output_tokens"), questions, "Output tokens per run (incl. thoughts)", fmt="{:.0f}")
    axes[1].set_title("Output tokens", fontsize=10)
    grouped_bars(axes[2], group(rows, "latency_s"), questions, "Wall-clock time per run [s]", fmt="{:.1f}")
    axes[2].set_title("Latency", fontsize=10)
    for ax in axes:
        ax.set_xticklabels([q for q in questions], fontsize=8)  # 3枚並ぶので短いラベル
    fig.suptitle(f"Pydantic AI vs ADK: tokens and latency\n{subtitle}", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.86))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    out2 = FIG_DIR / "adk_compare_tokens.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)

    print(f"[input] {path}  ({len(rows)} successful runs)")
    print(f"[saved] {out1}")
    print(f"[saved] {out2}")


if __name__ == "__main__":
    main()
