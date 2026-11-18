"""第5回 宿題（基礎）: RAG_TOP_K の変更前後の集計と、再検索の順位を図にする（API は呼ばない）。

使い方:
    uv run --with matplotlib python -m scripts.plot_topk

lectures/05/results/ の topk_summary.csv と replay_queries.csv を読み、figures/ に PNG を2枚書き出す。
    topk_compare.png   左：質問ごとの必須グループの取得率、右：質問ごとの入力トークン数（K=5 と K=10）
    replay_ranks.png   本番の検索語で検索し直したときの、必須グループの最良順位
matplotlib は依存に加えず、--with で一時的に使う。乱数は使わない（重なる点は決まった幅で横にずらす）。
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 画面を開かずに PNG だけ書き出す
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "lectures/05/results"
FIG_DIR = RESULTS / "figures"
DPI = 150
WIDTH = 8.0

QIDS = ["V1", "V2", "V3", "V6", "X2", "L1"]
KS = [5, 10]
# 色覚多様性に配慮して検証済みの2色（全図で K ごとに固定）。色だけに頼らず、ハッチングと点の形も変える
COLORS = {5: "#2a78d6", 10: "#eb6834"}
HATCHES = {5: None, 10: "///"}
MARKERS = {5: "o", 10: "^"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e2e1dc"

TOP_N = 20            # 再検索で順位を取った件数
BEYOND_Y = 22.5       # 21位以下の点をまとめて置く位置
CUTOFFS = {5: "top-5", 10: "top-10"}

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
        "axes.titlesize": 10.5,
        "legend.frameon": False,
        "hatch.linewidth": 1.0,
    }
)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_runs(path: Path) -> dict:
    """topk_summary.csv から、質問×K ごとの成功行を取り出す（複数あれば最後の行）。"""
    runs = {}
    for row in sorted(read_csv(path), key=lambda r: r["timestamp"]):
        if row["row_type"] == "run" and row["ok"] == "True":
            runs[(row["qid"], int(row["top_k"]))] = row
    return runs


def bar(ax, x, height, k, width):
    # ハッチの線は面と同じ色相の上に、背景色で引く（色だけで K を区別しない）
    return ax.bar(x, height, width, color=COLORS[k], hatch=HATCHES[k], edgecolor=SURFACE, linewidth=0.8)


def plot_compare(runs: dict, path: Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(WIDTH, 3.9))
    width = 0.36
    groups = {}
    for k, shift in zip(KS, (-width / 2 - 0.01, width / 2 + 0.01)):
        for i, qid in enumerate(QIDS):
            run = runs.get((qid, k))
            if run is None:
                continue
            x = i + shift
            rate = float(run["required_rate"])
            bar(left, x, rate, k, width)
            groups[qid] = run["required_hit"].split("/")[1]

            tokens = int(float(run["input_tokens"]))
            bar(right, x, tokens, k, width)
            if int(run["n_searches"]) > 1:
                right.text(x, tokens + 150, "×2", ha="center", va="bottom", fontsize=8.5, color=INK)

    left.set_title("Required evidence groups retrieved", loc="left")
    left.set_ylabel("Share of required groups")
    left.set_ylim(0, 1.08)
    left.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    right.set_title("Input tokens per question", loc="left")
    right.set_ylabel("Input tokens")
    right.set_ylim(0, 14000)
    right.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}k"))
    right.text(0.02, 0.97, "×2 = two searches in that run (others: one)", transform=right.transAxes,
               ha="left", va="top", fontsize=8.5, color=INK_2)
    left.set_xticks(range(len(QIDS)), [f"{qid}\nn={groups.get(qid, '?')}" for qid in QIDS])
    left.set_xlabel("Question (n = number of required groups)")
    right.set_xticks(range(len(QIDS)), QIDS)
    right.set_xlabel("Question")
    for ax in (left, right):
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y")
        ax.set_axisbelow(True)

    handles = [Patch(facecolor=COLORS[k], hatch=HATCHES[k], edgecolor=SURFACE, label=f"RAG_TOP_K = {k}") for k in KS]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), handlelength=2.2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_replay(rows: list[dict], path: Path) -> None:
    """点は「検索語 × 必須グループ」。同じ質問・同じK・同じ順位の点は、決まった幅で横に並べる。"""
    base = {5: -0.2, 10: 0.2}   # 質問の目盛りから見た、K ごとの点の位置
    step = 0.075                # 重なる点をずらす幅
    stacks = defaultdict(list)
    for row in sorted(rows, key=lambda r: (r["qid"], int(r["top_k"]), int(r["query_no"]), int(r["group"]))):
        if row["kind"] != "required" or row["qid"] not in QIDS:
            continue
        rank = int(row["best_rank"]) if row["best_rank"] else None
        stacks[(row["qid"], int(row["top_k"]), rank)].append(row)

    fig, ax = plt.subplots(figsize=(WIDTH, 4.8))
    ax.axhspan(TOP_N + 1, BEYOND_Y + 1.5, color=GRID, alpha=0.45, linewidth=0)
    for cutoff, name in CUTOFFS.items():
        ax.axhline(cutoff + 0.5, color=INK_2, linewidth=0.9)
        ax.annotate(name, xy=(1.0, cutoff + 0.5), xycoords=("axes fraction", "data"), xytext=(5, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=9, color=INK)

    for (qid, k, rank), members in stacks.items():
        y = rank if rank is not None else BEYOND_Y
        for i, row in enumerate(members):
            x = QIDS.index(qid) + base[k] + (i - (len(members) - 1) / 2) * step
            ax.scatter(x, y, s=58, marker=MARKERS[k], color=COLORS[k], edgecolor=SURFACE, linewidth=1.0, zorder=3)
            # K を 5 から 10 にすると入る点（6〜10位）だけ、どのページかを書く
            if rank is not None and 5 < rank <= 10:
                page = row["best_page"]
                ax.annotate(f"L{int(page[3:5])} p.{int(page[-3:])}", xy=(x, y), xytext=(0, -7),
                            textcoords="offset points", ha="center", va="top", fontsize=7.5, color=INK_2)

    ax.set_title("Best rank of each required evidence group when the agent's own queries are re-run",
                 loc="left", pad=26)
    ax.set_xticks(range(len(QIDS)), QIDS)
    ax.set_xlim(-0.6, len(QIDS) - 0.4)
    ax.set_xlabel("Question")
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(BEYOND_Y + 1.5, 0.2)   # 1位を上にする
    ax.set_yticks([1, 5, 10, 15, 20, BEYOND_Y], ["1", "5", "10", "15", "20", f">{TOP_N}"])
    ax.set_ylabel("Best rank in the group (1 = closest)")
    for i in range(1, len(QIDS)):
        ax.axvline(i - 0.5, color=GRID, linewidth=0.6, zorder=0)

    handles = [Line2D([], [], linestyle="none", marker=MARKERS[k], markersize=7.5, color=COLORS[k],
                      markeredgecolor=SURFACE, label=f"Queries from the K = {k} run") for k in KS]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.2,
              handletextpad=0.3, columnspacing=1.6)
    ax.text(-0.5, 13.0, "One point = one search query × one required group\n"
            "(rank of the group's best-ranked page, top 20 searched)",
            ha="left", va="top", fontsize=8.5, color=INK_2, linespacing=1.4)
    fig.tight_layout()
    fig.subplots_adjust(right=0.91)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    compare, replay = FIG_DIR / "topk_compare.png", FIG_DIR / "replay_ranks.png"
    plot_compare(load_runs(RESULTS / "topk_summary.csv"), compare)
    plot_replay(read_csv(RESULTS / "replay_queries.csv"), replay)
    for path in (compare, replay):
        print(f"[出力] {path}")


if __name__ == "__main__":
    main()
