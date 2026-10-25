"""第1回 宿題（発展）: 計測結果の CSV を図にする（API は呼ばない）。

使い方:
    uv run --with matplotlib python -m scripts.plot_latency

lectures/01/results/ の CSV を読み、lectures/01/results/figures/ に PNG を4枚書き出す。
matplotlib は依存に加えず、--with で一時的に使う。
R² は scripts.measure_latency と同じ関数で計算するので、summary.md と同じ値になる。
"""

import csv
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 画面を開かずに PNG だけ書き出す
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

from scripts.measure_latency import RESULTS_DIR, _num, fit_line  # noqa: E402

GRID_CSV = RESULTS_DIR / "raw_grid_20261025-145314.csv"
BURST_CSVS = [RESULTS_DIR / "raw_burst_20261025-174655.csv", RESULTS_DIR / "raw_burst_20261025-175341.csv"]
FIG_DIR = RESULTS_DIR / "figures"
DPI = 150

MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash"]
# 色覚多様性に配慮して検証済みの、カテゴリ配色の先頭3色（全図でモデルごとに固定）
COLORS = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a"]))
RATE_LIMITED = "#d03b3b"  # 429 用のステータス色（ハッチングと併用して色だけに頼らない）
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e2e1dc"
LOG_TICKS = [2, 5, 10, 20, 50, 100, 200]

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


def load(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _ok(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["status"] == "ok"]


def _log_y(ax) -> None:
    """y を対数軸にし、目盛りを 2, 5, 10, ... の普通の数字で表示する。"""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator(LOG_TICKS))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(2, 250)


def _tint(hex_color: str, amount: float) -> str:
    """色を背景色に寄せて薄くする（amount=0 で元の色、1 で背景色）。"""
    rgb = [int(hex_color[i : i + 2], 16) for i in (1, 3, 5)]
    bg = [int(SURFACE[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(c + (b - c) * amount):02x}" for c, b in zip(rgb, bg))


def fig1_latency_by_model(rows: list[dict]) -> plt.Figure:
    ok = _ok(rows)
    qids = ["Q1", "Q2", "Q3"]
    offsets = {1: -0.12, 2: 0.0, 3: 0.12}  # rep ごとに少し横にずらす（乱数は使わない）
    fig, ax = plt.subplots(figsize=(8, 5))
    ticks = []
    for mi, model in enumerate(MODELS):
        color = COLORS[model]
        for qi, q in enumerate(qids):
            x = mi * 4 + qi
            ticks.append(x)
            g = [r for r in ok if r["model"] == model and r["question_id"] == q]
            for r in g:
                retry = r["error"].startswith("retry")
                ax.scatter(
                    x + offsets.get(int(r["rep"]), 0.0), _num(r["latency_s"]), s=42,
                    facecolor=SURFACE if retry else color, edgecolor=color, linewidth=1.6, zorder=3,
                )  # fmt: skip
            if g:
                median = statistics.median(_num(r["latency_s"]) for r in g)
                ax.hlines(median, x - 0.28, x + 0.28, color=INK, linewidth=2, zorder=4)
        ax.text(mi * 4 + 1, -0.1, model, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=10)

    ax.set_xticks(ticks, qids * len(MODELS))
    ax.set_xlim(-0.6, ticks[-1] + 0.6)
    _log_y(ax)
    ax.grid(axis="y")
    ax.set_ylabel("Latency (s, log scale)")
    ax.set_title("Latency per trial by model and question (status = ok)", loc="left")
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor=INK_2, markeredgecolor=INK_2, label="ok trial"),
        Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor=SURFACE, markeredgecolor=INK_2,
               markeredgewidth=1.6, label="ok after retry (first attempt: 503)"),
        Line2D([], [], color=INK, linewidth=2, label="median"),
    ]  # fmt: skip
    ax.legend(handles=handles, loc="upper left", ncol=3)
    fig.subplots_adjust(bottom=0.17)
    return fig


def fig2_latency_vs_tokens(rows: list[dict]) -> plt.Figure:
    ok = [r for r in _ok(rows) if _num(r["output_tokens"]) is not None]
    fig, ax = plt.subplots(figsize=(8, 5))
    xmax = 0.0
    for model in MODELS:
        g = [r for r in ok if r["model"] == model]
        if not g:
            continue
        color = COLORS[model]
        xs = [_num(r["output_tokens"]) for r in g]
        ys = [_num(r["latency_s"]) for r in g]
        has_thoughts = any(_num(r["thoughts_tokens"]) for r in g)
        # 思考トークンのないモデル（lite）の点を、3.8-flash の点と線より手前に描く
        ax.scatter(
            xs, ys, s=42, color=color, edgecolor=SURFACE, linewidth=1, zorder=4 if has_thoughts else 6,
            label=f"{model}: output ({_r2_label(xs, ys)})",
        )  # fmt: skip
        xmax = max(xmax, *xs)
        if model == "gemini-3.1-flash-lite":
            y_out = max(ys)  # 150.9 秒の外れ値
            ax.annotate(
                "3.1-flash-lite outlier", xy=(xs[ys.index(y_out)], y_out), xytext=(-24, 16),
                textcoords="offset points", ha="right", va="center", fontsize=8, color=INK_2,
                arrowprops={"arrowstyle": "-", "color": INK_2, "linewidth": 0.8}, zorder=7,
            )  # fmt: skip
        if has_thoughts:
            xt = [x + (_num(r["thoughts_tokens"]) or 0) for x, r in zip(xs, g)]
            for x0, x1, y in zip(xs, xt, ys):
                ax.plot([x0, x1], [y, y], color=color, linewidth=1, alpha=0.5, zorder=2)
            ax.scatter(
                xt, ys, s=42, marker="s", facecolor=SURFACE, edgecolor=color, linewidth=1.6, zorder=3,
                label=f"{model}: output + thoughts ({_r2_label(xt, ys)})",
            )  # fmt: skip
            xmax = max(xmax, *xt)

    ax.set_xlim(0, xmax * 1.06)
    _log_y(ax)
    ax.grid(True)
    ax.set_xlabel("Tokens per request")
    ax.set_ylabel("Latency (s, log scale)")
    ax.set_title("Latency vs. token count (status = ok)", loc="left")
    ax.legend(loc="lower right", fontsize=9)
    return fig


def _r2_label(xs: list[float], ys: list[float]) -> str:
    fit = fit_line(xs, ys)
    return "R² = –" if fit is None or fit[2] is None else f"R² = {fit[2]:.3f}"


def fig3_token_breakdown(rows: list[dict]) -> plt.Figure:
    ok = _ok(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.5
    top = 0.0
    for i, model in enumerate(MODELS):
        g = [r for r in ok if r["model"] == model]
        color = COLORS[model]
        output = statistics.mean(_num(r["output_tokens"]) for r in g)
        thoughts = statistics.mean(_num(r["thoughts_tokens"]) or 0 for r in g)  # 空欄は思考なし（0）
        # 積み上げの境目に背景色の細いすき間を入れる
        ax.bar(i, output, width=width, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(i + width / 2 + 0.04, output / 2, f"output {output:.0f}", va="center", fontsize=9)
        if thoughts:
            ax.bar(
                i, thoughts, bottom=output, width=width, facecolor=_tint(color, 0.6), edgecolor=color,
                hatch="//", linewidth=0, zorder=3,
            )  # fmt: skip
            ax.text(i + width / 2 + 0.04, output + thoughts / 2, f"thoughts {thoughts:.0f}", va="center", fontsize=9)
            note = f"total {output + thoughts:.0f}"
        else:
            note = "no thoughts reported"
        ax.text(i, output + thoughts + 40, note, ha="center", va="bottom", fontsize=9, color=INK_2)
        top = max(top, output + thoughts)

    ax.set_xticks(range(len(MODELS)), MODELS)
    ax.set_xlim(-0.5, len(MODELS) - 0.2)
    ax.set_ylim(0, top * 1.12)
    ax.grid(axis="y", zorder=0)
    ax.set_ylabel("Mean tokens per request")
    ax.set_title("Mean output and thoughts tokens per request (status = ok, Q1–Q3 combined)", loc="left")
    handles = [
        Patch(facecolor=INK_2, label="output (candidates)"),
        Patch(facecolor=_tint(INK_2, 0.6), edgecolor=INK_2, hatch="//", linewidth=0, label="thoughts"),
    ]
    ax.legend(handles=handles, loc="upper left")
    return fig


def fig4_burst_timeline(paths: list[Path]) -> plt.Figure:
    runs = [load(p) for p in paths]
    fig, axes = plt.subplots(len(runs), 1, figsize=(8, 4 * len(runs)), sharex=True)
    xmax = 0.0
    for i, (ax, path, rows) in enumerate(zip(axes, paths, runs), 1):
        first_sent = min(_num(r["elapsed_from_start_s"]) for r in rows)
        for r in rows:
            sent = _num(r["elapsed_from_start_s"]) - first_sent
            latency = _num(r["latency_s"])
            limited = r["status"] == "429"
            color = COLORS["gemini-3.5-flash-lite"] if r["status"] == "ok" else RATE_LIMITED if limited else INK_2
            ax.barh(
                int(r["rep"]), latency, left=sent, height=0.7, color=color, edgecolor=SURFACE,
                hatch="///" if limited else None, linewidth=0.5, zorder=3,
            )  # fmt: skip
            xmax = max(xmax, sent + latency)
        c = Counter(r["status"] for r in rows)
        started = datetime.strptime(path.stem.rsplit("_", 1)[-1], "%Y%m%d-%H%M%S")
        ax.set_title(f"Run {i} (started {started:%H:%M:%S}): ok {c['ok']} / 429 {c['429']}", loc="left")
        ax.set_yticks(range(1, 21))
        ax.tick_params(axis="y", labelsize=8)
        ax.set_ylim(20.6, 0.4)  # 1 を上にする
        ax.set_ylabel("Request #")
        ax.grid(axis="x", zorder=0)

    axes[0].set_xlim(0, xmax * 1.3)  # 右側に凡例の余白をとる
    axes[-1].set_xlabel("Seconds since first request sent")
    handles = [
        Patch(facecolor=COLORS["gemini-3.5-flash-lite"], label="ok"),
        Patch(facecolor=RATE_LIMITED, edgecolor=SURFACE, hatch="///", label="429 (rate limited)"),
    ]
    axes[0].legend(handles=handles, loc="center right")
    fig.suptitle("Burst: 20 concurrent requests to gemini-3.5-flash-lite (Q1)", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    return fig


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    grid = load(GRID_CSV)
    figures = {
        "fig1_latency_by_model.png": fig1_latency_by_model(grid),
        "fig2_latency_vs_tokens.png": fig2_latency_vs_tokens(grid),
        "fig3_token_breakdown.png": fig3_token_breakdown(grid),
        "fig4_burst_timeline.png": fig4_burst_timeline(BURST_CSVS),
    }
    for name, fig in figures.items():
        path = FIG_DIR / name
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        print(f"[保存] {path}")


if __name__ == "__main__":
    main()
