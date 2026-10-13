import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DATA = (
    ROOT
    / "cssd-fixed-multiseed-server32-w72-final"
    / "fixed_30_20_server32_w72_report_bundle"
    / "outputs"
    / "fixed_30_20_server32_w72"
    / "report"
    / "tables4_7_paper.csv"
)
OUT_DIR = Path(__file__).resolve().parent

with DATA.open("r", encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
values = {(row["规模"], row["算法"]): float(row["RPR"]) for row in rows}

scales = ["J100", "J250", "J500", "J1000"]
algorithms = ["ALNS", "TS", "Kraul-GA", "IBS-CABC"]
colors = {
    "ALNS": "#2C6EAA",
    "TS": "#2E8B57",
    "Kraul-GA": "#7B61A8",
    "IBS-CABC": "#C93232",
}
markers = {"ALNS": "^", "TS": "D", "Kraul-GA": "P", "IBS-CABC": "s"}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    }
)

fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.4), sharex=True, sharey=True)
fig.subplots_adjust(left=0.14, right=0.975, top=0.84, bottom=0.12, wspace=0.16, hspace=0.30)
fig.suptitle("Scale-wise relative performance ratio", fontsize=17, y=0.965)
fig.text(
    0.5,
    0.915,
    "Scale-level mean objective divided by the minimum among the four stochastic algorithms; lower is better",
    ha="center",
    va="center",
    fontsize=10.5,
    color="#4D5660",
)

y_positions = list(range(len(algorithms)))
for ax, scale in zip(axes.flat, scales):
    ax.axvline(1.0, color="#303840", linewidth=1.2, linestyle="--", zorder=1)
    for y, algorithm in zip(y_positions, algorithms):
        value = values[(scale, algorithm)]
        ax.hlines(y, 1.0, value, color=colors[algorithm], alpha=0.45, linewidth=2.4, zorder=2)
        ax.scatter(
            value,
            y,
            s=72,
            marker=markers[algorithm],
            facecolor=colors[algorithm],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        offset = 0.0017 if value > 1.0002 else 0.0015
        ax.text(value + offset, y, f"{value:.4f}", va="center", fontsize=9.2, color="#27313A")

    repeats = "30 seeds" if scale in {"J100", "J250"} else "20 seeds"
    ax.set_title(f"{scale}  |  20 instances × {repeats}", loc="left", fontweight="semibold")
    ax.set_xlim(0.997, 1.108)
    ax.set_ylim(-0.55, 3.55)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#D9DEE3", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#7B858E")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors="#4D5660")

for ax in axes[:, 0]:
    ax.set_yticks(y_positions, algorithms)
for ax in axes[:, 1]:
    ax.tick_params(labelleft=False)
for ax in axes[1, :]:
    ax.set_xlabel("Relative performance ratio (RPR)")

fig.text(
    0.5,
    0.035,
    "Each instance is represented by its across-seed mean; the plotted ratio is formed after aggregation at each scale.",
    ha="center",
    va="center",
    fontsize=9.3,
    color="#59636D",
)

for suffix in ("png", "pdf", "svg"):
    path = OUT_DIR / f"figure5_rpr_multiseed.{suffix}"
    fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight", facecolor="white")

plt.close(fig)
print(OUT_DIR / "figure5_rpr_multiseed.png")
