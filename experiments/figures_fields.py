"""Рисунки к калибровке уровней скученности.

1. fields_levels.png — примеры полей: строки — уровни L0…L4, столбцы — M = 4 и M = 16
   при двух способах задавать σ. Один и тот же номер повтора на всех уровнях
   (общие случайные числа), в углу панели — показатель R.
2. R_by_level.png — медиана и межквартильный размах R по уровням для M = 4, 8, 16 (ρ = 6),
   слева σ как доля стороны поля, справа — как доля расстояния между центрами.

Запуск из корня репозитория (после calibrate_levels):  python -m experiments.figures_fields
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from allocsim.clustering import clustering_index
from allocsim.fields import LEVELS, sample_field
from allocsim.instance import main_field

OUT = Path("results/calibration")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {4: "#2a78d6", 8: "#eb6834", 16: "#1baf7a"}   # проверено validate_palette.js
MARKERS = {4: "o", 8: "s", 16: "^"}
LEVEL_NOTE = {"L0": "равномерно", "L1": "L1", "L2": "L2", "L3": "L3", "L4": "L4"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 9, "axes.titlesize": 9.5, "axes.titlecolor": INK,
})


def fig_fields(rep=3):
    cols = [(4, 6, "abs", "M = 4, N = 24\n(оба способа совпадают)"),
            (16, 6, "abs", "M = 16, N = 96\nσ от стороны поля"),
            (16, 6, "rel", "M = 16, N = 96\nσ от расстояния между центрами")]
    fig, axes = plt.subplots(len(LEVELS), len(cols), figsize=(8.2, 11.6))
    for i, lvl in enumerate(LEVELS):
        for j, (m, rho, mode, title) in enumerate(cols):
            ax = axes[i, j]
            f = main_field(2, 1000.0, m, rho, lvl, sigma_mode=mode)
            pts, _ = sample_field(f, rep)
            ax.scatter(pts[:, 0], pts[:, 1], s=9, color=SERIES[4], linewidths=0)
            ax.set_xlim(0, 1000)
            ax.set_ylim(0, 1000)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.03, 0.97, f"R = {clustering_index(pts, 1000.0):.2f}", transform=ax.transAxes,
                    va="top", ha="left", fontsize=8.5, color=INK,
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.85))
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(LEVEL_NOTE[lvl], color=INK, fontsize=9.5)
    fig.suptitle("Поля задач по уровням скученности (один повтор, общие случайные числа)",
                 fontsize=10.5, color=INK, y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "fields_levels.png", dpi=160)
    plt.close(fig)


def fig_r_by_level():
    df = pd.read_csv(OUT / "R_values.csv")
    df = df[(df.dim == 2) & (df.rho == 6) & (df.centers_per_agent == 1.0)]
    levels = list(LEVELS)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), sharey=True)
    titles = {"abs": "σ — доля стороны поля (первоначальный вариант)",
              "rel": "σ — доля расстояния между центрами (предложение)"}
    for ax, mode in zip(axes, ("abs", "rel")):
        for m in (4, 8, 16):
            g = df[(df["mode"] == mode) & (df.M == m)].groupby("level")["R"]
            med = g.median().reindex(levels)
            q1, q3 = g.quantile(0.25).reindex(levels), g.quantile(0.75).reindex(levels)
            x = range(len(levels))
            ax.fill_between(x, q1, q3, color=SERIES[m], alpha=0.12, linewidth=0)
            ax.plot(x, med, color=SERIES[m], lw=2, marker=MARKERS[m], ms=5.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=f"M = {m}")
            if mode == "abs":
                ax.text(len(levels) - 1 + 0.12, med.iloc[-1], f"M = {m}", color=INK2,
                        va="center", fontsize=8.5)
        if mode == "rel":
            ax.text(len(levels) - 1, 1.12, "линии M = 4, 8, 16\nпочти совпадают", color=INK2,
                    ha="right", va="top", fontsize=8.5)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(["равн.", "L1", "L2", "L3", "L4"])
        ax.set_xlim(-0.3, len(levels) - 0.3)
        ax.set_ylim(0, 1.25)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(titles[mode])
        ax.set_xlabel("уровень скученности")
    axes[0].set_ylabel("показатель скученности R\n(медиана и межквартильный размах)")
    axes[0].legend(frameon=False, loc="lower left", fontsize=8.5)
    fig.suptitle("Один и тот же уровень при разном числе аппаратов (ρ = 6, 200 полей на точку)",
                 fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "R_by_level.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_fields()
    fig_r_by_level()
    print("рисунки →", OUT)


if __name__ == "__main__":
    main()
