"""Рисунки к статье по итогам основного эксперимента.

1. fig_fields.png  — рис. 1: примеры полей задач: равномерное поле, средняя и сильная скученность
                     (M = 8, ρ = 6, один повтор), база в углу.
2. fig_front.png   — рис. 2: точный фронт «суммарный путь — время выполнения» на одном примере и
                     положение методов; оси — в долях минимумов S* и Λ*.
3. fig_levels.png  — рис. 3: медиана разрыва каждого метода по уровням скученности,
                     столбцы — три предпочтения, строки — режим старта (основной эксперимент, все M, ρ, типы).
4. fig_levels_types.png — то же отдельно для однородной и разнотипной группы (в статью не вошёл).
5. fig_bestmap.png — рис. 4: карта лучших методов: в каждой ячейке «аппаратов × задач на аппарат × уровень»
                     метод с наименьшей медианой разрыва; панели — предпочтение × старт × типы.

Рисунки рассчитаны на печать в оттенках серого: у методов разные маркеры и типы линий.

Запуск:  python -m experiments.figures_main [--src data/all.csv.gz] [--front-json data/front_example.json]
         [--out results/fig]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from allocsim.clustering import clustering_index
from allocsim.fields import sample_field
from allocsim.instance import main_field

SURFACE, INK, INK2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#dddcd8"
METHODS = ["greedy", "cbba", "gaca", "kmeans_hungarian", "skfu"]
NAMES = {"greedy": "жадный", "cbba": "CBBA", "gaca": "GACA", "kmeans_hungarian": "k-means + ВМ",
         "skfu": "секторы"}
SHORT = {"greedy": "Жд", "cbba": "CB", "gaca": "GA", "kmeans_hungarian": "KM", "skfu": "Ск"}
COLORS = {"greedy": "#8a8986", "cbba": "#2a78d6", "gaca": "#eb6834", "kmeans_hungarian": "#1baf7a",
          "skfu": "#9b59b6"}
MARKERS = {"greedy": "x", "cbba": "o", "gaca": "s", "kmeans_hungarian": "^", "skfu": "D"}
LINES = {"greedy": ":", "cbba": "-", "gaca": "--", "kmeans_hungarian": "-.", "skfu": (0, (1, 1))}
PREFS = {"g_S": "важен путь (g_S)", "g_Lam": "важно время (g_Λ)", "g_w50": "поровну (g_0,5)"}
STARTS = {"base": "общая база в углу", "scattered": "рассредоточенный старт"}
LEVELS = ["L0", "L1", "L2", "L3", "L4"]
FLOOR = 0.005   # в долях; на рисунке разрывы показаны в процентах
PCT = 100.0
LEVEL_TICKS = ["L0\nравном.", "L1", "L2", "L3", "L4\nскопл."]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 9, "axes.titlesize": 9.5, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
})


def fig_fields(out, rep=3):
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.95))
    for ax, lvl, title in zip(axes, ("L0", "L2", "L4"), ("равномерное поле", "средняя скученность",
                                                         "сильная скученность")):
        pts, _ = sample_field(main_field(2, 1000.0, 8, 6, lvl), rep)
        ax.scatter(pts[:, 0], pts[:, 1], s=10, color=INK, linewidths=0)
        ax.scatter([0], [0], s=60, marker="s", color=COLORS["gaca"], clip_on=False, zorder=3)
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(f"{lvl}: {title}\nR = {clustering_index(pts, 1000.0):.2f}")
    fig.tight_layout()
    fig.savefig(out / "fig_fields.png", dpi=200)
    plt.close(fig)


def fig_front(out, path):
    rec = json.loads(Path(path).read_text())
    ref = rec["reference"]
    front = sorted(map(tuple, ref.get("ref_front") or ref["front"]))
    S0, L0 = min(s for s, _ in front), min(l for _, l in front)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    fs = np.array([s / S0 for s, _ in front])
    fl = np.array([l / L0 for _, l in front])
    ax.step(fs, fl, where="post", color=INK2, lw=1.0)
    ax.scatter(fs, fl, s=22, color=INK, zorder=3, label="точный фронт")
    for m in METHODS:
        v = rec["methods"][m]
        ax.scatter(v["S"] / S0, v["Lam"] / L0, s=55, marker=MARKERS[m], color=COLORS[m], zorder=4,
                   label=NAMES[m], linewidths=1.4 if m == "greedy" else 0)
    ax.set_xlabel("суммарный путь S / S*")
    ax.set_ylabel("время выполнения Λ / Λ*")
    ax.legend(fontsize=8, loc="upper right")
    typ = "разнотипная" if rec["hetero"] else "однородная"
    st = "общая база" if rec["start"] == "base" else "рассредоточенный старт"
    ax.set_title(f"M = {rec['M']}, N = {rec['N']}, {rec['level']}, {st}, {typ} группа", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "fig_front.png", dpi=200)
    plt.close(fig)


def _median_by_level(d, pref):
    return d.groupby(["level", "method"])[pref].median().unstack()


def _levels_grid(d, rows, out_name, row_title):
    fig, axes = plt.subplots(len(rows), 3, figsize=(9.0, 2.7 * len(rows)), squeeze=False)
    for i, (rlabel, dd) in enumerate(rows):
        for j, (pref, ptitle) in enumerate(PREFS.items()):
            ax = axes[i, j]
            med = _median_by_level(dd, pref)
            for m in METHODS:
                if m in med:
                    ax.plot(range(5), PCT * np.maximum(med.loc[LEVELS, m].to_numpy(), FLOOR), marker=MARKERS[m],
                            ls=LINES[m], color=COLORS[m], label=NAMES[m], ms=4.5, lw=1.3)
            ax.set_yscale("log")
            ax.set_ylim(PCT * FLOOR * 0.8, PCT * 4)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
            ax.set_xticks(range(5))
            ax.set_xticklabels(LEVEL_TICKS, fontsize=7.5)
            if i == 0:
                ax.set_title(ptitle)
            if j == 0:
                ax.set_ylabel(f"{rlabel}\nмедиана разрыва, %")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, fontsize=8)
    fig.text(0.01, 0.005, "Логарифмическая шкала; медианы меньше 0,5 % показаны на уровне 0,5 %.",
             fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1 - 0.09 / len(rows)))
    fig.savefig(out_name, dpi=200)
    plt.close(fig)


def fig_levels(out, d):
    m = d[d.prefix == "main"]
    _levels_grid(m, [(STARTS[s], m[m.start == s]) for s in ("base", "scattered")], out / "fig_levels.png",
                 "")
    rows = []
    for s in ("base", "scattered"):
        for het in (False, True):
            rows.append((f"{'база' if s == 'base' else 'рассредоточ.'}, {'разнотип.' if het else 'однотип.'}",
                         m[(m.start == s) & (m.hetero == het)]))
    _levels_grid(m, rows, out / "fig_levels_types.png", "")


def fig_bestmap(out, d):
    m = d[(d.prefix == "main") & d.method.isin(METHODS)]
    combos = [(M, r) for M in (4, 8, 16) for r in (3, 6, 12)]
    panels = [(p, s, h) for p in PREFS for s in ("base", "scattered") for h in (False, True)]
    fig, axes = plt.subplots(3, 4, figsize=(10.0, 8.6))
    for ax, (pref, s, het) in zip(axes.ravel(), panels):
        dd = m[(m.start == s) & (m.hetero == het)]
        med = dd.groupby(["M", "rho", "level", "method"])[pref].median().unstack()
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(len(combos) - 0.5, -0.5)
        for yi, (M, r) in enumerate(combos):
            for xi, lvl in enumerate(LEVELS):
                if (M, r, lvl) not in med.index:
                    continue
                row = med.loc[(M, r, lvl)].dropna().sort_values()
                best, val = row.index[0], row.iloc[0]
                second = row.iloc[1] if len(row) > 1 else np.nan
                ax.add_patch(plt.Rectangle((xi - 0.5, yi - 0.5), 1, 1, color=COLORS[best],
                                           alpha=0.85 if second - val > 0.02 else 0.35, lw=0))
                ax.text(xi, yi, SHORT[best], ha="center", va="center", fontsize=7, color=INK)
        ax.set_xticks(range(5))
        ax.set_xticklabels(LEVELS, fontsize=7.5)
        ax.set_yticks(range(len(combos)))
        ax.set_yticklabels([f"M={M}, ρ={r}" for M, r in combos], fontsize=7)
        ax.grid(False)
        ax.set_title(f"{PREFS[pref].split(' (')[0]}; {'база' if s == 'base' else 'рассред.'}; "
                     f"{'разнотип.' if het else 'однотип.'}", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in METHODS]
    fig.legend(handles, [f"{SHORT[k]} — {NAMES[k]}" for k in METHODS], loc="lower center", ncol=5,
               fontsize=8)
    fig.text(0.5, 0.035, "Яркая заливка — лучший метод опережает второй больше чем на 2 п. п. по медиане разрыва; "
                         "бледная — почти поровну", ha="center", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "fig_bestmap.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/all.csv.gz")
    ap.add_argument("--front-json", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path("results/fig")
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(args.src)
    fig_fields(out)
    if args.front_json:
        fig_front(out, args.front_json)
    fig_levels(out, d)
    fig_bestmap(out, d)
    print("→", out)


if __name__ == "__main__":
    main()
