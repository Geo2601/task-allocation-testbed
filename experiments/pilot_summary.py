"""Сводка пилота (план эксперимента): сколько примеров на ячейку нужно и сколько займёт эксперимент.

1. Число примеров. Для каждой ячейки и каждого из трёх предпочтений (g_S, g_Lam, g_w50) берутся
   два лучших по медиане метода; по парным разностям их разрывов оценивается разброс sd, и
   n = ((z_{1−α/2} + z_{1−β}) · sd / δ)² — сколько примеров нужно, чтобы различить разницу δ
   при мощности 0,8 и α = 0,005 (поправка Холма на 10 пар методов внутри ячейки, худший случай).
   Разброс по 3 примерам на ячейку шумный, поэтому n даётся и по медиане sd по ячейкам, и по
   «объединённому» sd (корень из среднего квадрата). Это грубая оценка для парного t-критерия;
   итоговая проверка — Уилкоксон с поправкой Холма.
2. Время. Медиана времени эталона и методов на пример по N и виду эталона → полное время
   эксперимента при заданном числе примеров и ядер.

Запуск:  python -m experiments.pilot_summary --src results/main/all.csv --reps 100 --cores 6
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm

PREFS = ("g_S", "g_Lam", "g_w50")
MAIN_METHODS = ("greedy", "cbba", "kmeans_hungarian", "gaca", "skfu")


def required_n(d, delta, alpha, power, methods=MAIN_METHODS):
    cells = ["prefix", "level", "M", "rho", "start", "hetero"]
    d = d[d.method.isin(methods)]
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    rows = []
    for key, g in d.groupby(cells):
        for pref in PREFS:
            piv = g.pivot_table(index="rep", columns="method", values=pref)
            if piv.shape[0] < 3 or piv.shape[1] < 2:
                continue
            best2 = piv.median().nsmallest(2).index
            diff = piv[best2[0]] - piv[best2[1]]
            sd = float(diff.std(ddof=1))
            rows.append(dict(zip(cells, key), pref=pref, best=best2[0], second=best2[1],
                             median_diff=float(diff.median()), sd=sd, reps=len(diff),
                             n_needed=int(np.ceil((z * sd / delta) ** 2)) if sd > 0 else 1))
    return pd.DataFrame(rows)


def timing(d):
    inst = d.drop_duplicates("key")
    meth = d.groupby(["key", "N"]).wall.sum().reset_index().groupby("N").wall.median()
    ref = inst.groupby(["N", "ref_kind"]).ref_wall.median().unstack()
    return ref, meth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results/main/all.csv")
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.005)
    ap.add_argument("--power", type=float, default=0.8)
    ap.add_argument("--reps", type=int, default=100, help="примеров на ячейку для расчёта времени")
    ap.add_argument("--cores", type=int, default=6)
    args = ap.parse_args()
    d = pd.read_csv(args.src)
    print(f"Примеров: {d.key.nunique()}, ячеек: {d.groupby(['prefix', 'level', 'M', 'rho', 'start', 'hetero']).ngroups}")
    print("\nМедианы разрывов по методам (все ячейки):")
    print(d[d.method.isin(MAIN_METHODS)].groupby("method")[list(PREFS) + ["idle"]].median().round(3).to_string())
    rn = required_n(d, args.delta, args.alpha, args.power)
    if len(rn):
        z = norm.ppf(1 - args.alpha / 2) + norm.ppf(args.power)
        sd_med = rn.groupby("pref").sd.median()
        sd_pool = rn.groupby("pref").sd.apply(lambda x: float(np.sqrt((x ** 2).mean())))
        rows = []
        for delta in (0.03, 0.05, 0.10):
            for pref in sd_med.index:
                rows.append(dict(разница=f"{delta:.0%}", показатель=pref,
                                 по_медиане_sd=int(np.ceil((z * sd_med[pref] / delta) ** 2)),
                                 по_объединённому_sd=int(np.ceil((z * sd_pool[pref] / delta) ** 2))))
        print("\nСколько примеров на ячейку нужно, чтобы различить два лучших метода "
              "(мощность 0,8; поправка Холма на 10 пар):")
        print(pd.DataFrame(rows).to_string(index=False))
    ref, meth = timing(d)
    print("\nМедиана времени эталона на пример, с (по N и виду эталона):")
    print(ref.round(1).to_string())
    print("\nМедиана времени всех методов на пример, с:")
    print(meth.round(1).to_string())
    inst = d.drop_duplicates("key").copy()
    w = inst["workers"] if "workers" in inst else 1
    inst["cpu"] = inst["ref_wall"] * w      # процессорное время эталона: время × потоки CP-SAT
    cells = ["prefix", "level", "M", "rho", "start", "hetero"]
    per_cell = inst.groupby(cells).cpu.mean() + \
        d.groupby(cells + ["key"]).wall.sum().groupby(cells).mean()
    total_h = per_cell.sum() * args.reps / 3600
    exact = inst["N"] <= inst["N"][inst["ref_kind"].str.startswith("exact")].max() \
        if inst["ref_kind"].str.startswith("exact").any() else inst["N"] < 0
    print(f"\nОценка полного эксперимента: {args.reps} примеров на ячейку, ячеек {len(per_cell)}: "
          f"{total_h:.0f} ч процессорного времени ≈ {total_h / args.cores:.0f} ч на {args.cores} ядрах "
          f"(по средним временам пилота; точная часть — {100 * inst[exact].cpu.sum() / max(inst.cpu.sum(), 1e-9):.0f} % "
          f"времени эталона).")
    out = args.src.replace(".csv", "_pilot_n.csv")
    rn.to_csv(out, index=False)
    print("→", out)


if __name__ == "__main__":
    main()
