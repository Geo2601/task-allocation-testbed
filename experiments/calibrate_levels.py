"""Калибровка уровней скученности (статья, раздел 4).

Для каждой ячейки «аппаратов M × задач на аппарат ρ» и каждого уровня генерируется REPS полей
и считается показатель R. Сравниваются два способа задавать разброс σ:
- 'abs' — σ = доля стороны поля (первоначальный вариант): 0,16 / 0,08 / 0,04 / 0,02;
- 'rel' — σ = доля характерного расстояния между центрами A/m^(1/dim):
  0,32 / 0,16 / 0,08 / 0,04 (при M = 4 на плоскости совпадает с 'abs').

Различимость соседних уровней — вероятность того, что поле более скученного уровня
получит меньшее R, чем поле соседнего менее скученного (AUC; 0,5 — не различимы, 1 — всегда).

Запуск из корня репозитория:  python -m experiments.calibrate_levels
Результаты: results/calibration/*.csv, summary.json
"""
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from allocsim.clustering import clustering_index
from allocsim.fields import FieldSpec, LEVELS, sample_field

OUT = Path("results/calibration")
REPS = 200
SIDE = 1000.0
ABS = {k: v for k, v in LEVELS.items()}
REL = {"L0": None, "L1": 0.32, "L2": 0.16, "L3": 0.08, "L4": 0.04}


def field_for(dim, m_agents, rho, frac, mode, centers_per_agent=1.0):
    m = max(1, int(round(m_agents * centers_per_agent)))
    return FieldSpec(dim=dim, side=SIDE, n_tasks=m_agents * rho, sigma_frac=frac,
                     n_centers=None if frac is None else m, sigma_mode=mode)


def run_grid(dim, agents, rhos, modes, centers_per_agent=1.0):
    rows = []
    for mode, m, rho in product(modes, agents, rhos):
        levels = ABS if mode == "abs" else REL
        for lvl, frac in levels.items():
            f = field_for(dim, m, rho, frac, mode, centers_per_agent)
            for rep in range(REPS):
                pts, _ = sample_field(f, rep)
                rows.append(dict(dim=dim, mode=mode, M=m, rho=rho, N=m * rho,
                                 centers_per_agent=centers_per_agent, level=lvl,
                                 sigma=f.sigma if f.sigma is not None else np.nan,
                                 rep=rep, R=clustering_index(pts, SIDE)))
    return pd.DataFrame(rows)


def auc(lo, hi):
    """P(R из hi < R из lo) + 0,5·P(равны): насколько уровень hi различим от lo."""
    lo, hi = np.asarray(lo), np.asarray(hi)
    less = (hi[:, None] < lo[None, :]).mean()
    eq = (hi[:, None] == lo[None, :]).mean()
    return float(less + 0.5 * eq)


def summarize(df):
    keys = ["dim", "mode", "M", "rho", "N", "centers_per_agent"]
    out = []
    lv = list(LEVELS)
    for key, g in df.groupby(keys):
        row = dict(zip(keys, key))
        for lvl in lv:
            r = g.loc[g.level == lvl, "R"]
            row[f"{lvl}_med"] = float(r.median())
            row[f"{lvl}_q25"] = float(r.quantile(0.25))
            row[f"{lvl}_q75"] = float(r.quantile(0.75))
        for a, b in zip(lv, lv[1:]):
            row[f"auc_{a}_{b}"] = auc(g.loc[g.level == a, "R"], g.loc[g.level == b, "R"])
        out.append(row)
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    main_df = run_grid(2, (4, 8, 16), (3, 6, 12), ("abs", "rel"))
    vol_df = run_grid(3, (8,), (6,), ("abs", "rel"))
    cen_df = pd.concat([run_grid(2, (8,), (6,), ("abs", "rel"), c) for c in (0.5, 2.0)])
    allr = pd.concat([main_df, vol_df, cen_df], ignore_index=True)
    allr.to_csv(OUT / "R_values.csv", index=False)
    summ = summarize(allr)
    summ.to_csv(OUT / "summary.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(dict(reps=REPS, abs_levels=ABS, rel_levels=REL),
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_columns", 40)
    show = ["dim", "mode", "M", "rho", "N", "centers_per_agent"] + [f"{l}_med" for l in LEVELS] + \
           [c for c in summ.columns if c.startswith("auc")]
    print(summ[show].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
