"""Почему кластеризация не выигрывает на скоплениях (статья, раздел 5.3, табл. 3; раздел 5.6).

Три величины на примерах основного эксперимента (однородная группа, M = 8, ρ = 6, 50 повторов
на уровень) и проверки «один тип в скоплении» (разнотипная группа, те же M и ρ):
- целостность скоплений: доля задач скопления, доставшихся одному аппарату (тому, у кого их больше
  всего), в среднем по скоплениям; 1 — каждое скопление целиком у одного аппарата. Считается для
  уровней со скоплениями L1–L4;
- неравномерность загрузки: коэффициент вариации длин путей аппаратов (стандартное отклонение,
  делённое на среднее; простаивающие аппараты входят с нулевой длиной);
- число простаивающих аппаратов.

Примеры порождаются заново из тех же параметров и сидов, что в основном эксперименте, поэтому
совпадают с ним точка в точку; эталон для этих величин не нужен.

Запуск:  python -m experiments.mechanism [--reps 50] [--jobs 4] [--out results/mechanism.csv]
Результат: печать и CSV с медианами по режиму старта, методу и уровню.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from allocsim.instance import make_instance
from allocsim.plan import main_plan, purity_check
from allocsim.runner import METHODS_MAIN

LEVELS = ("L0", "L1", "L2", "L3", "L4")


def integrity(inst, assignment):
    cid = inst.cluster_id
    if (cid < 0).all():
        return np.nan
    vals = []
    for c in np.unique(cid):
        a = assignment[cid == c]
        vals.append(np.bincount(a).max() / len(a))
    return float(np.mean(vals))


def cells(kind="main", M=8, rho=6):
    """Ячейки для расчёта: 'main' — однородная группа основного эксперимента при данных M и ρ;
    'purity' — проверка «один тип в скоплении» (разнотипная группа, M = 8, ρ = 6)."""
    if kind == "purity":
        return purity_check()
    return [c for c in main_plan() if c["M"] == M and c["rho"] == rho and not c["hetero"]]


def _one(args):
    cell, rep = args
    inst = make_instance(cell["spec"], rep)
    rows = []
    for name, fn in METHODS_MAIN.items():
        r = fn(inst)
        L = r.lengths
        rows.append(dict(start=cell["start"], level=cell["level"], rep=rep, method=name, R=inst.R,
                         integrity=integrity(inst, r.assignment), cv_len=float(L.std() / L.mean()),
                         idle=r.idle, max_share=float(np.bincount(r.assignment, minlength=inst.n_agents).max()
                                                      / inst.n_tasks)))
    return rows


def measure(cell_list, reps=50, jobs=1):
    """Величины механизма на каждом примере (строка — пример × метод)."""
    tasks = [(c, rep) for c in cell_list for rep in range(reps)]
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            parts = list(ex.map(_one, tasks, chunksize=4))
    else:
        parts = [_one(t) for t in tasks]
    return pd.DataFrame([r for p in parts for r in p])


def medians(d):
    return d.groupby(["start", "method", "level"])[["integrity", "cv_len", "idle", "max_share"]].median()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--rho", type=int, default=6)
    ap.add_argument("--kind", choices=("main", "purity"), default="main")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results/mechanism.csv")
    args = ap.parse_args()
    med = medians(measure(cells(args.kind, args.M, args.rho), args.reps, args.jobs))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    med.round(3).to_csv(out)
    for col, title in (("integrity", "Целостность скоплений (1 — скопление целиком у одного аппарата)"),
                       ("cv_len", "Неравномерность длин путей (коэффициент вариации)"),
                       ("idle", "Простаивающих аппаратов")):
        print(f"\n{title}, медианы:")
        print(med[col].unstack("level").round(2).to_string())
    print("→", out)


if __name__ == "__main__":
    main()
