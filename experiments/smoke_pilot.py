"""Сквозная проверка стенда на малых ячейках: генератор → точный фронт → методы → показатели.
Не результат статьи: мало примеров, короткие лимиты. Нужна, чтобы убедиться, что все части
работают вместе и числа правдоподобны.

Запуск из корня репозитория:  python -m experiments.smoke_pilot --time-limit 20 --workers 2
Результат: results/smoke/smoke_pilot.csv
"""
import argparse
import os
import time
from itertools import product
from pathlib import Path

import pandas as pd

from allocsim.exact import pareto_front
from allocsim.instance import InstanceSpec, main_field, make_instance
from allocsim.methods import METHODS
from allocsim.metrics import score_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=20)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-inner", type=int, default=3)
    args = ap.parse_args()
    out = Path("results/smoke")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    sizes = [(4, 3), (4, 6), (8, 3)]
    for (M, rho), lvl, start, het, rep in product(sizes, ("L0", "L4"), ("base", "scattered"),
                                                   (False, True), range(args.reps)):
        inst = make_instance(InstanceSpec(field=main_field(2, 1000.0, M, rho, lvl), n_agents=M,
                                          start_mode=start, heterogeneous=het), rep)
        t0 = time.perf_counter()
        front, sols = pareto_front(inst, n_inner=args.n_inner, time_limit=args.time_limit,
                                   workers=args.workers, heuristic_share=0.1)
        exact = all(r.optimal for r in sols)
        results = [fn(inst) for fn in METHODS.values()]
        scores, ref = score_all(results, front)
        for name, sc in scores.items():
            rows.append(dict(M=M, rho=rho, N=inst.n_tasks, level=lvl, start=start, hetero=het,
                             rep=rep, R=inst.R, method=name, front_exact=exact,
                             front_points=len(ref), **sc))
        pd.DataFrame(rows).to_csv(out / "smoke_pilot.csv", index=False)
        print(f"{time.strftime('%H:%M:%S')} M={M} N={inst.n_tasks} {lvl} {start} het={het} rep={rep} "
              f"фронт: {len(front)} т., точный={exact}, {time.perf_counter() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()
