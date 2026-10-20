"""Предел точного решения, дополнение: помогают ли подсказки от маршрутного решателя и какая формулировка лучше
на примерах N = 24, которые CP-SAT не доказал за лимит без подсказок.

Запуск из корня репозитория: python -m experiments.exact_hints --time-limit 60 --workers 2
"""
import argparse
import os
from pathlib import Path

import pandas as pd

from allocsim.best_known import solve_routing
from allocsim.exact import solve
from allocsim.instance import InstanceSpec, main_field, make_instance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=600)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--hint-time", type=float, default=5)
    ap.add_argument("--src", default=None)
    args = ap.parse_args()
    src = Path(args.src or f"results/exact_limits/exact_limits_{int(args.time_limit)}s_{args.workers}w_rel.csv")
    d = pd.read_csv(src)
    todo = d[(d.N == 24) & (d.status != "OPTIMAL")]
    rows = []
    for _, c in todo.iterrows():
        sm = c.sigma_mode if "sigma_mode" in c and isinstance(c.sigma_mode, str) else "abs"  # старые файлы — 'abs'
        spec = InstanceSpec(field=main_field(2, 1000.0, int(c.M), int(c.rho), c.level, sigma_mode=sm),
                            n_agents=int(c.M), start_mode=c.start, heterogeneous=bool(c.hetero))
        inst = make_instance(spec, int(c.rep))
        h = solve_routing(inst, c.objective, args.hint_time)
        S_lb = None
        if c.objective == "Lambda":
            S_lb = d[(d.start == c.start) & (d.hetero == c.hetero) & (d.level == c.level) & (d.M == c.M)
                     & (d.rho == c.rho) & (d.rep == c.rep) & (d.objective == "S")].bound.iloc[0]
        row = dict(start=c.start, hetero=c.hetero, level=c.level, M=c.M, N=c.N, rep=c.rep,
                   objective=c.objective, no_hint_value=c.value, no_hint_gap=c.gap,
                   routing_value=h.S if c.objective == "S" else h.Lam)
        for model in ("giant", "vehicle"):
            r = solve(inst, c.objective, time_limit=args.time_limit, workers=args.workers,
                      model=model, hint=h.routes, S_lb=S_lb)
            row[f"{model}_status"] = r.status
            row[f"{model}_value"] = r.S if c.objective == "S" else r.Lam
            row[f"{model}_gap"] = r.gap
            row[f"{model}_wall"] = round(r.wall, 1)
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        pd.DataFrame(rows).to_csv(src.with_name(src.stem + "_hints.csv"), index=False)


if __name__ == "__main__":
    main()
