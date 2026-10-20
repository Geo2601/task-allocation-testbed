"""Предел точного решения, дополнение: насколько маршрутный решатель (лучшее известное решение) близок к доказанному
оптимуму. Берутся примеры из прогона experiments.exact_limits, где CP-SAT доказал оптимум, эвристика решает их
с лимитом --hint-time секунд.

Запуск из корня репозитория: python -m experiments.best_known_quality --src results/exact_limits/<файл>.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from allocsim.best_known import solve_routing
from allocsim.instance import InstanceSpec, main_field, make_instance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--hint-time", type=float, default=5)
    args = ap.parse_args()
    d = pd.read_csv(args.src)
    d = d[d.status == "OPTIMAL"]
    rows = []
    for _, c in d.iterrows():
        sm = c.sigma_mode if "sigma_mode" in c and isinstance(c.sigma_mode, str) else "abs"  # старые файлы — 'abs'
        spec = InstanceSpec(field=main_field(2, 1000.0, int(c.M), int(c.rho), c.level, sigma_mode=sm),
                            n_agents=int(c.M), start_mode=c.start, heterogeneous=bool(c.hetero))
        inst = make_instance(spec, int(c.rep))
        h = solve_routing(inst, c.objective, args.hint_time)
        val = h.S if c.objective == "S" else h.Lam
        rows.append(dict(start=c.start, hetero=c.hetero, level=c.level, M=c.M, N=c.N, rep=c.rep,
                         objective=c.objective, optimum=c.value, heuristic=val,
                         excess=val / c.value - 1))
    out = pd.DataFrame(rows)
    path = Path(args.src).with_name(Path(args.src).stem + f"_heuristic_{int(args.hint_time)}s.csv")
    out.to_csv(path, index=False)
    print(out.groupby(["objective", "N"]).excess.describe(percentiles=[0.5, 0.9]).round(4).to_string())
    print("доля совпадений с оптимумом (≤ 0,1 %):", round((out.excess <= 1e-3).mean(), 3), "→", path)


if __name__ == "__main__":
    main()
