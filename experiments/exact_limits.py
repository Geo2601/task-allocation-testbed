"""Предел точного решения: до какого размера точное решение доказывается за лимит времени
(статья, раздел 2, пробный расчёт на 65 примерах).

Для размеров основного эксперимента (M, N = M·ρ) по возрастанию N, обоих режимов старта, однородной
и разнотипной группы и крайних уровней скученности (L0, L4) решаются две задачи: минимум
суммарного пути S и минимум времени выполнения Λ (с нижней оценкой S из первой задачи).
Перед CP-SAT маршрутный решатель за --hint-time секунд находит хорошее решение — оно идёт
подсказкой (так эталон и будет считаться в основном эксперименте) и заодно показывает качество
лучшего известного решения там, где оптимум доказан.

Фиксируется: доказан ли оптимум, время до доказательства, разрыв до нижней оценки решателя,
значение маршрутного решателя. По времени до доказательства потом видно, какая доля задач
решается при любом меньшем лимите.

Порядок — по возрастанию N сразу во всех сериях, поэтому прерванный прогон всё равно даёт
полную картину для малых N. Если в серии (старт × типы × уровень) ни один пример размера N
не доказан ни по одной задаче, бо́льшие N в этой серии пропускаются.

Прогон можно остановить (Ctrl+C) и запустить той же командой снова — сделанное пропускается.

Запуск из корня репозитория:
  python -m experiments.exact_limits                              # 600 с, все ядра, N ≤ 48
  python -m experiments.exact_limits --time-limit 60 --workers 2  # быстрый прогон
Результат: results/exact_limits/exact_limits_<лимит>s_<ядра>w_<σ>.csv
"""
import argparse
import os
import time
from itertools import product
from pathlib import Path

import pandas as pd

from allocsim.best_known import solve_routing
from allocsim.exact import solve
from allocsim.instance import InstanceSpec, main_field, make_instance
from allocsim.plan import SIGMA_MODE

SIZES = [(4, 3), (4, 6), (8, 3), (4, 12), (8, 6), (16, 3), (8, 12), (16, 6), (16, 12)]
KEY = ["start", "hetero", "level", "M", "rho", "rep"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=600)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--reps-small", type=int, default=2, help="повторов при N ≤ 24")
    ap.add_argument("--reps-large", type=int, default=1, help="повторов при N > 24")
    ap.add_argument("--levels", default="L0,L4")
    ap.add_argument("--max-n", type=int, default=48, help="наибольшее N (96 при 60 с не решалось)")
    ap.add_argument("--hint-time", type=float, default=5, help="0 — без подсказок")
    ap.add_argument("--model", default="portfolio")
    ap.add_argument("--sigma-mode", default=SIGMA_MODE)
    args = ap.parse_args()

    out = Path("results/exact_limits")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"exact_limits_{int(args.time_limit)}s_{args.workers}w_{args.sigma_mode}.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {tuple(r[k] for k in KEY): r for r in rows if r["objective"] == "Lambda"}
    if rows:
        print(f"Продолжение: в {path} уже {len(done)} примеров", flush=True)

    levels = args.levels.split(",")
    series = list(product(("base", "scattered"), (False, True), levels))
    failed_n = {s: None for s in series}
    t_start = time.perf_counter()
    for M, rho in SIZES:
        N = M * rho
        if N > args.max_n:
            continue
        reps = args.reps_small if N <= 24 else args.reps_large
        for ser in series:
            start, het, lvl = ser
            if failed_n[ser] is not None and N > failed_n[ser]:
                continue
            any_ok = False
            for rep in range(reps):
                key = (start, het, lvl, M, rho, rep)
                if key in done:
                    prev = [r for r in rows if tuple(r[k] for k in KEY) == key]
                    any_ok |= any(r["status"] == "OPTIMAL" for r in prev)
                    continue
                spec = InstanceSpec(field=main_field(2, 1000.0, M, rho, lvl, sigma_mode=args.sigma_mode),
                                    n_agents=M, start_mode=start, heterogeneous=het)
                inst = make_instance(spec, rep)
                hS = hL = None
                if args.hint_time > 0:
                    hS = solve_routing(inst, "S", args.hint_time)
                    hL = solve_routing(inst, "Lambda", args.hint_time)
                rs = solve(inst, "S", time_limit=args.time_limit, workers=args.workers,
                           model=args.model, hint=hS.routes if hS else None)
                rl = solve(inst, "Lambda", time_limit=args.time_limit, workers=args.workers,
                           model=args.model, S_lb=rs.bound, hint=hL.routes if hL else None)
                for obj, r, h in (("S", rs, hS), ("Lambda", rl, hL)):
                    rows.append(dict(start=start, hetero=het, level=lvl, M=M, rho=rho, N=N, rep=rep,
                                     objective=obj, status=r.status, wall=round(r.wall, 2),
                                     gap=r.gap, value=r.S if obj == "S" else r.Lam, bound=r.bound,
                                     routing_value=None if h is None else (h.S if obj == "S" else h.Lam),
                                     R=round(inst.R, 3), time_limit=args.time_limit,
                                     workers=args.workers, model=args.model,
                                     sigma_mode=args.sigma_mode, hint_time=args.hint_time))
                    any_ok |= r.optimal
                pd.DataFrame(rows).to_csv(path, index=False)
                el = (time.perf_counter() - t_start) / 3600
                print(f"{time.strftime('%H:%M:%S')} [{el:4.1f} ч] {start:9s} het={het!s:5s} {lvl} "
                      f"M={M:2d} N={N:3d} rep={rep}  S: {rs.status[:4]} {rs.wall:6.1f}с gap={_g(rs)}  "
                      f"Λ: {rl.status[:4]} {rl.wall:6.1f}с gap={_g(rl)}", flush=True)
            if not any_ok and failed_n[ser] is None:
                failed_n[ser] = N
    print("Готово →", path)


def _g(r):
    return "—" if r.gap is None else f"{r.gap:.3f}"


if __name__ == "__main__":
    main()
