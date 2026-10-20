"""Основной эксперимент и проверки устойчивости (статья, раздел 4) — параллельный прогон.

Каждый пример — отдельный JSON в папке результатов; уже посчитанные пропускаются, поэтому
прогон можно прерывать, продолжать и делить между машинами по диапазонам повторов
(--reps-from / --reps-to). Порядок — по повторам, внутри повтора по всем ячейкам: прерванный
прогон всё равно покрывает все ячейки поровну.

Точное решение (N ≤ 24) лучше считать немногими процессами с несколькими потоками CP-SAT
(в пробном расчёте доказательства получены при 12 потоках), лучшее известное (N > 24) — многими процессами
по одному потоку: маршрутный решатель однопоточный.

Примеры:
  python -m experiments.run_main --set main --max-n 24 --reps-to 3 --jobs 2 --workers 6
  python -m experiments.run_main --set all --min-n 25 --reps-to 3 --jobs 12 --workers 1
  python -m experiments.run_main --set all --reps-from 0 --reps-to 50 --jobs 16
  python -m experiments.run_main --collect                                   # собрать таблицу
Результат: results/<out>/<пример>.json; таблица — results/<out>/all.csv.
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from allocsim.plan import (center_base_check, clusters_double_check, clusters_half_check, main_plan,
                           purity_check, volume_check)
from allocsim.runner import RunConfig, collect, run_instance

SETS = {"main": main_plan, "volume": volume_check, "center": center_base_check}
# Дополнительные проверки (14.10.2026): число скоплений и чистота типов в скоплениях (Г4).
# В --set all не входят, запускаются явно: --set clusters_half,clusters_double,purity
EXTRA_SETS = {"clusters_half": clusters_half_check, "clusters_double": clusters_double_check,
              "purity": purity_check}
ALL_SETS = dict(SETS, **EXTRA_SETS)


def _job(args):
    cell, rep, cfg, out, prefix = args
    t0 = time.perf_counter()
    p = run_instance(cell, rep, cfg, out, prefix)
    return (p is not None), time.perf_counter() - t0, cell, rep, prefix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="main",
                    help="main | volume | center | all (= три предыдущих) | clusters_half | clusters_double | purity")
    ap.add_argument("--reps-from", type=int, default=0)
    ap.add_argument("--reps-to", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--workers", type=int, default=None, help="потоков CP-SAT на пример")
    ap.add_argument("--exact-max-n", type=int, default=24)
    ap.add_argument("--exact-time", type=float, default=60)
    ap.add_argument("--bk-time", type=float, default=10)
    ap.add_argument("--n-inner", type=int, default=3)
    ap.add_argument("--full-front-reps", type=int, default=10,
                    help="полный фронт у повторов меньше этого номера, у остальных — облегчённый эталон")
    ap.add_argument("--levels", default=None, help="например L0,L4")
    ap.add_argument("--max-n", type=int, default=None, help="пропустить ячейки с большим N")
    ap.add_argument("--min-n", type=int, default=None, help="пропустить ячейки с меньшим N")
    ap.add_argument("--exact-s-max-n", type=int, default=48)
    ap.add_argument("--no-extra", action="store_true", help="без вариантов методов (только пять методов статьи)")
    ap.add_argument("--out", default="results/main")
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.collect:
        d = collect(out)
        d.to_csv(out / "all.csv", index=False)
        print(f"{d.key.nunique()} примеров, {len(d)} строк → {out / 'all.csv'}")
        return

    workers = args.workers or max(1, (os.cpu_count() or 2) // args.jobs)
    cfg = RunConfig(exact_max_n=args.exact_max_n, exact_s_max_n=args.exact_s_max_n,
                    exact_time=args.exact_time, bk_time=args.bk_time,
                    n_inner=args.n_inner, workers=workers, extra=not args.no_extra,
                    full_front_reps=args.full_front_reps)
    sets = list(SETS) if args.set == "all" else args.set.split(",")
    levels = set(args.levels.split(",")) if args.levels else None
    tasks = []
    for rep in range(args.reps_from, args.reps_to):
        for s in sets:
            for cell in ALL_SETS[s]():
                if levels and cell["level"] not in levels:
                    continue
                if args.max_n and cell["M"] * cell["rho"] > args.max_n:
                    continue
                if args.min_n and cell["M"] * cell["rho"] < args.min_n:
                    continue
                tasks.append((cell, rep, cfg, out, s))
    print(f"{len(tasks)} примеров; процессов {args.jobs}, потоков CP-SAT на пример {workers}",
          flush=True)
    t0, done, new = time.perf_counter(), 0, 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(_job, t) for t in tasks]
        for f in as_completed(futs):
            ok, wall, cell, rep, prefix = f.result()
            done += 1
            new += ok
            if ok:
                el = (time.perf_counter() - t0) / 60
                print(f"{time.strftime('%H:%M:%S')} [{el:6.1f} мин] {done}/{len(tasks)} {prefix} "
                      f"{cell['level']} M={cell['M']} ρ={cell['rho']} {cell['start']} "
                      f"het={cell['hetero']} rep={rep}: {wall:.0f} с", flush=True)
    d = collect(out)
    d.to_csv(out / "all.csv", index=False)
    print(f"Готово: новых {new}; всего {d.key.nunique()} примеров → {out / 'all.csv'}")


if __name__ == "__main__":
    main()
