"""Доказанный минимум суммарного пути для подвыборки примеров с эталоном «лучшее известное»
(статья, раздел 2: при N = 48 минимум S доказывается, Λ — нет). Нужен, чтобы оценить, насколько
лучшее известное S отстаёт от оптимума, и считать g_S от оптимума на этой подвыборке.
Лучше запускать одним процессом со всеми ядрами: CP-SAT доказывает быстрее с несколькими потоками.

Запуск:  python -m experiments.refine_exact_s --out results/main --max-n 48 --reps-to 10 --workers 12
"""
import argparse
import os
import time
from pathlib import Path

from allocsim.runner import refine_exact_S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/main")
    ap.add_argument("--max-n", type=int, default=48)
    ap.add_argument("--reps-to", type=int, default=10)
    ap.add_argument("--time", type=float, default=60)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    import json
    files = []
    for f in sorted(Path(args.out).glob("*.json")):
        rec = json.loads(f.read_text())
        if rec["N"] <= args.max_n and rec["rep"] < args.reps_to and rec["reference"]["kind"] == "best_known":
            files.append(f)
    print(f"{len(files)} примеров без доказанного S*", flush=True)
    t0, ok = time.perf_counter(), 0
    for i, f in enumerate(files, 1):
        st = refine_exact_S(f, args.time, args.workers)
        ok += st == "OPTIMAL"
        print(f"{time.strftime('%H:%M:%S')} [{(time.perf_counter() - t0) / 60:5.1f} мин] {i}/{len(files)} "
              f"{f.stem}: {st}", flush=True)
    print(f"Доказано {ok} из {len(files)}")


if __name__ == "__main__":
    main()
