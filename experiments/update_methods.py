"""Пересчитать отдельные методы на уже посчитанных примерах, не трогая эталон.

Пример (после ответа авторов GACA 12.10.2026: уточнено правило разрезания групп; добавлен CBBA
в настройке авторов):
  python -m experiments.update_methods --out results/experiment --methods gaca,gaca_cut_end,cbba_tdr_marginal --jobs 12
"""
import argparse
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from allocsim.runner import ALL_METHODS, collect, update_methods


def _job(args):
    path, names = args
    return update_methods(path, names)


def _n_tasks(path):
    m = re.search(r"_M(\d+)_r(\d+)_", path.name)
    return int(m.group(1)) * int(m.group(2)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/experiment")
    ap.add_argument("--methods", required=True)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--min-n", type=int, default=0, help="только примеры с числом задач не меньше")
    ap.add_argument("--max-n", type=int, default=10 ** 9, help="только примеры с числом задач не больше")
    ap.add_argument("--no-collect", action="store_true", help="не собирать all.csv в конце")
    args = ap.parse_args()
    names = args.methods.split(",")
    unknown = [n for n in names if n not in ALL_METHODS]
    if unknown:
        raise SystemExit(f"Нет таких методов: {unknown}. Есть: {sorted(ALL_METHODS)}")
    files = [f for f in sorted(Path(args.out).glob("*.json")) if args.min_n <= _n_tasks(f) <= args.max_n]
    files.sort(key=_n_tasks, reverse=True)       # сначала тяжёлые: ровнее загрузка процессов
    print(f"{len(files)} примеров, методы: {names}", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for i, _ in enumerate(ex.map(_job, [(f, names) for f in files], chunksize=8), 1):
            if i % 200 == 0 or i == len(files):
                print(f"{i}/{len(files)} [{(time.perf_counter() - t0) / 60:.1f} мин]", flush=True)
    if args.no_collect:
        return
    d = collect(args.out)
    d.to_csv(Path(args.out) / "all.csv", index=False)
    print("→", Path(args.out) / "all.csv")


if __name__ == "__main__":
    main()
