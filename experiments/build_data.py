"""Собрать папку data/ — основу таблиц и рисунков статьи — из результатов прогонов.

Полный расчёт с нуля (на Windows те же шаги делают файлы scripts/*.bat):
  1. python -m experiments.exact_limits                       → results/exact_limits/  (раздел 2)
  2. scripts/run_experiment_windows.bat 50                     → results/experiment/    (основной эксперимент,
                                                                 трёхмерная постановка, база в центре)
  3. scripts/run_checks_windows.bat 50                         → results/checks/        (число скоплений,
                                                                 «один тип в скоплении»)
  4. python -m experiments.gaca_check --reps 100 --ref-time 2 --side 25,55   → results/gaca_check/ (табл. 5)
  5. python -m experiments.skfu_check --reps 100               → results/skfu/          (раздел 5.7)
  6. python -m experiments.build_data                          → data/
  7. python -m experiments.article                             → results/article/

Вариант GACA с другим прочтением функции Φ (gaca_route) в основном эксперименте посчитан только
при N ≤ 24; при N > 24 его строки в таблицу не входят. В статье этот вариант не используется.

Запуск:  python -m experiments.build_data [--results results] [--data data] [--jobs 4]
"""
import argparse
import gzip
import os
import shutil
from pathlib import Path

import pandas as pd

from allocsim.runner import collect
from experiments import mechanism, reference_quality

GZ = {"method": "gzip", "mtime": 0}     # одинаковые байты при одинаковом содержимом
FRONT_EXAMPLE = "main_L0_M4_r6_base_het_rep007.json"
EXACT_LIMITS = "exact_limits/exact_limits_600s_12w_rel.csv"
GACA = "gaca_check/gaca_setting_ref2s_ex0s_side25-55.csv"
SKFU = "skfu/skfu_check.csv"


def _gzip_copy(src, dst):
    with open(src, "rb") as fi, open(dst, "wb") as fo, gzip.GzipFile(fileobj=fo, mode="wb", mtime=0) as gz:
        shutil.copyfileobj(fi, gz)


def _table(folder):
    f = folder / "all.csv"
    return pd.read_csv(f) if f.exists() else collect(folder)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default="data")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--no-mechanism", action="store_true", help="не пересчитывать data/mechanism*.csv")
    args = ap.parse_args()
    res, data = Path(args.results), Path(args.data)
    data.mkdir(parents=True, exist_ok=True)

    d = pd.concat([_table(res / "experiment"), _table(res / "checks")], ignore_index=True)
    d = d[~((d.method == "gaca_route") & (d.N > 24))]
    d.to_csv(data / "all.csv.gz", index=False, compression=GZ)
    print(f"all.csv.gz: {d.key.nunique()} примеров, {len(d)} строк")

    q = reference_quality.collect(res / "experiment")
    q.to_csv(data / "reference_quality.csv.gz", index=False, compression=GZ)
    print(f"reference_quality.csv.gz: {len(q)} примеров")

    shutil.copyfile(res / "experiment" / FRONT_EXAMPLE, data / "front_example.json")
    shutil.copyfile(res / EXACT_LIMITS, data / "exact_limits.csv")
    _gzip_copy(res / GACA, data / "gaca_check.csv.gz")
    shutil.copyfile(res / SKFU, data / "skfu_check.csv")

    if not args.no_mechanism:
        mechanism.medians(mechanism.measure(mechanism.cells("main"), 50, args.jobs)).to_csv(data / "mechanism.csv")
        mechanism.medians(mechanism.measure(mechanism.cells("purity"), 50, args.jobs)).to_csv(
            data / "mechanism_purity.csv")
    print("→", data)


if __name__ == "__main__":
    main()
