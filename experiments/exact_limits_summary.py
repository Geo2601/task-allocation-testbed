"""Предел точного решения, сводка по файлу прогона: сколько задач доказано при каждом размере, какая доля доказывается
при меньшем лимите (по времени до доказательства), насколько маршрутный решатель отстаёт от
доказанного оптимума.

Запуск из корня репозитория:
  python -m experiments.exact_limits_summary                     # самый свежий файл прогона
  python -m experiments.exact_limits_summary --src results/exact_limits/exact_limits_600s_6w_rel.csv
Результат: печать и файл <src>_summary.csv рядом с исходным.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

BUDGETS = (10, 30, 60, 120, 300, 600)


def summarize(d):
    rows = []
    for (obj, N), g in d.groupby(["objective", "N"]):
        opt = g[g.status == "OPTIMAL"]
        row = dict(objective=obj, N=N, n=len(g), proven=len(opt),
                   wall_median_proven=round(opt.wall.median(), 1) if len(opt) else None,
                   gap_max_unproven=round(g[g.status != "OPTIMAL"].gap.max(), 3)
                   if (g.status != "OPTIMAL").any() else None)
        lim = g.time_limit.iloc[0]
        for b in BUDGETS:
            if b <= lim:
                row[f"share_{b}s"] = round((opt.wall <= b).sum() / len(g), 3)
        if "routing_value" in g and g.routing_value.notna().any() and len(opt):
            ex = opt.routing_value / opt.value - 1
            row["routing_excess_median"] = round(ex.median(), 4)
            row["routing_excess_max"] = round(ex.max(), 4)
            row["routing_match_share"] = round((ex <= 1e-3).mean(), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None)
    args = ap.parse_args()
    if args.src is None:
        pat = re.compile(r"^exact_limits_\d+s_\d+w(_(abs|rel))?$")
        files = [f for f in Path("results/exact_limits").glob("exact_limits_*.csv") if pat.match(f.stem)]
        if not files:
            raise SystemExit("В results/exact_limits нет файлов прогона (python -m experiments.exact_limits)")
        args.src = str(max(files, key=lambda f: f.stat().st_mtime))
        print("Файл:", args.src)
    d = pd.read_csv(args.src)
    s = summarize(d)
    path = Path(args.src).with_name(Path(args.src).stem + "_summary.csv")
    s.to_csv(path, index=False)
    print(s.to_string(index=False))
    print("Отдельно по сериям (старт × типы × уровень), доказано из всего:")
    by = d.assign(ok=d.status == "OPTIMAL").groupby(["N", "start", "hetero", "level"]).ok.agg(["sum", "size"])
    print(by.to_string())
    print("→", path)


if __name__ == "__main__":
    main()
