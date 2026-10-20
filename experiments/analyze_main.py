"""Анализ основного эксперимента (статья, раздел 4): правило выбора метода для трёх предпочтений и
сравнение методов внутри ячеек.

Обучение — повторы 0–6 каждого десятка (70 %), проверка — повторы 7–9 (30 %) и ячейки проверок
устойчивости, которые в обучение не входят.

Запуск:  python -m experiments.analyze_main [--src data/all.csv.gz] [--out results/analysis]
Результат: печать и файлы analysis_<показатель>.txt, cells_<показатель>.csv.
"""
import argparse
from pathlib import Path

from allocsim.analysis import cell_comparison, evaluate_policy

import pandas as pd

PREFS = {"g_S": "важен суммарный путь", "g_Lam": "важно время выполнения",
         "g_w50": "важно поровну", "g_w25": "поровну, вес пути 0,25 (проверка)",
         "g_w75": "поровну, вес пути 0,75 (проверка)"}
METHODS = ("greedy", "cbba", "kmeans_hungarian", "gaca", "skfu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/all.csv.gz")
    ap.add_argument("--out", default="results/analysis")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--min-leaf", type=int, default=40)
    ap.add_argument("--min-gain", type=float, default=0.005)
    args = ap.parse_args()
    d = pd.read_csv(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for pref, title in PREFS.items():
        if pref not in d.columns:
            continue
        tree, reg = evaluate_policy(d, pref, METHODS, lambda inst: (inst["rep"] % 10) < 7,
                                    max_depth=args.depth, min_leaf=args.min_leaf,
                                    min_gain=args.min_gain)
        text = (f"=== {title} ({pref}) ===\nПравило выбора:\n{tree.rules()}\n\n"
                f"Сожаление (средний лишний разрыв против лучшего метода на примере):\n"
                f"{reg.round(4).to_string(index=False)}\n")
        print(text)
        (out / f"analysis_{pref}.txt").write_text(text, encoding="utf-8")
        if pref in ("g_S", "g_Lam", "g_w50"):
            cell_comparison(d, pref, METHODS).to_csv(out / f"cells_{pref}.csv", index=False)
    print("→", out)


if __name__ == "__main__":
    main()
