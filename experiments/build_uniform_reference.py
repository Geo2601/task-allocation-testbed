"""Таблица нормировки показателя скученности: среднее расстояние до ближайшего соседа
для N равномерных точек в единичном квадрате и кубе, 1000 полей на каждое (dim, N).

Запуск из корня репозитория:  python -m experiments.build_uniform_reference
Результат: allocsim/data/uniform_nn_reference.json
"""
import json
import time

from allocsim.clustering import REFERENCE_PATH, simulate_uniform_reference

N_MAX = 256
REPS = 1000
SEED = 20261010


def main():
    t0 = time.time()
    rows = []
    for dim in (2, 3):
        for n in range(2, N_MAX + 1):
            mean, sd = simulate_uniform_reference(n, dim, reps=REPS, seed=SEED)
            rows.append(dict(dim=dim, n=n, mean=mean, sd=sd))
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_PATH.write_text(json.dumps(dict(
        description="Среднее (mean) и ст. отклонение (sd) среднего расстояния до ближайшего "
                    "соседа для n равномерных точек в единичном квадрате (dim=2) или кубе (dim=3)",
        reps=REPS, seed=SEED, table=rows), ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{len(rows)} строк за {time.time() - t0:.0f} с → {REFERENCE_PATH}")


if __name__ == "__main__":
    main()
