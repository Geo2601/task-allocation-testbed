"""Сверка с СКФУ в их условиях (РТК, 2023, т. 11, № 2, табл. 2–3; JKSUCI, 2023, 101580).

Условия авторов: куб 100×100×100, 100 задач равномерно, 5/10/15/20 аппаратов, одна база в углу,
100 карт на каждый размер. Разнотипность — как в их коде симуляции: 5 типов, у задачи один тип,
у аппарата от 1 до 3 типов, каждый тип умеют не меньше двух аппаратов.

У авторов: метод СКФУ (DFCM) короче жадного (GrA) по суммарному пути на 22,1 / 22,0 / 27,5 / 30,0 %
(по табл. 2; в тексте для 5 аппаратов — 20,1 %).

Длина пути считается двумя способами: с возвратом на базу (как везде в стенде) и без возврата
(как в коде авторов). Для жадного — по маршрутам «как летел» (порядок выбора задач) и после единого
решателя маршрутов; у метода СКФУ маршруты строит решатель (у авторов — перебор или имитация
отжига внутри кластера).

Запуск:  python -m experiments.skfu_check --reps 100
Результат: results/skfu/skfu_check.csv и сводка в печати.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from allocsim.fields import FieldSpec
from allocsim.instance import InstanceSpec, make_instance
from allocsim.methods import greedy, kmeans_hungarian, skfu

BASE_SEED = 2023
METHODS = {
    "skfu": lambda i: skfu(i),
    "skfu_strips": lambda i: skfu(i, partition="strips"),
    "greedy": lambda i: greedy(i),
    "greedy_round_robin": lambda i: greedy(i, order="round_robin"),
    "kmeans_hungarian": kmeans_hungarian,
}
# у авторов (РТК, 2023, табл. 2 и 3)
PAPER_S = {"DFCM": {5: 3424.12, 10: 4457.12, 15: 4961.51, 20: 5459.49},
           "GrA": {5: 4395.45, 10: 5716.99, 15: 6844.80, 20: 7803.58}}
PAPER_T = {"DFCM": {5: 823.80, 10: 559.66, 15: 500.65, 20: 544.22},
           "GrA": {5: 1130.96, 10: 752.90, 15: 621.21, 20: 524.56}}


def instance(M, rep):
    spec = InstanceSpec(field=FieldSpec(dim=3, side=100.0, n_tasks=100), n_agents=M,
                        start_mode="base", base_pos="corner", heterogeneous=True, n_types=5,
                        max_types_per_agent=3, caps_scheme="random", min_cover=2)
    return make_instance(spec, rep, base_seed=BASE_SEED)


def lengths(inst, routes, closed):
    out = []
    for k, r in enumerate(routes):
        if not r:
            out.append(0.0)
            continue
        pts = np.vstack([inst.starts[k], inst.points[r]] + ([inst.starts[k][None]] if closed else []))
        out.append(float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    args = ap.parse_args()
    out = Path("results/skfu")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "skfu_check.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["M"], r["rep"]) for r in rows}
    t0 = time.perf_counter()
    for rep in range(args.reps):
        for M in (5, 10, 15, 20):
            if (M, rep) in done:
                continue
            inst = instance(M, rep)
            for name, fn in METHODS.items():
                r = fn(inst)
                row = dict(M=M, rep=rep, method=name, idle=r.idle,
                           leftovers=r.info.get("leftovers"))
                for closed in (True, False):
                    tag = "closed" if closed else "open"
                    L = lengths(inst, r.routes, closed)
                    row[f"S_{tag}"], row[f"Lam_{tag}"] = L.sum(), L.max()
                    if r.raw_routes is not None:
                        Lr = lengths(inst, r.raw_routes, closed)
                        row[f"S_raw_{tag}"], row[f"Lam_raw_{tag}"] = Lr.sum(), Lr.max()
                rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)
        if (rep + 1) % 10 == 0:
            print(f"{time.strftime('%H:%M:%S')} [{(time.perf_counter() - t0) / 60:4.1f} мин] "
                  f"{rep + 1}/{args.reps}", flush=True)
    d = pd.DataFrame(rows)
    print(summary(d).to_string())
    print("→", path)


def summary(d):
    """Средние S и выигрыш метода СКФУ над жадным: у нас (разные способы считать путь) и у авторов."""
    m = d.groupby(["M", "method"]).mean(numeric_only=True)
    res = []
    for M in (5, 10, 15, 20):
        sk = m.loc[(M, "skfu")]
        row = dict(M=M, skfu_S_open=sk.S_open, skfu_S_closed=sk.S_closed,
                   paper_DFCM_S=PAPER_S["DFCM"][M], paper_GrA_S=PAPER_S["GrA"][M],
                   paper_gain=1 - PAPER_S["DFCM"][M] / PAPER_S["GrA"][M])
        for g in ("greedy", "greedy_round_robin"):
            gr = m.loc[(M, g)]
            row[f"gain_vs_{g}_rerouted_open"] = 1 - sk.S_open / gr.S_open
            row[f"gain_vs_{g}_raw_open"] = 1 - sk.S_open / gr.S_raw_open
        row["gain_vs_round_robin_raw_closed"] = 1 - sk.S_closed / m.loc[(M, "greedy_round_robin")].S_raw_closed
        row["skfu_Lam_open"], row["paper_DFCM_T"] = sk.Lam_open, PAPER_T["DFCM"][M]
        res.append(row)
    return pd.DataFrame(res).round(3)


if __name__ == "__main__":
    main()
