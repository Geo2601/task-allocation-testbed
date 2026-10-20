"""Сверка с GACA (статья, раздел 5.7, табл. 5): воспроизведение результата GACA в условиях авторов (Rodriguez и др., arXiv 2608.15884).

Условия статьи: плоскость, старты и задачи равномерно, 5/10/15/20 аппаратов, 2–4 задачи на
аппарат, пути без возврата, критерий — суммарный путь; «доля оптимальности» = 100 · C_opt / C_alg
по стоимости маршрутов, как их выдал метод. У авторов: медиана GACA ≈ 97 %, CBBA — 81–84 %.

Здесь: N равномерно от 2M до 4M; сторона поля 1000 (у авторов 25–55, на долю оптимальности масштаб
не влияет). Эталон — лучшее из маршрутного решателя (--ref-time с) и, если --exact-time > 0,
CP-SAT с подсказкой (доказанный оптимум помечается). Методы — GACA в двух прочтениях Φ и без
разрезания/перехвата групп, CBBA с оценкой «прирост пути» и с оценкой «ценность, убывающая со
временем прибытия» (как в CBBA-Python), жадный, «k-means + венгерский метод».

Прогон можно прервать и продолжить той же командой.
Запуск:  python -m experiments.gaca_check --reps 100 --ref-time 2 --side 25,55   # табл. 5
Результат: results/gaca_check/gaca_setting_ref<с>s_ex<с>s[_side<диапазон>].csv и сводка в печати.
"""
import argparse
import os
import time
from pathlib import Path

import pandas as pd

from allocsim.best_known import solve_routing
from allocsim.exact import solve
from allocsim.fields import FieldSpec
from allocsim.instance import InstanceSpec, make_instance
from allocsim.methods import cbba, gaca, greedy, kmeans_hungarian
from allocsim.methods.cbba import LAM_AUTHOR
from allocsim.seeds import rng

BASE_SEED = 2608
METHODS = {
    "gaca": lambda i: gaca(i),
    "gaca_route": lambda i: gaca(i, phi="route"),
    "gaca_nosplit": lambda i: gaca(i, split=False),
    "gaca_nosteal": lambda i: gaca(i, steal=False),
    "cbba": lambda i: cbba(i),
    "cbba_tdr": lambda i: cbba(i, score="tdr"),
    "gaca_cut_end": lambda i: gaca(i, split_rule="cut_end"),
    # CBBA в настройке авторов GACA (уточнено у авторов 12.10.2026): λ = 0,9 за 10 единиц пути, пакет не ограничен
    "cbba_tdr_author": lambda i: cbba(i, score="tdr", lam_abs=LAM_AUTHOR),
    "cbba_tdr_marginal_author": lambda i: cbba(i, score="tdr_marginal", lam_abs=LAM_AUTHOR),
    "greedy": greedy,
    "kmeans_hungarian": kmeans_hungarian,
}


def instance(M, rep, side_range=None):
    # метки сидов «kt2…» сохранены от первого прогона: от них зависят поля табл. 5
    N = int(rng("kt2N", M, rep).integers(2 * M, 4 * M + 1))
    side = 1000.0 if side_range is None else float(rng("kt2side", M, rep).uniform(*side_range))
    spec = InstanceSpec(field=FieldSpec(dim=2, side=side, n_tasks=N), n_agents=M,
                        start_mode="scattered")
    return make_instance(spec, rep, base_seed=BASE_SEED)


def summary(d):
    piv = d.pivot_table(index="M", columns="method", values="opt_raw", aggfunc="median")
    return (100 * piv).round(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--ref-time", type=float, default=5)
    ap.add_argument("--exact-time", type=float, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--side", default="1000",
                    help="сторона поля: число или диапазон «25,55» (как у авторов; нужно для CBBA с "
                         "наградой, убывающей со временем, — она зависит от масштаба)")
    args = ap.parse_args()
    side_range = tuple(map(float, args.side.split(","))) if "," in args.side else None
    out = Path("results/gaca_check")
    out.mkdir(parents=True, exist_ok=True)
    tag = "" if side_range is None else f"_side{args.side.replace(',', '-')}"
    path = out / f"gaca_setting_ref{args.ref_time:g}s_ex{args.exact_time:g}s{tag}.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["M"], r["rep"]) for r in rows}
    t0 = time.perf_counter()
    for rep in range(args.reps):
        for M in (5, 10, 15, 20):
            if (M, rep) in done:
                continue
            inst = instance(M, rep, side_range)
            h = solve_routing(inst, "S", args.ref_time)
            ref, proven = h.S, False
            if args.exact_time > 0:
                e = solve(inst, "S", time_limit=args.exact_time, workers=args.workers, hint=h.routes)
                if e.S is not None and e.S < ref:
                    ref = e.S
                proven = e.optimal
            for name, fn in METHODS.items():
                r = fn(inst)
                rows.append(dict(M=M, N=inst.n_tasks, rep=rep, method=name, ref=ref,
                                 ref_proven=proven, S_raw=r.raw_S if r.raw_S is not None else r.S,
                                 S=r.S, opt_raw=ref / (r.raw_S if r.raw_S is not None else r.S),
                                 opt=ref / r.S, rounds=r.rounds, idle=r.idle,
                                 converged=r.info.get("converged"), wall=round(r.wall, 3)))
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"{time.strftime('%H:%M:%S')} [{(time.perf_counter() - t0) / 60:5.1f} мин] повтор {rep + 1}/{args.reps}",
              flush=True)
    d = pd.DataFrame(rows)
    print("Медиана доли оптимальности, %, по маршрутам «как выдал метод»:")
    print(summary(d).to_string())
    print("→", path)


if __name__ == "__main__":
    main()
