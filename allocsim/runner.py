"""Прогон эксперимента: пример → эталон (точный фронт или лучший известный) → все методы → показатели.

Каждый пример считается независимо и сохраняется в свой JSON-файл, поэтому прогон можно
прерывать и продолжать, запускать параллельно в нескольких процессах и на разных машинах
(разные диапазоны повторов), а потом собрать всё в одну таблицу (collect).

Эталон (статья, раздел 2; по пробному расчёту предела точного решения): при N ≤ exact_max_n (24) — точный фронт
CP-SAT (pareto_front, лимит на каждую задачу exact_time); иначе — фронт маршрутного решателя
(best_known_front, лимит на точку bk_time), а при N ≤ exact_s_max_n (48) к нему добавляется
доказанный минимум суммарного пути (в пробном расчёте он доказывался всегда, медиана 10 с). В обоих
случаях эталон дополняется решениями методов (metrics.reference_front).
"""
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .best_known import best_known_front
from .exact import nondominated, pareto_front, solve
from .instance import make_instance
from .methods import cbba, gaca, greedy, kmeans_hungarian, skfu
from .metrics import score_all

# пять методов статьи и их варианты для сверок (раздел 5.7)
METHODS_MAIN = {
    "greedy": greedy,
    "cbba": cbba,
    "kmeans_hungarian": kmeans_hungarian,
    "gaca": gaca,
    "skfu": skfu,
}
# λ как у авторов GACA (0,9 за 10 единиц при стороне поля около 40), пересчитанный на сторону поля:
# λ·сторона = 4·ln(1/0,9) ≈ 0,421
TDR_DISCOUNT = float(4 * np.log(1 / 0.9))
METHODS_EXTRA = {
    "cbba_tdr": lambda i: cbba(i, score="tdr"),
    "cbba_tdr_marginal": lambda i: cbba(i, score="tdr_marginal", discount=TDR_DISCOUNT),
    "gaca_route": lambda i: gaca(i, phi="route"),
    "gaca_cut_end": lambda i: gaca(i, split_rule="cut_end"),
    "greedy_round_robin": lambda i: greedy(i, order="round_robin"),
}
ALL_METHODS = dict(METHODS_MAIN, **METHODS_EXTRA)


@dataclass
class RunConfig:
    exact_max_n: int = 24
    exact_s_max_n: int = 48          # до какого N добавлять доказанный минимум S к лучшему известному
    exact_time: float = 60.0         # лимит на одну задачу CP-SAT, с
    bk_time: float = 10.0            # лимит маршрутного решателя на одну точку фронта, с
    n_inner: int = 3                 # точек ε-ограничений в полном фронте
    full_front_reps: int = 10        # полный фронт — у повторов 0…9, дальше облегчённый эталон
    workers: int = 1                 # потоков CP-SAT на один пример
    extra: bool = True               # считать и варианты методов (METHODS_EXTRA)
    heuristic_share: float = 0.1


def instance_key(cell: dict, rep: int, prefix: str) -> str:
    return (f"{prefix}_{cell['level']}_M{cell['M']}_r{cell['rho']}_{cell['start']}_"
            f"{'het' if cell['hetero'] else 'hom'}_rep{rep:03d}")


def _ser(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_ser(v) for v in x]
    if isinstance(x, dict):
        return {k: _ser(v) for k, v in x.items()}
    return x


def run_instance(cell: dict, rep: int, cfg: RunConfig, out_dir: Path, prefix: str) -> Optional[Path]:
    key = instance_key(cell, rep, prefix)
    path = Path(out_dir) / f"{key}.json"
    if path.exists():
        return None
    inst = make_instance(cell["spec"], rep)
    t0 = time.perf_counter()
    if inst.n_tasks <= cfg.exact_max_n:
        lean = rep >= cfg.full_front_reps
        front, sols = pareto_front(inst, n_inner=cfg.n_inner, time_limit=cfg.exact_time,
                                   workers=cfg.workers, heuristic_share=cfg.heuristic_share,
                                   lean=lean)
        kind = "exact" if all(r.optimal for r in sols) else "exact_partial"
        if lean:
            kind += "_lean"
    else:
        front, sols = best_known_front(inst, n_inner=cfg.n_inner, time_limit=cfg.bk_time)
        kind = "best_known"
        if inst.n_tasks <= cfg.exact_s_max_n:
            hint = min((r for r in sols if r.routes is not None), key=lambda r: r.S).routes
            rS = solve(inst, "S", time_limit=cfg.exact_time, workers=cfg.workers, hint=hint)
            sols.append(rS)
            if rS.S is not None:
                front = nondominated(list(front) + [(rS.S, rS.Lam)])
            if rS.optimal:
                kind = "best_known+exact_S"
    ref_wall = time.perf_counter() - t0
    solves = [dict(objective=r.objective, status=r.status, S=r.S, Lam=r.Lam, bound=r.bound,
                   gap=r.gap, wall=round(r.wall, 3), T=r.T, w=r.info.get("w")) for r in sols]

    methods = dict(METHODS_MAIN, **(METHODS_EXTRA if cfg.extra else {}))
    results = [fn(inst) for fn in methods.values()]
    scores, ref = score_all(results, front)
    meth = {}
    for r in results:
        meth[r.name] = dict(S=r.S, Lam=r.Lam, raw_S=r.raw_S, raw_Lam=r.raw_Lam, idle=r.idle,
                            rounds=r.rounds, messages=r.messages, wall=round(r.wall, 4),
                            info={k: v for k, v in r.info.items() if k != "cluster_sizes"},
                            **{k: v for k, v in scores[r.name].items()
                               if k.startswith(("g_", "eps"))})
    rec = dict(key=key, prefix=prefix, rep=rep, level=cell["level"], M=cell["M"], rho=cell["rho"],
               N=inst.n_tasks, start=cell["start"], hetero=cell["hetero"],
               dim=int(inst.points.shape[1]), base_pos=cell["spec"].base_pos, R=inst.R,
               spec=cell["spec"].to_dict(), reference=dict(kind=kind, front=front, ref_front=ref,
                                                          solves=solves, wall=round(ref_wall, 2)),
               methods=meth, config=cfg.__dict__)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_ser(rec), ensure_ascii=False))
    os.replace(tmp, path)
    return path


def collect(out_dir) -> "pandas.DataFrame":
    """Все JSON прогона → таблица «пример × метод»."""
    import pandas as pd
    rows = []
    for f in sorted(Path(out_dir).glob("*.json")):
        rec = json.loads(f.read_text())
        base = {k: rec[k] for k in ("key", "prefix", "rep", "level", "M", "rho", "N", "start",
                                    "hetero", "dim", "base_pos", "R")}
        base["ref_kind"] = rec["reference"]["kind"]
        base["ref_wall"] = rec["reference"]["wall"]
        base["workers"] = rec.get("config", {}).get("workers", 1)
        for name, m in rec["methods"].items():
            row = dict(base, method=name)
            row.update({k: v for k, v in m.items() if k != "info"})
            row["converged"] = m.get("info", {}).get("converged")
            rows.append(row)
    return pd.DataFrame(rows)


def _rescore(rec):
    """Пересчитать показатели методов по сохранённым S, Λ после изменения эталонного фронта."""
    from .exact import eps_indicator, weighted_value
    from .metrics import WEIGHTS
    pts = [tuple(p) for p in rec["reference"]["front"]] + \
        [(m["S"], m["Lam"]) for m in rec["methods"].values()]
    ref = nondominated(pts)
    S0, L0 = min(s for s, _ in ref), min(l for _, l in ref)
    for m in rec["methods"].values():
        m["g_S"], m["g_Lam"] = m["S"] / S0 - 1.0, m["Lam"] / L0 - 1.0
        for w in WEIGHTS:
            F0 = min(weighted_value(s, l, w, (S0, L0)) for s, l in ref)
            m[f"g_w{int(round(100 * w)):02d}"] = weighted_value(m["S"], m["Lam"], w, (S0, L0)) / F0 - 1
        m["eps"] = eps_indicator(m["S"], m["Lam"], ref)
    rec["reference"]["ref_front"] = ref


def refine_exact_S(path: Path, time_limit: float, workers: int) -> Optional[str]:
    """Добавить к эталону «лучшее известное» доказанный минимум S (подвыборка при N = 48):
    решить задачу на минимум S с подсказкой маршрутного решателя, добавить точку во фронт,
    пересчитать показатели методов. Возвращает статус решения или None, если делать нечего."""
    from .best_known import solve_routing
    from .fields import FieldSpec
    from .instance import InstanceSpec
    rec = json.loads(Path(path).read_text())
    if rec["reference"]["kind"] != "best_known":
        return None
    sd = dict(rec["spec"])
    sd["field"] = FieldSpec(**sd["field"])
    inst = make_instance(InstanceSpec(**sd), rec["rep"])
    h = solve_routing(inst, "S", 5)
    r = solve(inst, "S", time_limit=time_limit, workers=workers, hint=h.routes)
    rec["reference"]["solves"].append(dict(objective="S", status=r.status, S=r.S, Lam=r.Lam,
                                           bound=r.bound, gap=r.gap, wall=round(r.wall, 3),
                                           T=None, w=None, refine=True))
    if r.S is not None:
        rec["reference"]["front"] = nondominated([tuple(p) for p in rec["reference"]["front"]] +
                                                 [(r.S, r.Lam)])
    if r.optimal:
        rec["reference"]["kind"] = "best_known+exact_S"
    _rescore(rec)
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(_ser(rec), ensure_ascii=False))
    os.replace(tmp, path)
    return r.status


def _method_record(r, scores=None):
    return dict(S=r.S, Lam=r.Lam, raw_S=r.raw_S, raw_Lam=r.raw_Lam, idle=r.idle, rounds=r.rounds,
                messages=r.messages, wall=round(r.wall, 4),
                info={k: v for k, v in r.info.items() if k != "cluster_sizes"})


def update_methods(path: Path, names) -> int:
    """Пересчитать (или добавить) методы names на готовом примере: эталон не пересчитывается,
    показатели всех методов пересчитываются от того же эталона. Нужно, когда метод уточнён
    после начала эксперимента (например, GACA по ответу авторов)."""
    from .fields import FieldSpec
    from .instance import InstanceSpec
    rec = json.loads(Path(path).read_text())
    sd = dict(rec["spec"])
    sd["field"] = FieldSpec(**sd["field"])
    inst = make_instance(InstanceSpec(**sd), rec["rep"])
    for name in names:
        r = ALL_METHODS[name](inst)
        rec["methods"][name] = _method_record(r)
    _rescore(rec)
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(_ser(rec), ensure_ascii=False))
    os.replace(tmp, path)
    return len(names)
