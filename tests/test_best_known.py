"""Проверки маршрутного решателя (лучшее известное решение) и расчёта фронта с подсказками."""
import numpy as np
import pytest

from allocsim.best_known import best_known_front, solve_routing
from allocsim.exact import pareto_front, solve
from allocsim.fields import FieldSpec
from allocsim.instance import InstanceSpec, make_instance
from allocsim.verify import brute_force

TOL = 0.05


def _inst(n=6, m=3, start="base", het=False, rep=0):
    f = FieldSpec(dim=2, side=1000.0, n_tasks=n, sigma_frac=0.08, n_centers=m)
    return make_instance(InstanceSpec(field=f, n_agents=m, start_mode=start, heterogeneous=het,
                                      min_cover=1 if het else 2), rep)


@pytest.mark.parametrize("start", ["base", "scattered"])
@pytest.mark.parametrize("het", [False, True])
def test_routing_feasible_and_not_better_than_optimum(start, het):
    inst = _inst(start=start, het=het, rep=4)
    bf = brute_force(inst)
    can = inst.can_do()
    for obj in ("S", "Lambda"):
        r = solve_routing(inst, obj, time_limit=2)
        assert sorted(j for x in r.routes for j in x) == list(range(inst.n_tasks))
        assert all(can[k, j] for k, x in enumerate(r.routes) for j in x)
        assert r.S >= bf["S"] - TOL and r.Lam >= bf["Lam"] - TOL
    # на 6 задачах эвристика обычно находит оптимум по S
    r = solve_routing(inst, "S", time_limit=2)
    assert r.S == pytest.approx(bf["S"], rel=0.02)
    # стоимость в OR-Tools совпадает с длиной маршрутов (матрица переходов собрана верно)
    assert r.info["objective_mismatch"] < 0.005 * (inst.n_tasks + inst.n_agents) + 1e-6


def test_front_with_hints_matches_brute_force():
    inst = _inst(n=6, m=3, start="scattered", het=True, rep=2)
    bf = brute_force(inst)["front"]
    front, sols = pareto_front(inst, n_inner=5, time_limit=30, workers=2, heuristic_share=0.05)
    for s, l in front:
        assert any(abs(s - bs) < TOL and abs(l - bl) < TOL for bs, bl in bf)
    assert front[0][1] == pytest.approx(bf[0][1], abs=TOL)
    assert front[-1][0] == pytest.approx(bf[-1][0], abs=TOL)


def test_hint_from_routing_accepted_by_cp_sat():
    inst = _inst(n=8, m=3, start="base", het=False, rep=1)
    h = solve_routing(inst, "Lambda", time_limit=2)
    r = solve(inst, "Lambda", time_limit=30, workers=2, hint=h.routes)
    assert r.optimal and r.Lam <= h.Lam + TOL


def test_routing_respects_length_cap():
    inst = _inst(n=8, m=3, start="scattered", het=False, rep=3)
    free = solve_routing(inst, "S", time_limit=2)
    T = 0.8 * free.Lam
    capped = solve_routing(inst, "S", time_limit=2, T=T)
    if capped.routes is not None:
        assert capped.lengths.max() <= T + 0.05
        assert capped.S >= free.S - 0.05


def test_best_known_front_is_close_to_exact_front():
    inst = _inst(n=6, m=3, start="base", het=True, rep=6)
    bf = brute_force(inst)["front"]
    front, _ = best_known_front(inst, n_inner=4, time_limit=1)
    # каждая точка эвристического фронта не лучше истинного фронта (не может его превзойти)
    for s, l in front:
        assert any(s >= bs - TOL and l >= bl - TOL for bs, bl in bf)
    assert front[0][1] == pytest.approx(bf[0][1], rel=0.05)
