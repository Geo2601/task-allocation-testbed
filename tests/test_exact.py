"""Проверки точного решения: сверка с полным перебором и с независимой MILP-моделью в HiGHS.
Запуск: python -m pytest -q"""
import numpy as np
import pytest

from allocsim.exact import eps_indicator, nondominated, pareto_front, route_lengths, solve, solve_lex
from allocsim.fields import FieldSpec
from allocsim.instance import InstanceSpec, main_field, make_instance
from allocsim.verify import brute_force, milp_solve

WORKERS = 2
TOL = 0.05   # округление расстояний до 0,01: ошибка не больше 0,005 на дугу


def _inst(n=6, m=3, start="base", het=False, level="L2", rep=0, n_types=3):
    f = FieldSpec(dim=2, side=1000.0, n_tasks=n) if level == "L0" else \
        FieldSpec(dim=2, side=1000.0, n_tasks=n, sigma_frac=0.08, n_centers=m)
    spec = InstanceSpec(field=f, n_agents=m, start_mode=start, heterogeneous=het,
                        n_types=n_types, min_cover=1 if het else 2)
    return make_instance(spec, rep)


def _check_partition(inst, routes):
    flat = sorted(j for r in routes for j in r)
    assert flat == list(range(inst.n_tasks))
    can = inst.can_do()
    for k, r in enumerate(routes):
        assert all(can[k, j] for j in r)


CASES = [dict(start=s, het=h, level=l, rep=r)
         for s in ("base", "scattered") for h in (False, True) for l in ("L0", "L2") for r in (0, 1)]


@pytest.mark.parametrize("model", ["vehicle", "giant", "portfolio"])
@pytest.mark.parametrize("case", CASES, ids=lambda c: "-".join(map(str, c.values())))
def test_matches_brute_force(case, model):
    inst = _inst(n=6, m=3, **case)
    bf = brute_force(inst)
    rs = solve(inst, "S", time_limit=60, workers=WORKERS, model=model)
    rl = solve(inst, "Lambda", time_limit=60, workers=WORKERS, model=model)
    assert rs.optimal and rl.optimal
    assert rs.S == pytest.approx(bf["S"], abs=TOL)
    assert rl.Lam == pytest.approx(bf["Lam"], abs=TOL)
    _check_partition(inst, rs.routes)
    _check_partition(inst, rl.routes)
    assert np.allclose(route_lengths(inst, rs.routes), rs.lengths)


@pytest.mark.parametrize("model", ["vehicle", "giant"])
@pytest.mark.parametrize("case", CASES[::2], ids=lambda c: "-".join(map(str, c.values())))
def test_front_matches_brute_force(case, model):
    inst = _inst(n=6, m=3, **case)
    bf = brute_force(inst)["front"]
    front, sols = pareto_front(inst, n_inner=6, time_limit=60, workers=WORKERS, model=model,
                               heuristic_share=0)
    assert all(r.optimal for r in sols)
    # каждая найденная точка лежит на истинном фронте
    for s, l in front:
        assert any(abs(s - bs) < TOL and abs(l - bl) < TOL for bs, bl in bf), (s, l, bf)
    # крайние точки совпадают
    assert front[0][1] == pytest.approx(bf[0][1], abs=TOL)
    assert front[0][0] == pytest.approx(bf[0][0], abs=TOL)
    assert front[-1][0] == pytest.approx(bf[-1][0], abs=TOL)
    assert front[-1][1] == pytest.approx(bf[-1][1], abs=TOL)


@pytest.mark.parametrize("model", ["vehicle", "giant"])
def test_base_homogeneous_min_sum_is_single_tour(model):
    """Вырождение: в режиме «база» однородной группы минимум S даёт один аппарат."""
    inst = _inst(n=6, m=3, start="base", het=False, level="L2", rep=3)
    r = solve_lex(inst, "S", time_limit=60, workers=WORKERS, model=model)
    assert r.optimal
    assert sum(1 for x in r.routes if x) == 1


@pytest.mark.parametrize("case", [dict(start="base", het=False), dict(start="scattered", het=True),
                                  dict(start="base", het=True)],
                         ids=lambda c: "-".join(map(str, c.values())))
def test_matches_milp(case):
    inst = _inst(n=7, m=3, level="L2", rep=5, **case)
    for obj, attr in (("S", "S"), ("Lambda", "Lam")):
        val, status = milp_solve(inst, obj, time_limit=120)
        assert status == 0
        for model in ("vehicle", "giant"):
            r = solve(inst, obj, time_limit=120, workers=WORKERS, model=model)
            assert r.optimal
            assert getattr(r, attr) == pytest.approx(val, abs=TOL)


@pytest.mark.parametrize("case", [dict(start="base", het=False), dict(start="scattered", het=False),
                                  dict(start="base", het=True), dict(start="scattered", het=True)],
                         ids=lambda c: "-".join(map(str, c.values())))
def test_models_agree_on_larger(case):
    """Две формулировки дают одинаковый оптимум на примерах, где перебор уже невозможен."""
    inst = _inst(n=10, m=3, level="L2", rep=7, **case)
    for obj, attr in (("S", "S"), ("Lambda", "Lam")):
        a = solve(inst, obj, time_limit=120, workers=WORKERS, model="vehicle")
        b = solve(inst, obj, time_limit=120, workers=WORKERS, model="giant")
        assert a.optimal and b.optimal
        assert getattr(a, attr) == pytest.approx(getattr(b, attr), abs=TOL)


def test_giant_single_vehicle():
    inst = _inst(n=6, m=1, start="base", het=False, level="L0", rep=0)
    bf = brute_force(inst)
    r = solve(inst, "S", time_limit=60, workers=WORKERS, model="giant")
    assert r.optimal and r.S == pytest.approx(bf["S"], abs=TOL)


def test_symmetry_breaking_keeps_optimum():
    inst = _inst(n=7, m=3, start="base", het=False, level="L2", rep=2)
    a = solve(inst, "Lambda", time_limit=60, workers=WORKERS, symmetry="index")
    b = solve(inst, "Lambda", time_limit=60, workers=WORKERS, symmetry="length")
    c = solve(inst, "Lambda", time_limit=60, workers=WORKERS, symmetry=None)
    assert a.optimal and b.optimal and c.optimal
    assert a.Lam == pytest.approx(c.Lam, abs=TOL) and b.Lam == pytest.approx(c.Lam, abs=TOL)


def test_lower_bound_cut_keeps_optimum():
    inst = _inst(n=6, m=3, start="scattered", het=True, level="L2", rep=1)
    bf = brute_force(inst)
    r = solve(inst, "Lambda", time_limit=60, workers=WORKERS, S_lb=bf["S"])
    assert r.optimal and r.Lam == pytest.approx(bf["Lam"], abs=TOL)


def test_eps_indicator():
    front = [(10.0, 5.0), (8.0, 7.0)]
    assert eps_indicator(10.0, 5.0, front) == pytest.approx(0.0)
    assert eps_indicator(11.0, 5.0, front) == pytest.approx(0.1)
    assert eps_indicator(8.8, 7.7, front) == pytest.approx(0.1)
    assert nondominated([(10, 5), (8, 7), (9, 7), (10, 6)]) == [(10, 5), (8, 7)]
