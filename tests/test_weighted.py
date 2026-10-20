"""Взвешенный критерий F_w = w·S/S° + (1 − w)·Λ/Λ° (три предпочтения, приняты 11.10.2026):
точный минимум сверяется с полным перебором, маршрутный решатель — не лучше оптимума,
показатели g_wXX — с ручным расчётом."""
import numpy as np
import pytest

from allocsim.best_known import best_known_front, solve_routing
from allocsim.exact import solve, weight_coefs, weighted_value
from allocsim.fields import FieldSpec
from allocsim.instance import InstanceSpec, make_instance
from allocsim.methods.base import MethodResult
from allocsim.metrics import gaps, score_all
from allocsim.verify import brute_force

WORKERS = 2


def _inst(n=6, m=3, start="base", het=False, rep=0):
    f = FieldSpec(dim=2, side=1000.0, n_tasks=n, sigma_frac=0.08, n_centers=m)
    return make_instance(InstanceSpec(field=f, n_agents=m, start_mode=start, heterogeneous=het,
                                      min_cover=1 if het else 2), rep)


def _bf_min_F(bf, w):
    norm = (bf["S"], bf["Lam"])
    return min(weighted_value(s, l, w, norm) for s, l in bf["front"]), norm


def test_weight_coefs_proportional():
    a, b = weight_coefs(0.5, (2000.0, 500.0))
    # a·S + b·Λ ∝ 0,5·S/2000 + 0,5·Λ/500  ⇒  a/b = (0,5/2000)/(0,5/500) = 0,25
    assert a / b == pytest.approx(0.25, rel=1e-6)


@pytest.mark.parametrize("model", ["vehicle", "giant", "portfolio"])
@pytest.mark.parametrize("case", [dict(start="base", het=False, rep=0),
                                  dict(start="base", het=True, rep=1),
                                  dict(start="scattered", het=False, rep=2),
                                  dict(start="scattered", het=True, rep=3)],
                         ids=lambda c: "-".join(map(str, c.values())))
def test_weighted_matches_brute_force(case, model):
    inst = _inst(**case)
    bf = brute_force(inst)
    for w in (0.25, 0.5, 0.75):
        F_star, norm = _bf_min_F(bf, w)
        r = solve(inst, "W", w=w, norm=norm, time_limit=60, workers=WORKERS, model=model)
        assert r.optimal
        assert r.info["F"] == pytest.approx(F_star, rel=1e-3)
        assert r.bound <= r.info["F"] * (1 + 1e-3)


def test_weighted_needs_weight_and_norm():
    inst = _inst()
    with pytest.raises(ValueError):
        solve(inst, "W", w=0.5, time_limit=5, workers=WORKERS)
    with pytest.raises(ValueError):
        solve(inst, "W", w=1.0, norm=(1.0, 1.0), time_limit=5, workers=WORKERS)


@pytest.mark.parametrize("start", ["base", "scattered"])
def test_routing_weighted_not_better_than_optimum(start):
    inst = _inst(n=7, start=start, rep=5)
    bf = brute_force(inst)
    for w in (0.25, 0.5, 0.75):
        F_star, norm = _bf_min_F(bf, w)
        r = solve_routing(inst, "W", time_limit=2, w=w, norm=norm)
        assert r.info["F"] >= F_star * (1 - 1e-6)
        assert r.info["F"] == pytest.approx(F_star, rel=0.05)


def test_best_known_front_has_weighted_points():
    inst = _inst(n=7, start="scattered", rep=6)
    front, sols = best_known_front(inst, n_inner=2, time_limit=1)
    assert sum(r.objective == "routing_W" for r in sols) == 3


def _res(name, S, Lam):
    """Результат метода с заданными S и Λ: длины путей Λ, Λ, …, остаток."""
    lengths, rem = [], S
    while rem > Lam + 1e-12:
        lengths.append(Lam)
        rem -= Lam
    lengths.append(rem)
    return MethodResult(name=name, assignment=None, routes=[[0]] * len(lengths),
                        lengths=np.array(lengths))


def test_gaps_weighted_by_hand():
    front = [(100.0, 50.0), (80.0, 80.0)]          # S° = 80, Λ° = 50
    g = gaps(_res("x", 100.0, 80.0), 80.0, 50.0, front)
    # F_0,5: точки фронта 0,5·100/80 + 0,5·1 = 1,125 и 0,5 + 0,5·80/50 = 1,3 ⇒ F° = 1,125
    # метод: 0,5·100/80 + 0,5·80/50 = 1,425 ⇒ g = 1,425/1,125 − 1
    assert g["g_w50"] == pytest.approx(1.425 / 1.125 - 1)
    assert g["g_S"] == pytest.approx(0.25) and g["g_Lam"] == pytest.approx(0.6)
    # точка фронта, дающая минимум F_w, имеет нулевой разрыв
    assert gaps(_res("y", 100.0, 50.0), 80.0, 50.0, front)["g_w50"] == pytest.approx(0.0)


def test_score_all_extreme_point_penalised_by_balanced():
    """«Всё одному аппарату» стоит на фронте (ε = 0), но проигрывает по F_0,5."""
    front = [(120.0, 40.0), (100.0, 100.0)]
    scores, ref = score_all([_res("one_vehicle", 100.0, 100.0)], front)
    sc = scores["one_vehicle"]
    assert sc["eps"] == pytest.approx(0.0) and sc["g_S"] == pytest.approx(0.0)
    assert sc["g_w50"] > 0.4
