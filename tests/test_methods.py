"""Проверки методов распределения: допустимость, воспроизводимость, базовые свойства."""
import numpy as np
import pytest

from allocsim.exact import solve
from allocsim.instance import InstanceSpec, main_field, make_instance
from allocsim.methods import METHODS, cbba, greedy, kmeans, kmeans_hungarian
from allocsim.methods.base import check_assignment


def _inst(M=4, rho=6, level="L3", start="base", het=False, rep=0):
    return make_instance(InstanceSpec(field=main_field(2, 1000.0, M, rho, level), n_agents=M,
                                      start_mode=start, heterogeneous=het), rep)


CASES = [dict(start=s, het=h, level=l) for s in ("base", "scattered") for h in (False, True)
         for l in ("L0", "L4")]


@pytest.mark.parametrize("name", list(METHODS))
@pytest.mark.parametrize("case", CASES, ids=lambda c: "-".join(map(str, c.values())))
def test_methods_feasible_and_reproducible(name, case):
    inst = _inst(M=4, rho=6, **case)
    a = METHODS[name](inst)
    b = METHODS[name](inst)
    check_assignment(inst, a.assignment)
    assert np.array_equal(a.assignment, b.assignment)
    assert sorted(j for r in a.routes for j in r) == list(range(inst.n_tasks))
    assert a.S == pytest.approx(a.lengths.sum()) and a.Lam == pytest.approx(a.lengths.max())
    if a.raw_lengths is not None:
        # единый решатель не хуже маршрута «как выдал метод» по каждому аппарату
        # (до 12 задач на аппарат он точный)
        small = [k for k, r in enumerate(a.routes) if len(r) <= 12]
        assert (a.lengths[small] <= a.raw_lengths[small] + 1e-6).all()


@pytest.mark.parametrize("name", list(METHODS))
def test_methods_not_better_than_exact(name):
    inst = _inst(M=4, rho=3, level="L3", start="scattered", het=True, rep=2)
    rs = solve(inst, "S", time_limit=60, workers=2)
    rl = solve(inst, "Lambda", time_limit=60, workers=2, S_lb=rs.bound)
    assert rs.optimal and rl.optimal
    r = METHODS[name](inst)
    assert r.S >= rs.S - 0.05 and r.Lam >= rl.Lam - 0.05


def test_greedy_first_choice_is_nearest():
    inst = _inst(M=4, rho=3, level="L0", start="base")
    r = greedy(inst)
    d = np.linalg.norm(inst.points - inst.starts[0], axis=1)
    assert r.raw_routes[0][0] == int(np.argmin(d))


def test_cbba_converges_and_counts_messages():
    for case in CASES:
        inst = _inst(M=8, rho=3, **case)
        r = cbba(inst)
        assert r.info["converged"]
        assert r.messages == r.rounds * 8 * 7
        # пути CBBA содержат ровно свои задачи
        for k, p in enumerate(r.raw_routes):
            assert sorted(p) == sorted(np.flatnonzero(r.assignment == k))


def test_kmeans_recovers_separated_clusters():
    rng = np.random.default_rng(0)
    centers = np.array([[100, 100], [900, 100], [500, 900]])
    x = np.vstack([c + rng.normal(0, 10, (20, 2)) for c in centers])
    lab, c = kmeans(x, 3, np.random.default_rng(1))
    for g in range(3):
        assert len(np.unique(lab[g * 20:(g + 1) * 20])) == 1
    assert len(np.unique(lab)) == 3


def test_kmeans_hungarian_one_group_per_agent_homogeneous():
    inst = _inst(M=4, rho=6, level="L4", start="scattered")
    r = kmeans_hungarian(inst)
    assert len(np.unique(r.assignment)) == 4


def test_kmeans_hungarian_types_respected():
    inst = _inst(M=8, rho=6, level="L2", start="scattered", het=True, rep=3)
    r = kmeans_hungarian(inst)
    check_assignment(inst, r.assignment)
    for typ in range(3):
        agents = np.flatnonzero(inst.capabilities[:, typ])
        used = np.unique(r.assignment[inst.task_type == typ])
        assert set(used) <= set(agents)


def test_reference_front_includes_better_method_points():
    from types import SimpleNamespace
    from allocsim.metrics import reference_front, score_all
    front = [(100.0, 50.0), (80.0, 70.0)]
    better = SimpleNamespace(name="m", S=90.0, Lam=55.0, idle=0, rounds=0, messages=0, wall=0.0,
                             raw_S=None, raw_Lam=None)
    ref = reference_front(front, [better])
    assert (90.0, 55.0) in ref
    scores, _ = score_all([better], front)
    assert scores["m"]["eps"] == 0.0
