"""Проверки GACA: группы-цепочки по графу ближайших соседей, корректность распределения,
повторяемость, разнотипная группа, варианты прочтения."""
import numpy as np
import pytest

from allocsim.fields import FieldSpec
from allocsim.instance import InstanceSpec, main_field, make_instance
from allocsim.methods import gaca, nn_chains


def _D(pts):
    pts = np.asarray(pts, float)
    return np.linalg.norm(pts[:, None] - pts[None], axis=2)


def test_chains_partition_and_paths():
    r = np.random.default_rng(0)
    for n in (1, 2, 5, 30, 120):
        D = _D(r.uniform(0, 100, (n, 2)))
        chains = nn_chains(D)
        assert sorted(j for c in chains for j in c) == list(range(n))
        assert all(len(c) == len(set(c)) for c in chains)


def test_chains_follow_nearest_neighbours():
    # две далёкие пары и точка, ближайшая к задаче 1: группы {0, 1, 4} и {2, 3}
    pts = [(0, 0), (1, 0), (100, 0), (101, 0), (50, 80)]
    chains = sorted(sorted(c) for c in nn_chains(_D(pts)))
    assert chains == [[0, 1, 4], [2, 3]]


def test_chains_resolve_degree_conflicts():
    # «звезда»: к центру ближе всех четыре точки — у центра не больше двух соседей в цепочке
    pts = [(0, 0), (1, 0), (-1.1, 0), (0, 1.2), (0, -1.3)]
    chains = nn_chains(_D(pts))
    for c in chains:
        if 0 in c:
            i = c.index(0)
            assert len([x for x in (i - 1, i + 1) if 0 <= x < len(c)]) <= 2
    assert sorted(j for c in chains for j in c) == list(range(5))


def test_chain_order_is_path():
    pts = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    (c,) = nn_chains(_D(pts))
    assert c in ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])


CASES = [(M, rho, lvl, start, het) for M, rho in ((4, 3), (8, 6))
         for lvl in ("L0", "L4") for start in ("base", "scattered") for het in (False, True)]


@pytest.mark.parametrize("case", CASES, ids=lambda c: "-".join(map(str, c)))
def test_gaca_valid_and_deterministic(case):
    M, rho, lvl, start, het = case
    inst = make_instance(InstanceSpec(field=main_field(2, 1000.0, M, rho, lvl), n_agents=M,
                                      start_mode=start, heterogeneous=het), 1)
    a, b = gaca(inst), gaca(inst)
    assert (a.assignment == b.assignment).all()
    can = inst.can_do()
    assert can[a.assignment, np.arange(inst.n_tasks)].all()
    assert sorted(j for r in a.raw_routes for j in r) == list(range(inst.n_tasks))
    assert a.info["unassigned_at_stop"] == 0
    assert a.rounds < 20 * inst.n_tasks
    # единый решатель маршрутов не ухудшает путь «как выдал метод»
    assert a.S <= a.raw_S + 1e-6


@pytest.mark.parametrize("kw", [dict(phi="route"), dict(split=False), dict(steal=False)])
def test_gaca_variants_valid(kw):
    inst = make_instance(InstanceSpec(field=main_field(2, 1000.0, 8, 6, "L2"), n_agents=8,
                                      start_mode="scattered", heterogeneous=True), 2)
    r = gaca(inst, **kw)
    assert inst.can_do()[r.assignment, np.arange(inst.n_tasks)].all()
    assert r.name.startswith("gaca_")


def test_gaca_heterogeneous_groups_single_type():
    inst = make_instance(InstanceSpec(field=main_field(2, 1000.0, 8, 6, "L3"), n_agents=8,
                                      start_mode="scattered", heterogeneous=True), 0)
    r = gaca(inst)
    # маршрут «как выдал метод» состоит из цепочек одного типа: соседние задачи разного типа
    # встречаются только на стыке групп, и аппарат умеет все свои типы
    for k, route in enumerate(r.raw_routes):
        assert all(inst.capabilities[k, inst.task_type[j]] for j in route)


def test_gaca_single_cluster_one_agent_near():
    """Скопление у старта одного аппарата и далеко от остальных достаётся ему целиком."""
    f = FieldSpec(dim=2, side=1000.0, n_tasks=6)
    inst = make_instance(InstanceSpec(field=f, n_agents=3, start_mode="scattered"), 0)
    inst.points = np.array([[10, 10], [20, 12], [30, 9], [40, 11], [50, 10], [60, 12]], float)
    inst.starts = np.array([[0, 10], [900, 900], [900, 100]], float)
    r = gaca(inst)
    assert (r.assignment == 0).all()
