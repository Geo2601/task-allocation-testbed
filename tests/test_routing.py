"""Проверки единого решателя маршрутов."""
from itertools import permutations

import numpy as np
import pytest

from allocsim.fields import FieldSpec
from allocsim.instance import InstanceSpec, make_instance
from allocsim.routing import _dist, _held_karp, _local_search, best_route, reroute


def _inst(n, start, rep=0, m=2):
    f = FieldSpec(dim=2, side=1000.0, n_tasks=n, sigma_frac=0.1, n_centers=3)
    return make_instance(InstanceSpec(field=f, n_agents=m, start_mode=start), rep)


def _brute(d):
    n = d.shape[0] - 1
    best = np.inf
    for p in permutations(range(n)):
        seq = (0,) + tuple(j + 1 for j in p) + (0,)
        best = min(best, sum(d[a, b] for a, b in zip(seq, seq[1:])))
    return best


@pytest.mark.parametrize("start", ["base", "scattered"])
@pytest.mark.parametrize("n", [2, 4, 7])
def test_held_karp_matches_brute_force(start, n):
    for rep in range(3):
        inst = _inst(n, start, rep)
        d = _dist(inst, 0, list(range(n)))
        order, cost = _held_karp(d)
        assert sorted(order) == list(range(n))
        assert cost == pytest.approx(_brute(d))


@pytest.mark.parametrize("start", ["base", "scattered"])
def test_local_search_close_to_exact(start):
    excess = []
    for rep in range(6):
        inst = _inst(11, start, rep)
        d = _dist(inst, 0, list(range(11)))
        _, exact = _held_karp(d)
        order, ls = _local_search(d)
        assert sorted(order) == list(range(11))
        assert ls >= exact - 1e-6
        excess.append(ls / exact - 1)
    assert max(excess) < 0.03 and np.median(excess) < 0.005


def test_best_route_and_reroute_consistency():
    inst = _inst(30, "base", 1, m=3)
    assignment = np.arange(30) % 3
    routes, lengths = reroute(inst, assignment)
    for k in range(3):
        assert sorted(routes[k]) == list(np.flatnonzero(assignment == k))
        pts = np.vstack([inst.starts[k], inst.points[routes[k]], inst.starts[k]])
        assert lengths[k] == pytest.approx(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    assert best_route(inst, 0, []) == ([], 0.0)
