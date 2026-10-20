"""Проверки назначения и статистики. Запуск: python -m pytest -q"""
from itertools import product

import numpy as np
import pytest

from allocsim.assign import assign_bottleneck, assign_min_sum
from allocsim.stats import bootstrap_median, compare_methods, holm, paired_wilcoxon


def _brute(cost, capacity, allowed):
    n, m = cost.shape
    best_sum, best_max = None, None
    for a in product(range(m), repeat=n):
        a = np.array(a)
        if not allowed[np.arange(n), a].all():
            continue
        if (np.bincount(a, minlength=m) > capacity).any():
            continue
        c = cost[np.arange(n), a]
        s, mx = c.sum(), c.max()
        best_sum = s if best_sum is None else min(best_sum, s)
        if best_max is None or mx < best_max[0] or (mx == best_max[0] and s < best_max[1]):
            best_max = (mx, s)
    return best_sum, best_max


@pytest.mark.parametrize("seed", range(30))
def test_assignment_matches_brute_force(seed):
    r = np.random.default_rng(seed)
    n, m = r.integers(2, 6), r.integers(2, 4)
    cost = r.integers(1, 20, size=(n, m)).astype(float)
    capacity = r.integers(1, 3, size=m)
    while capacity.sum() < n:
        capacity[r.integers(m)] += 1
    allowed = r.random((n, m)) < 0.8
    allowed[np.arange(n), r.integers(0, m, n)] = True
    best_sum, best_max = _brute(cost, capacity, allowed)
    if best_sum is None:
        with pytest.raises(ValueError):
            assign_min_sum(cost, capacity, allowed)
        return
    a = assign_min_sum(cost, capacity, allowed)
    assert allowed[np.arange(n), a].all()
    assert (np.bincount(a, minlength=m) <= capacity).all()
    assert cost[np.arange(n), a].sum() == pytest.approx(best_sum)
    b = assign_bottleneck(cost, capacity, allowed)
    cb = cost[np.arange(n), b]
    assert allowed[np.arange(n), b].all() and (np.bincount(b, minlength=m) <= capacity).all()
    assert (cb.max(), cb.sum()) == pytest.approx(best_max)


def test_holm_known_values():
    # пример: p = 0,01; 0,04; 0,03 → 0,03; 0,06; 0,06
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_bootstrap_median_covers():
    x = np.random.default_rng(0).normal(5, 1, 400)
    lo, hi = bootstrap_median(x)
    assert lo < 5 < hi


def test_compare_methods_detects_shift():
    r = np.random.default_rng(1)
    base = r.normal(100, 10, 60)
    res = compare_methods({"A": base, "B": base + 5 + r.normal(0, 1, 60), "C": base + r.normal(0, 1, 60)})
    assert res["friedman_p"] < 1e-6
    pab = next(p for p in res["pairs"] if (p["a"], p["b"]) == ("A", "B"))
    assert pab["p_holm"] < 1e-6 and pab["median_diff"] < 0
    assert paired_wilcoxon(base, base)["p"] == 1.0
