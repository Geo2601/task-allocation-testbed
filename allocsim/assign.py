"""Назначение «предметы → аппараты» на минимум суммы и на минимум максимума, с ёмкостями аппаратов
и запрещёнными парами.

Предметы — то, что раздаётся целиком: кластеры задач, участки поля, отдельные задачи.
Каждый предмет получает ровно одного аппарата; аппарат k берёт не больше capacity[k] предметов;
allowed[i, k] = False запрещает отдавать предмет i аппарату k (например, чужой тип задач).

- assign_min_sum   — минимум суммы стоимостей (венгерский метод на размноженных столбцах);
- assign_bottleneck — минимум максимальной стоимости, а при этом пороге — минимум суммы.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

BIG = 1e12


def _expand(cost, capacity, allowed):
    cost = np.asarray(cost, float)
    n_items, n_agents = cost.shape
    capacity = np.ones(n_agents, int) if capacity is None else np.asarray(capacity, int)
    allowed = np.ones_like(cost, bool) if allowed is None else np.asarray(allowed, bool)
    if capacity.sum() < n_items:
        raise ValueError("суммарной ёмкости аппаратов не хватает на все предметы")
    cols = np.repeat(np.arange(n_agents), capacity)
    c = np.where(allowed, cost, BIG)[:, cols]
    return c, cols, allowed[:, cols]


def assign_min_sum(cost, capacity=None, allowed=None):
    """Индекс аппарата для каждого предмета; минимум суммы стоимостей."""
    c, cols, ok = _expand(cost, capacity, allowed)
    rows, cidx = linear_sum_assignment(c)
    if not ok[rows, cidx].all():
        raise ValueError("допустимого назначения нет")
    out = np.empty(c.shape[0], int)
    out[rows] = cols[cidx]
    return out


def _has_full_matching(mask):
    g = csr_matrix(mask.astype(int))
    match = maximum_bipartite_matching(g, perm_type="column")
    return (match >= 0).sum() == mask.shape[0]


def assign_bottleneck(cost, capacity=None, allowed=None):
    """Минимум максимальной стоимости назначения, затем минимум суммы при этом пороге."""
    c, cols, ok = _expand(cost, capacity, allowed)
    if not _has_full_matching(ok):
        raise ValueError("допустимого назначения нет")
    vals = np.unique(c[ok])
    lo, hi = 0, len(vals) - 1
    while lo < hi:   # наименьший порог, при котором все предметы можно раздать
        mid = (lo + hi) // 2
        if _has_full_matching(ok & (c <= vals[mid])):
            hi = mid
        else:
            lo = mid + 1
    tau = vals[lo]
    rows, cidx = linear_sum_assignment(np.where(ok & (c <= tau), c, BIG))
    out = np.empty(c.shape[0], int)
    out[rows] = cols[cidx]
    return out
