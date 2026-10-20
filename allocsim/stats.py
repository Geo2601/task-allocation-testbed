"""Статистика сравнения методов (статья, раздел 4). Дизайн парный: все методы на одних
и тех же примерах. Поправка Холма, бутстреп, парный критерий Уилкоксона, критерий Фридмана и попарные
сравнения.
"""
from itertools import combinations

import numpy as np
from scipy import stats


def holm(pvals):
    """Поправка Холма. На входе список p-значений, на выходе скорректированные (в том же порядке)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def bootstrap_median(x, n_boot=5000, seed=0, level=0.95):
    """Бутстреп-интервал для медианы."""
    x = np.asarray(x, float)
    r = np.random.default_rng(seed)
    meds = np.median(x[r.integers(0, len(x), size=(n_boot, len(x)))], axis=1)
    a = (1 - level) / 2
    return float(np.quantile(meds, a)), float(np.quantile(meds, 1 - a))


def paired_wilcoxon(a, b):
    """Парный критерий Уилкоксона для a − b; медиана разности и бутстреп-интервал для неё."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if len(d) < 5 or np.allclose(d, 0):
        return dict(n=len(d), p=1.0, median_diff=float(np.median(d)) if len(d) else 0.0,
                    ci=(0.0, 0.0))
    res = stats.wilcoxon(a, b, zero_method="wilcox")
    return dict(n=len(d), p=float(res.pvalue), median_diff=float(np.median(d)),
                ci=bootstrap_median(d))


def compare_methods(values: dict):
    """values: {метод: массив показателя по одним и тем же примерам}.
    Критерий Фридмана, затем попарный Уилкоксон с поправкой Холма."""
    names = list(values)
    mat = np.column_stack([np.asarray(values[k], float) for k in names])
    out = dict(methods=names, n=mat.shape[0])
    if len(names) >= 3:
        fr = stats.friedmanchisquare(*mat.T)
        out["friedman_p"] = float(fr.pvalue)
    ranks = stats.rankdata(mat, axis=1).mean(axis=0)
    out["mean_rank"] = dict(zip(names, map(float, ranks)))
    pairs = list(combinations(range(len(names)), 2))
    tests = [paired_wilcoxon(mat[:, i], mat[:, j]) for i, j in pairs]
    adj = holm([t["p"] for t in tests]) if tests else []
    out["pairs"] = [dict(a=names[i], b=names[j], p_holm=float(p), **t)
                    for (i, j), t, p in zip(pairs, tests, adj)]
    return out
