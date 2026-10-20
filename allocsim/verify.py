"""Независимые эталоны для проверки точного решения на малых примерах.

- brute_force — полный перебор назначений и порядков обхода (N ≤ 7);
- milp_solve  — смешанное целочисленное линейное программирование в HiGHS (scipy.optimize.milp),
  трёхиндексная модель с ограничениями Миллера — Таккера — Землина. Независимый решатель
  и независимая формулировка, чтобы сверить CP-SAT на примерах побольше.
"""
from itertools import permutations, product

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .exact import distance_matrices, nondominated


def _best_route(d, tasks):
    """Минимальная длина пути аппарата по задачам tasks (узлы 1…N в матрице d) перебором порядков."""
    if not tasks:
        return 0.0
    best = np.inf
    for perm in permutations(tasks):
        seq = (0,) + perm + (0,)
        best = min(best, sum(d[a, b] for a, b in zip(seq, seq[1:])))
    return best


def brute_force(inst):
    """Все сочетания (S, Λ) при оптимальных маршрутах каждого назначения: S*, Λ* и фронт."""
    D = distance_matrices(inst)
    can = inst.can_do()
    n, M = inst.n_tasks, inst.n_agents
    cache = {}
    pts = []
    for a in product(range(M), repeat=n):
        if not all(can[a[j], j] for j in range(n)):
            continue
        L = []
        for k in range(M):
            tasks = tuple(j + 1 for j in range(n) if a[j] == k)
            key = (k, tasks)
            if key not in cache:
                cache[key] = _best_route(D[k], tasks)
            L.append(cache[key])
        pts.append((sum(L), max(L)))
    S_star = min(p[0] for p in pts)
    L_star = min(p[1] for p in pts)
    return dict(S=S_star, Lam=L_star, front=nondominated(pts))


def milp_solve(inst, objective="S", time_limit=120):
    """S* или Λ* через HiGHS. Возвращает (значение, статус решателя)."""
    D = distance_matrices(inst)
    can = inst.can_do()
    n, M = inst.n_tasks, inst.n_agents
    idx = {}
    def var(key):
        if key not in idx:
            idx[key] = len(idx)
        return idx[key]
    for k in range(M):
        nodes = [0] + [j + 1 for j in range(n) if can[k, j]]
        for i in nodes:
            for j in nodes:
                if i != j:
                    var(("x", k, i, j))
        for j in range(n):
            if can[k, j]:
                var(("u", k, j + 1))
    lam = var(("lam",))
    nv = len(idx)
    c = np.zeros(nv)
    integrality = np.zeros(nv)
    lb, ub = np.zeros(nv), np.full(nv, np.inf)
    for key, v in idx.items():
        if key[0] == "x":
            integrality[v], ub[v] = 1, 1
            if objective == "S":
                c[v] = D[key[1]][key[2], key[3]]
        elif key[0] == "u":
            lb[v], ub[v] = 1, n
    if objective == "Lambda":
        c[lam] = 1
    rows, lo, hi = [], [], []
    def add(coefs, l, h):
        rows.append(coefs); lo.append(l); hi.append(h)
    for j in range(1, n + 1):  # каждая задача: ровно один вход по всем аппаратам
        add({idx[("x", k, i, j)]: 1 for k in range(M) for i in range(n + 1)
             if ("x", k, i, j) in idx}, 1, 1)
    for k in range(M):
        nodes = [0] + [j + 1 for j in range(n) if can[k, j]]
        for j in nodes:  # вход = выход
            coefs = {}
            for i in nodes:
                if i != j:
                    coefs[idx[("x", k, i, j)]] = coefs.get(idx[("x", k, i, j)], 0) + 1
                    coefs[idx[("x", k, j, i)]] = coefs.get(idx[("x", k, j, i)], 0) - 1
            add(coefs, 0, 0)
        add({idx[("x", k, 0, j)]: 1 for j in nodes[1:]}, 0, 1)  # из старта не больше одного раза
        for i in nodes[1:]:  # Миллер — Таккер — Землин
            for j in nodes[1:]:
                if i != j:
                    add({idx[("u", k, i)]: 1, idx[("u", k, j)]: -1, idx[("x", k, i, j)]: n},
                        -np.inf, n - 1)
        lam_row = {idx[("x", k, i, j)]: D[k][i, j] for i in nodes for j in nodes if i != j}
        lam_row[lam] = -1
        add(lam_row, -np.inf, 0)
    A = lil_matrix((len(rows), nv))
    for r, coefs in enumerate(rows):
        for v, a in coefs.items():
            A[r, v] = a
    res = milp(c=c, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=integrality,
               bounds=Bounds(lb, ub), options=dict(time_limit=time_limit, mip_rel_gap=1e-9))
    return (float(res.fun) if res.x is not None else None), res.status
