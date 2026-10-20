"""Единый решатель маршрутов (статья, раздел 3): после любого метода распределения маршрут
каждого аппарата по его задачам перестраивается одним и тем же способом. Так методы сравниваются
по распределению, а не по качеству построения маршрута.

- до EXACT_MAX задач — точно, динамическим программированием Хелда — Карпа;
- больше — детерминированный локальный поиск: ближайший сосед из нескольких начал, затем
  2-opt и or-opt до отсутствия улучшений. Детерминированность нужна, чтобы результат не зависел
  от скорости машины (сначала предполагался «OR-Tools с фиксированным временем» — заменено по этой
  причине; качество проверяется тестами против точного решения).

Путь аппарата начинается в его старте; в режиме «база» замыкается в старт, иначе открыт.
"""
import numpy as np

EXACT_MAX = 12


def _dist(inst, k, tasks):
    """Матрица (n+1)×(n+1): узел 0 — старт аппарата k, 1…n — задачи tasks; возврат 0 без возврата."""
    pts = np.vstack([inst.starts[k], inst.points[tasks]])
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    if not inst.returns:
        d[1:, 0] = 0.0
    return d


def _held_karp(d):
    n = d.shape[0] - 1
    full = 1 << n
    dp = np.full((full, n), np.inf)
    parent = np.full((full, n), -1, dtype=np.int64)
    for j in range(n):
        dp[1 << j, j] = d[0, j + 1]
    bits = [1 << j for j in range(n)]
    dd = d[1:, 1:]
    for mask in range(1, full):
        row = dp[mask]
        if not np.isfinite(row).any():
            continue
        free = [j for j in range(n) if not mask & bits[j]]
        if not free:
            continue
        # для каждой свободной j: лучший предшественник i в mask
        cand = row[:, None] + dd[:, free]
        best_i = np.argmin(cand, axis=0)
        best_v = cand[best_i, np.arange(len(free))]
        for idx, j in enumerate(free):
            nm = mask | bits[j]
            if best_v[idx] < dp[nm, j]:
                dp[nm, j] = best_v[idx]
                parent[nm, j] = best_i[idx]
    last = dp[full - 1] + d[1:, 0]
    j = int(np.argmin(last))
    cost = float(last[j])
    order, mask = [], full - 1
    while j >= 0:
        order.append(j)
        pj = parent[mask, j]
        mask ^= bits[j]
        j = int(pj)
    return order[::-1], cost


def _path_cost(d, order):
    seq = [0] + [j + 1 for j in order] + [0]
    return float(sum(d[a, b] for a, b in zip(seq, seq[1:])))


def _nearest_neighbor(d, first):
    n = d.shape[0] - 1
    order, left = [first], set(range(n)) - {first}
    while left:
        cur = order[-1] + 1
        nxt = min(left, key=lambda j: (d[cur, j + 1], j))
        order.append(nxt)
        left.remove(nxt)
    return order


def _two_opt(d, order):
    seq = [0] + [j + 1 for j in order] + [0]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(seq) - 2):
            for j in range(i + 1, len(seq) - 1):
                a, b, c, e = seq[i - 1], seq[i], seq[j], seq[j + 1]
                delta = d[a, c] + d[b, e] - d[a, b] - d[c, e]
                # разворот меняет направление дуг внутри — матрица симметрична, кроме возврата в старт
                if j + 1 == len(seq) - 1:
                    delta = (d[a, c] + d[b, 0]) - (d[a, b] + d[c, 0])
                if delta < -1e-9:
                    seq[i:j + 1] = seq[i:j + 1][::-1]
                    improved = True
    return [x - 1 for x in seq[1:-1]]


def _or_opt(d, order):
    """Перенос цепочки из 1–3 задач в другое место пути (в прямом или обратном порядке).
    Приращение длины считается за O(1); первое найденное улучшение принимается."""
    seq = [0] + [j + 1 for j in order] + [0]
    improved = True
    while improved:
        improved = False
        for s_len in (1, 2, 3):
            n_inner = len(seq) - 2
            for i in range(1, n_inner - s_len + 2):
                first, last = seq[i], seq[i + s_len - 1]
                prev, nxt = seq[i - 1], seq[i + s_len]
                gain = d[prev, first] + d[last, nxt] - d[prev, nxt]
                rest = seq[:i] + seq[i + s_len:]
                chunk = seq[i:i + s_len]
                best = (-1e-9, None, None)
                for p in range(1, len(rest)):
                    if p == i:
                        continue
                    a, b = rest[p - 1], rest[p]
                    fwd = d[a, first] + d[last, b] - d[a, b] - gain
                    if fwd < best[0]:
                        best = (fwd, p, False)
                    if s_len > 1:
                        rev = d[a, last] + d[first, b] - d[a, b] - gain
                        if rev < best[0]:
                            best = (rev, p, True)
                if best[1] is not None:
                    _, p, rev = best
                    ins = chunk[::-1] if rev else chunk
                    seq = rest[:p] + ins + rest[p:]
                    improved = True
                    break
            if improved:
                break
    return [x - 1 for x in seq[1:-1]]


def _local_search(d, n_starts=5):
    n = d.shape[0] - 1
    firsts = sorted(range(n), key=lambda j: (d[0, j + 1], j))[:n_starts]
    best, best_cost = None, np.inf
    for f in firsts:
        o = _nearest_neighbor(d, f)
        while True:
            c0 = _path_cost(d, o)
            o = _or_opt(d, _two_opt(d, o))
            if _path_cost(d, o) >= c0 - 1e-9:
                break
        c = _path_cost(d, o)
        if c < best_cost - 1e-9:
            best, best_cost = o, c
    return best, best_cost


def best_route(inst, k, tasks):
    """Порядок обхода задач tasks аппаратом k и длина пути."""
    tasks = list(tasks)
    if not tasks:
        return [], 0.0
    d = _dist(inst, k, tasks)
    if len(tasks) == 1:
        order, cost = [0], _path_cost(d, [0])
    elif len(tasks) <= EXACT_MAX:
        order, cost = _held_karp(d)
    else:
        order, cost = _local_search(d)
    return [tasks[i] for i in order], cost


def reroute(inst, assignment):
    """Маршруты всех аппаратов по назначению (массив «задача → аппарат»)."""
    routes, lengths = [], np.zeros(inst.n_agents)
    for k in range(inst.n_agents):
        r, c = best_route(inst, k, np.flatnonzero(np.asarray(assignment) == k))
        routes.append(r)
        lengths[k] = c
    return routes, lengths
