"""Лучшее известное решение для примеров, где точное решение не укладывается в лимит
(статья, раздел 2): маршрутный решатель OR-Tools с направленным локальным поиском
(guided local search) и фиксированным временем.

- objective='S'      — минимум суммарного пути (стоимость дуг); с T — при длине пути каждого
  аппарата не больше T (ёмкость измерения «путь»), это точки ε-ограничений для фронта;
- objective='Lambda' — минимум времени выполнения: стоимость разброса длин путей с большим весом
  (SetGlobalSpanCostCoefficient) плюс суммарный путь как второстепенный показатель;
- objective='W'      — минимум F_w = w·S/S° + (1 − w)·Λ/Λ°: стоимость дуг a·d, разброс длин
  с весом b (коэффициенты — exact.weight_coefs).

Нижнюю оценку маршрутный решатель не даёт — её берут из CP-SAT (exact.solve с тем же лимитом).
Решение отсюда используется и как подсказка CP-SAT.
"""
import time

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from .exact import (SCALE, WEIGHTS, ExactResult, distance_matrices, route_lengths, weight_coefs,
                    weighted_value)

SPAN_WEIGHT = 1000


def solve_routing(inst, objective="S", time_limit=60, scale=SCALE, seed_routes=None, T=None,
                  w=None, norm=None):
    n, M = inst.n_tasks, inst.n_agents
    can = inst.can_do()
    D = [np.rint(d * scale).astype(np.int64) for d in distance_matrices(inst)]
    # узлы: 0…N−1 — задачи, N+k — старт аппарата k, N+M+k — финиш аппарата k
    starts = [n + k for k in range(M)]
    ends = [n + M + k for k in range(M)]
    manager = pywrapcp.RoutingIndexManager(n + 2 * M, M, starts, ends)
    routing = pywrapcp.RoutingModel(manager)

    big = int(max(d.max() for d in D) * (n + 2)) + 1

    def matrix_for(k):
        """Матрица переходов аппарата k по индексам узлов модели (считается на стороне C++,
        без обратных вызовов Python — на порядки быстрее)."""
        Dk = D[k]
        size = n + 2 * M
        mat = np.full((size, size), big, dtype=np.int64)
        mat[:n, :n] = Dk[1:, 1:]
        mat[n + k, :n] = Dk[0, 1:]            # старт → задача
        mat[:n, n + M + k] = Dk[1:, 0]        # задача → финиш (возврат или 0)
        mat[n + k, n + M + k] = 0             # не вылетает
        np.fill_diagonal(mat, 0)
        return mat.tolist()                   # индексы — номера узлов (как требует OR-Tools)

    mats = [matrix_for(k) for k in range(M)]
    cbs = [routing.RegisterTransitMatrix(mt) for mt in mats]
    if objective == "W":
        a, b = weight_coefs(w, norm)
        cost_cbs = [routing.RegisterTransitMatrix((np.array(mt, dtype=np.int64) * a).tolist())
                    for mt in mats]
    else:
        cost_cbs = cbs
    for k in range(M):
        routing.SetArcCostEvaluatorOfVehicle(cost_cbs[k], k)
    ub = int(max(d.max() for d in D) * (n + 2))
    cap = ub if T is None else int(np.floor(T * scale))
    routing.AddDimensionWithVehicleTransits(cbs, 0, cap, True, "dist")
    if objective == "Lambda":
        routing.GetDimensionOrDie("dist").SetGlobalSpanCostCoefficient(SPAN_WEIGHT)
    elif objective == "W":
        routing.GetDimensionOrDie("dist").SetGlobalSpanCostCoefficient(b)
    for j in range(n):
        allowed = [int(k) for k in range(M) if can[k, j]]
        if len(allowed) < M:   # SetAllowedVehiclesForIndex в этой версии обёртки не принимает список
            routing.VehicleVar(manager.NodeToIndex(j)).SetValues(allowed)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromMilliseconds(int(time_limit * 1000))
    t0 = time.perf_counter()
    if seed_routes is not None:
        init = routing.ReadAssignmentFromRoutes([list(r) for r in seed_routes], True)
        sol = routing.SolveFromAssignmentWithParameters(init, params) if init else \
            routing.SolveWithParameters(params)
    else:
        sol = routing.SolveWithParameters(params)
    wall = time.perf_counter() - t0
    res = ExactResult(objective=f"routing_{objective}", status="FEASIBLE" if sol else "UNKNOWN",
                      wall=wall)
    if sol:
        routes = []
        for k in range(M):
            idx, r = routing.Start(k), []
            idx = sol.Value(routing.NextVar(idx))
            while not routing.IsEnd(idx):
                r.append(manager.IndexToNode(idx))
                idx = sol.Value(routing.NextVar(idx))
            routes.append(r)
        res.routes = routes
        res.lengths = route_lengths(inst, routes)
        res.S, res.Lam = float(res.lengths.sum()), float(res.lengths.max())
        if objective == "S":
            # самопроверка: стоимость решения в OR-Tools совпадает с длиной маршрутов
            res.info["objective_mismatch"] = abs(sol.ObjectiveValue() / scale - res.S)
        if objective == "W":
            res.info.update(w=w, norm=tuple(norm), F=weighted_value(res.S, res.Lam, w, norm))
    return res


def best_known_front(inst, n_inner=5, time_limit=60, weights=WEIGHTS):
    """Фронт из лучших известных решений: минимум Λ, минимум S, n_inner точек
    «минимум S при L_k ≤ T» и минимумы F_w маршрутным решателем. Нижних оценок не даёт."""
    from .exact import nondominated
    a = solve_routing(inst, "Lambda", time_limit)
    b = solve_routing(inst, "S", time_limit)
    sols = [a, b]
    if a.Lam is not None and b.Lam is not None and b.Lam > a.Lam * (1 + 1e-9):
        for T in np.linspace(a.Lam, b.Lam, n_inner + 2)[1:-1]:
            r = solve_routing(inst, "S", time_limit, seed_routes=a.routes, T=T)
            r.T = T
            sols.append(r)
    if a.Lam is not None and b.S is not None:
        norm = (b.S, a.Lam)
        for w in weights:
            cand = [r for r in sols if r.routes is not None]
            seed = min(cand, key=lambda r: weighted_value(r.S, r.Lam, w, norm)).routes
            sols.append(solve_routing(inst, "W", time_limit, seed_routes=seed, w=w, norm=norm))
    pts = [(r.S, r.Lam) for r in sols if r.S is not None]
    return nondominated(pts), sols
