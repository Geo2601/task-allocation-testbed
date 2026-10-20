"""Жадный алгоритм (нижняя планка): освободившийся аппарат берёт ближайшую свободную задачу из
тех, что умеет. Освобождается раньше тот, у кого меньше пройденный путь; при равенстве — меньший
номер. Аппарат, которому нечего взять, заканчивает работу.

Вариант order='round_robin' — GrA в работах СКФУ (JKSUCI, 2023, раздел 1.3.1): аппараты по очереди
(по номеру) берут ближайшую к себе свободную задачу, пока задачи есть; каждый получает поровну
с точностью до одной задачи."""
import numpy as np

from .base import Timer, finalize


def greedy(inst, order="least_path"):
    n, M = inst.n_tasks, inst.n_agents
    can = inst.can_do()
    with Timer() as t:
        pos = inst.starts.copy()
        dist_done = np.zeros(M)
        free = np.ones(n, bool)
        active = can.any(axis=1)
        routes = [[] for _ in range(M)]
        assignment = np.full(n, -1)
        turn = 0
        while free.any():
            cand = np.flatnonzero(active)
            if order == "round_robin":
                k = int(cand[np.searchsorted(cand, turn % M) % len(cand)])
                turn = k + 1
            else:
                k = int(cand[np.lexsort((cand, dist_done[cand]))[0]])
            mine = np.flatnonzero(free & can[k])
            if len(mine) == 0:
                active[k] = False
                continue
            d = np.linalg.norm(inst.points[mine] - pos[k], axis=1)
            j = int(mine[np.lexsort((mine, d))[0]])
            assignment[j] = k
            routes[k].append(j)
            dist_done[k] += float(np.linalg.norm(inst.points[j] - pos[k]))
            pos[k] = inst.points[j]
            free[j] = False
    name = "greedy" if order == "least_path" else f"greedy_{order}"
    return finalize(inst, name, assignment, raw_routes=routes, wall=t.wall)
