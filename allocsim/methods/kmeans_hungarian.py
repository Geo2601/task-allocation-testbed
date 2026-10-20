"""«Сначала кластеризация, потом маршруты»: точки делятся k-means на кластеры, кластеры раздаются
аппаратам венгерским методом по расстоянию от старта аппарата до центра кластера
(статья, раздел 3; так же устроена первая стадия у Elango и др., 2011).

Однородная группа: M кластеров (не больше числа задач), каждому аппарату — один.
Разнотипная группа (наше приспособление): кластеризация внутри каждого типа отдельно; число кластеров
типа t = числу аппаратов, умеющих t (не больше числа задач типа t); кластеры типа t раздаются
только умеющим t аппаратам, по одному на аппарат. Аппарат с двумя типами получает по кластеру
каждого типа; эти назначения независимы.

k-means: начальные центры k-means++, 10 запусков, лучший по сумме квадратов; сид выводится из
примера, поэтому результат воспроизводим.
"""
import numpy as np

from ..assign import assign_min_sum
from ..seeds import rng as make_rng
from .base import Timer, finalize


def kmeans(x, k, r, n_init=10, max_iter=100):
    """Метки и центры; k-means++ и алгоритм Ллойда."""
    x = np.asarray(x, float)
    best = (np.inf, None, None)
    for _ in range(n_init):
        centers = [x[r.integers(len(x))]]
        for _ in range(1, k):
            d2 = np.min(((x[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
            p = d2 / d2.sum() if d2.sum() > 0 else np.full(len(x), 1 / len(x))
            centers.append(x[r.choice(len(x), p=p)])
        c = np.array(centers)
        for _ in range(max_iter):
            lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
            new = c.copy()
            for g in range(k):
                pts = x[lab == g]
                if len(pts):
                    new[g] = pts.mean(0)
                else:   # пустая группа — переносим центр в самую далёкую точку
                    far = np.argmax(((x - c[lab]) ** 2).sum(-1))
                    new[g] = x[far]
            if np.allclose(new, c):
                break
            c = new
        lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
        inertia = ((x - c[lab]) ** 2).sum()
        if inertia < best[0]:
            best = (inertia, lab, c)
    return best[1], best[2]


def kmeans_hungarian(inst):
    n, M = inst.n_tasks, inst.n_agents
    caps = inst.capabilities
    r = make_rng("kmeans", inst.base_seed, inst.spec.to_dict(), inst.rep)
    assignment = np.full(n, -1)
    with Timer() as t:
        for typ in range(caps.shape[1]):
            tasks = np.flatnonzero(inst.task_type == typ)
            agents = np.flatnonzero(caps[:, typ])
            if len(tasks) == 0:
                continue
            k = min(len(agents), len(tasks))
            lab, centers = kmeans(inst.points[tasks], k, r)
            cost = np.linalg.norm(centers[:, None, :] - inst.starts[agents][None], axis=2)
            who = assign_min_sum(cost)               # группа → аппарат (по одной на аппарат)
            assignment[tasks] = agents[who[lab]]
    return finalize(inst, "kmeans_hungarian", assignment, wall=t.wall)
