"""Метод СКФУ — распределение задач с делением поля на кластеры (DFCM; Петренко, Тебуева,
Антонов, Сакольчик и др.: JKSUCI, 2023, 35(6), 101580; РТК, 2023, т. 11, № 2, с. 99–109).
Реализация по тексту JKSUCI (раздел 2.2, шаги 1–5); где текст неоднозначен — по коду
симуляции авторов (github.com/BenJoice/Task-Allocation, Source.cpp, 01.08.2022).

Шаг 1. Поле приводится к наименьшему кубу (на плоскости — квадрату), в который входят все задачи
и база, и делится на M частей равного объёма. Как делится — по коду авторов: плоскостями через
вертикальное ребро куба, проходящее через базу; плоскости пересекают дальние грани в точках,
равномерно расставленных по их периметру, — получаются клинья («сектора») равного объёма.
На плоскости — сектора из угла с базой. При базе внутри поля (проверка «база в центре») —
сектора равной площади вокруг базы (то же правило: треугольники с вершиной в базе и основаниями
на сторонах поля). Вариант partition='strips' — равные слои вдоль последней оси (второе прочтение
фразы «вдоль ребра» в тексте статьи; оставлен для сравнения).

Шаг 2. Оценка аппарата i для кластера j (формула 12): p_ij = Σ_c [i умеет c] · n_jc / n_c, где n_jc —
число задач типа c в кластере j, n_c — всего задач типа c.

Шаг 3. Кластеры раздаются аппаратам по одному так, чтобы сумма p_ij была наибольшей (в статье —
перебор перестановок, здесь — венгерский метод, результат тот же). При равенстве — меньший
суммарный путь от старта до центров масс кластеров (формулы 9–10).

Шаг 4. Аппарат берёт задачи своего кластера, которые умеет. Оставшиеся задачи каждого типа
делятся между умеющими его аппаратами (формула 15): аппараты упорядочены по длине
предварительного маршрута R_i; доли пропорциональны R_i и раздаются в обратном порядке —
самая большая доля аппарату с самым коротким маршрутом (так в коде авторов; в статье — «для
усреднения пройденного пути»). Каждый добирает нужное число ближайших к своим точкам задач.

Шаг 5. Маршруты — единым решателем маршрутов стенда (у авторов — перебор до 10 задач, иначе
имитация отжига).

Рассредоточенный старт (у авторов база одна): деление поля — из угла поля (0, …, 0), как при базе
в углу; при равных p_ij кластеры раздаются по расстоянию от старта аппарата до центра масс.

Отличия кода авторов от статьи (для сведения; реализуется статья): кластеры раздаются жадно, а не
перебором; остаток задач не распределяется — итог пересчитывается по доле нераспределённых задач;
возврат на базу в длину пути не входит.
"""
import numpy as np

from ..assign import assign_min_sum
from ..routing import best_route
from .base import Timer, finalize


def _bounding_cube(points, apex):
    allp = np.vstack([points, apex[None]])
    lo = allp.min(axis=0)
    side = float((allp.max(axis=0) - lo).max())
    return lo, max(side, 1e-12)


def sector_labels(points, apex, lo, side, M):
    """Номер сектора равной площади (по проекции на плоскость xy) для каждой точки.
    Квадрат [lo, lo + side]², вершина секторов — apex (внутри квадрата или на его границе)."""
    x0, y0 = lo[0], lo[1]
    x1, y1 = x0 + side, y0 + side
    ax, ay = apex[0], apex[1]
    # стороны против часовой стрелки: низ, право, верх, лево; h — расстояние от вершины до стороны
    h = np.array([ay - y0, x1 - ax, y1 - ay, ax - x0])
    cum = np.concatenate([[0.0], np.cumsum(0.5 * h * side)])
    total = cum[-1]
    labels = np.zeros(len(points), int)
    for i, p in enumerate(points):
        dx, dy = p[0] - ax, p[1] - ay
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            continue
        tx = (x1 - ax) / dx if dx > 0 else ((x0 - ax) / dx if dx < 0 else np.inf)
        ty = (y1 - ay) / dy if dy > 0 else ((y0 - ay) / dy if dy < 0 else np.inf)
        if tx <= ty:
            qy = ay + tx * dy
            s, u = (1, qy - y0) if dx > 0 else (3, y1 - qy)
        else:
            qx = ax + ty * dx
            s, u = (2, x1 - qx) if dy > 0 else (0, qx - x0)
        A = cum[s] + 0.5 * h[s] * u
        labels[i] = min(M - 1, int(np.floor(A / (total / M) + 1e-12)))
    return labels


def strip_labels(points, lo, side, M):
    """Равные слои вдоль последней оси."""
    z = (points[:, -1] - lo[-1]) / side
    return np.minimum(M - 1, np.floor(z * M + 1e-12).astype(int))


def _split_counts(total, weights):
    """Целые доли total, пропорциональные weights (наибольшие остатки)."""
    w = np.asarray(weights, float)
    w = np.full(len(w), 1.0 / len(w)) if w.sum() <= 0 else w / w.sum()
    raw = total * w
    cnt = np.floor(raw).astype(int)
    rest = total - cnt.sum()
    for i in np.argsort(-(raw - cnt), kind="stable")[:rest]:
        cnt[i] += 1
    return cnt


def skfu(inst, partition="sectors"):
    n, M, dim = inst.n_tasks, inst.n_agents, inst.points.shape[1]
    can = inst.can_do()
    caps = inst.capabilities
    types = inst.task_type
    P = inst.points
    with Timer() as t:
        single_base = bool(np.allclose(inst.starts, inst.starts[0]))
        apex = inst.starts[0].copy() if single_base else np.zeros(dim)
        lo, side = _bounding_cube(P, apex)
        if partition == "sectors":
            lab = sector_labels(P, apex, lo, side, M)
        elif partition == "strips":
            lab = strip_labels(P, lo, side, M)
        else:
            raise ValueError(partition)

        # шаг 2: оценки p_ij
        T = caps.shape[1]
        n_c = np.bincount(types, minlength=T).astype(float)
        n_jc = np.zeros((M, T))
        np.add.at(n_jc, (lab, types), 1.0)
        frac = np.divide(n_jc, n_c, out=np.zeros_like(n_jc), where=n_c > 0)
        p = caps.astype(float) @ frac.T                    # аппараты × кластеры

        # шаги 1 и 3: раздача кластеров (максимум Σ p, при равенстве — меньше путь до центров масс)
        cent = np.array([P[lab == j].mean(axis=0) if (lab == j).any() else apex for j in range(M)])
        dist = np.linalg.norm(cent[:, None, :] - inst.starts[None, :, :], axis=2)   # кластеры × аппараты
        diam = float(np.linalg.norm(np.ptp(np.vstack([P, inst.starts]), axis=0))) or 1.0
        cost = -1e6 * p.T + dist / diam
        owner = np.asarray(assign_min_sum(cost))          # кластер j → аппарат

        # шаг 4, первый круг: свои задачи своего кластера
        assignment = np.full(n, -1)
        for j in range(M):
            k = owner[j]
            mine = np.flatnonzero((lab == j) & can[k])
            assignment[mine] = k

        # шаг 4, второй круг: остаток по типам
        left = np.flatnonzero(assignment < 0)
        leftovers = int(len(left))
        if len(left):
            R = np.array([best_route(inst, k, np.flatnonzero(assignment == k))[1] for k in range(M)])
            for c in range(T):
                Lc = [int(j) for j in left if types[j] == c]
                if not Lc:
                    continue
                Ac = np.flatnonzero(caps[:, c])
                order = Ac[np.lexsort((Ac, R[Ac]))]          # по возрастанию R_i
                cnt = np.sort(_split_counts(len(Lc), R[order]))[::-1]
                for k, q in zip(order, cnt):
                    for _ in range(q):
                        own = np.vstack([inst.starts[k][None], P[assignment == k]])
                        d = np.min(np.linalg.norm(P[Lc][:, None, :] - own[None], axis=2), axis=1)
                        j = Lc.pop(int(np.argmin(d)))
                        assignment[j] = k
    name = "skfu" if partition == "sectors" else f"skfu_{partition}"
    return finalize(inst, name, assignment, wall=t.wall,
                    info=dict(partition=partition, leftovers=leftovers,
                              cluster_sizes=np.bincount(lab, minlength=M).tolist()))
