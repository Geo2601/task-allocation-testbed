"""Показатель скученности R (статья, раздел 4).

R = (среднее расстояние от точки до ближайшей соседней) /
    (то же среднее для равномерного поля той же формы и с тем же числом точек).

Знаменатель — не формула, а моделирование: для каждой размерности и каждого N заранее
посчитано среднее по 1000 равномерным полям единичного размера (файл data/uniform_nn_reference.json,
скрипт experiments/build_uniform_reference.py). Расстояния масштабируются линейно со стороной
поля, поэтому одной таблицы хватает для любого размера квадрата или куба. Так показатель
одинаково честен для плоскости и куба и сам учитывает края поля.

R ≈ 1 — точки разбросаны как случайно-равномерные; R < 1 — скопления; R > 1 — точки
расставлены регулярнее случайного.

Для справки — классические формулы без учёта краёв:
плоскость 0,5·√(S/N) (Clark, Evans, 1954), объём 0,554·(V/N)^(1/3);
с поправкой на край квадрата — формула Donnelly (1978).
"""
import json
import math
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REFERENCE_PATH = Path(__file__).with_name("data") / "uniform_nn_reference.json"


def mean_nn_distance(points) -> float:
    """Среднее расстояние от точки до ближайшей соседней."""
    pts = np.asarray(points, float)
    if len(pts) < 2:
        raise ValueError("нужно хотя бы две точки")
    dist, _ = cKDTree(pts).query(pts, k=2)
    return float(dist[:, 1].mean())


def _batch_mean_nn(batch):
    """Средние расстояния до ближайшего соседа для пачки полей формы (reps, N, d)."""
    diff = batch[:, :, None, :] - batch[:, None, :, :]
    d2 = np.einsum("rijk,rijk->rij", diff, diff)
    n = batch.shape[1]
    d2[:, np.arange(n), np.arange(n)] = np.inf
    return np.sqrt(d2.min(axis=2)).mean(axis=1)


def simulate_uniform_reference(n: int, dim: int, reps: int = 1000, seed: int = 0, chunk: int = 100):
    """Среднее и стандартное отклонение среднего расстояния до ближайшего соседа
    для N равномерных точек в единичном квадрате (dim=2) или кубе (dim=3)."""
    r = np.random.default_rng([seed, n, dim])
    vals = []
    done = 0
    while done < reps:
        k = min(chunk, reps - done)
        vals.append(_batch_mean_nn(r.random((k, n, dim))))
        done += k
    v = np.concatenate(vals)
    return float(v.mean()), float(v.std(ddof=1))


@lru_cache(maxsize=1)
def _load_reference():
    if not REFERENCE_PATH.exists():
        return {}
    data = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    table = {}
    for row in data["table"]:
        table[(row["dim"], row["n"])] = (row["mean"], row["sd"])
    return table


_extra_cache = {}


def uniform_reference(n: int, dim: int):
    """(среднее, ст. отклонение) для единичного поля. Если N нет в таблице — считается
    на месте (1000 повторов) и запоминается до конца сеанса, с предупреждением."""
    table = _load_reference()
    if (dim, n) in table:
        return table[(dim, n)]
    if (dim, n) not in _extra_cache:
        warnings.warn(f"N={n}, dim={dim} нет в таблице нормировки — считаю на месте", stacklevel=2)
        _extra_cache[(dim, n)] = simulate_uniform_reference(n, dim)
    return _extra_cache[(dim, n)]


def clustering_index(points, side: float) -> float:
    """Показатель скученности R для поля со стороной side."""
    pts = np.asarray(points, float)
    n, dim = pts.shape
    ref_mean, _ = uniform_reference(n, dim)
    return mean_nn_distance(pts) / (side * ref_mean)


def clark_evans_expected(n: int, side: float, dim: int) -> float:
    """Ожидаемое расстояние до ближайшего соседа без учёта краёв (бесконечное поле)."""
    if dim == 2:
        return 0.5 * math.sqrt(side ** 2 / n)
    c3 = math.gamma(4.0 / 3.0) * (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)  # ≈ 0,554
    return c3 * (side ** 3 / n) ** (1.0 / 3.0)


def donnelly_expected(n: int, side: float) -> float:
    """То же для квадрата с поправкой на край (Donnelly, 1978)."""
    area, perimeter = side ** 2, 4.0 * side
    return 0.5 * math.sqrt(area / n) + (0.0514 + 0.041 / math.sqrt(n)) * perimeter / n
