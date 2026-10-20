"""Поля задач: точки посещения на плоскости (квадрат) или в объёме (куб).

Два вида полей:
- равномерное — точки независимо и равномерно в поле;
- со скоплениями (процесс Томаса с фиксированным числом точек) — m центров равномерно в поле,
  каждая точка выбирает центр равновероятно и смещается от него на нормальную величину
  с разбросом σ по каждой оси; вылетевшие за поле точки отражаются внутрь.

Общие случайные числа по уровням скученности. Для одного номера повтора rep центры, выбор центра
и нормальные смещения z одинаковы на всех уровнях, меняется только σ: точки = центр + σ·z.
Поэтому уровни одного повтора — это одна и та же конфигурация, «сжатая» сильнее или слабее,
и разница между уровнями не тонет в разбросе случайных полей.
"""
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from .seeds import rng as make_rng

# Уровни скученности основного эксперимента. None — равномерное поле.
# 'abs' (первоначальный вариант): σ как доля стороны поля.
LEVELS = {
    "L0": None,
    "L1": 0.16,
    "L2": 0.08,
    "L3": 0.04,
    "L4": 0.02,
}
# 'rel' (предложение по итогам калибровки 10.10.2026): σ как доля характерного расстояния
# между центрами скоплений side / m^(1/dim). При M = 4 на плоскости совпадает с 'abs'.
LEVELS_REL = {
    "L0": None,
    "L1": 0.32,
    "L2": 0.16,
    "L3": 0.08,
    "L4": 0.04,
}


@dataclass(frozen=True)
class FieldSpec:
    """Описание поля задач.

    dim        — 2 (квадрат) или 3 (куб);
    side       — сторона поля;
    n_tasks    — число точек N;
    sigma_frac — разброс точек вокруг центра; None — равномерное поле;
    n_centers  — число центров скоплений m (для равномерного поля не используется);
    sigma_mode — 'abs': σ = sigma_frac·side (первоначальный вариант);
                 'rel': σ = sigma_frac·side / m^(1/dim), то есть доля от характерного
                 расстояния между центрами (вариант для сравнения на калибровке).
    """
    dim: int = 2
    side: float = 1000.0
    n_tasks: int = 24
    sigma_frac: Optional[float] = None
    n_centers: Optional[int] = None
    sigma_mode: str = "abs"

    def __post_init__(self):
        if self.dim not in (2, 3):
            raise ValueError("dim должен быть 2 или 3")
        if self.side <= 0 or self.n_tasks < 2:
            raise ValueError("нужно side > 0 и n_tasks ≥ 2")
        if self.sigma_frac is not None:
            if self.sigma_frac <= 0:
                raise ValueError("sigma_frac должен быть > 0")
            if not self.n_centers or self.n_centers < 1:
                raise ValueError("для поля со скоплениями нужен n_centers ≥ 1")
            if self.sigma_mode not in ("abs", "rel"):
                raise ValueError("sigma_mode: 'abs' или 'rel'")

    @property
    def is_uniform(self) -> bool:
        return self.sigma_frac is None

    @property
    def sigma(self) -> Optional[float]:
        if self.is_uniform:
            return None
        if self.sigma_mode == "abs":
            return self.sigma_frac * self.side
        return self.sigma_frac * self.side / self.n_centers ** (1.0 / self.dim)

    def to_dict(self):
        return asdict(self)


def reflect_into_box(x, side):
    """Отражение координат внутрь отрезка [0, side] (зеркально от границ, сколько угодно раз)."""
    y = np.mod(x, 2.0 * side)
    return np.where(y > side, 2.0 * side - y, y)


def sample_field(spec: FieldSpec, rep: int, base_seed: int = 0):
    """Точки поля и номер скопления каждой точки (−1 для равномерного поля).

    Номер скопления — служебная информация для проверок и картинок. Методам распределения
    он не передаётся: они видят только координаты.
    """
    n, d, a = spec.n_tasks, spec.dim, spec.side
    if spec.is_uniform:
        r = make_rng("uniform", base_seed, d, a, n, rep)
        pts = r.uniform(0.0, a, size=(n, d))
        return pts, np.full(n, -1, dtype=int)

    # Сид не зависит от σ — общие случайные числа по уровням скученности.
    r = make_rng("thomas", base_seed, d, a, n, spec.n_centers, rep)
    centers = r.uniform(0.0, a, size=(spec.n_centers, d))
    assign = r.integers(0, spec.n_centers, size=n)
    z = r.standard_normal(size=(n, d))
    pts = reflect_into_box(centers[assign] + spec.sigma * z, a)
    return pts, assign.astype(int)
