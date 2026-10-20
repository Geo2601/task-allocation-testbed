"""План эксперимента (статья, раздел 4): перечень ячеек основного эксперимента и проверок устойчивости."""
from itertools import product

from .fields import FieldSpec, LEVELS
from .instance import InstanceSpec, main_field

SIDE = 1000.0
AGENTS = (4, 8, 16)
TASKS_PER_AGENT = (3, 6, 12)
START_MODES = ("base", "scattered")
HETERO = (False, True)
# Способ задавать σ. Принято 11.10.2026: 'rel' — доля характерного расстояния между центрами
# скоплений side / m^(1/dim); уровни одинаково скучены при любом числе аппаратов (калибровка).
# 'abs' (доля стороны поля, первоначальный вариант) оставлен для сравнения.
SIGMA_MODE = "rel"


def main_plan(levels=tuple(LEVELS)):
    """Основной эксперимент: плоскость, база в углу. 5 × 3 × 3 × 2 × 2 = 180 ячеек."""
    cells = []
    for lvl, m, rho, sm, het in product(levels, AGENTS, TASKS_PER_AGENT, START_MODES, HETERO):
        cells.append(dict(
            level=lvl, M=m, rho=rho, start=sm, hetero=het,
            spec=InstanceSpec(field=main_field(2, SIDE, m, rho, lvl, sigma_mode=SIGMA_MODE), n_agents=m,
                              start_mode=sm, base_pos="corner", heterogeneous=het)))
    return cells


def volume_check(levels=tuple(LEVELS)):
    """Проверка в объёме: куб со стороной SIDE, M = 8, ρ = 6. 5 × 2 × 2 = 20 ячеек."""
    return [dict(level=lvl, M=8, rho=6, start=sm, hetero=het,
                 spec=InstanceSpec(field=main_field(3, SIDE, 8, 6, lvl, sigma_mode=SIGMA_MODE), n_agents=8,
                                   start_mode=sm, heterogeneous=het))
            for lvl, sm, het in product(levels, START_MODES, HETERO)]


def center_base_check(levels=tuple(LEVELS)):
    """База в центре: плоскость, M = 8, ρ = 6, режим «база». 5 × 2 = 10 ячеек."""
    return [dict(level=lvl, M=8, rho=6, start="base", hetero=het,
                 spec=InstanceSpec(field=main_field(2, SIDE, 8, 6, lvl, sigma_mode=SIGMA_MODE), n_agents=8,
                                   start_mode="base", base_pos="center", heterogeneous=het))
            for lvl, het in product(levels, HETERO)]


CLUSTERED = ("L1", "L2", "L3", "L4")     # уровни со скоплениями (L0 — равномерное поле)


def _clusters_check(centers_per_agent, levels=CLUSTERED):
    return [dict(level=lvl, M=8, rho=6, start=sm, hetero=het,
                 spec=InstanceSpec(field=main_field(2, SIDE, 8, 6, lvl, centers_per_agent=centers_per_agent,
                                                    sigma_mode=SIGMA_MODE),
                                   n_agents=8, start_mode=sm, heterogeneous=het))
            for lvl, sm, het in product(levels, START_MODES, HETERO)]


def clusters_half_check(levels=CLUSTERED):
    """Число скоплений m = M/2: плоскость, M = 8, ρ = 6, уровни L1–L4 × старт × типы = 16 ячеек.
    Скоплений меньше, чем аппаратов: кластеризации по числу аппаратов приходится делить скопления."""
    return _clusters_check(0.5, levels)


def clusters_double_check(levels=CLUSTERED):
    """Число скоплений m = 2M: то же, 16 ячеек. Скоплений больше, чем аппаратов."""
    return _clusters_check(2.0, levels)


def purity_check(levels=CLUSTERED, purity=1.0):
    """Проверка Г4: разнотипная группа, в каждом скоплении задачи одного типа (type_purity = 1).
    M = 8, ρ = 6, уровни L1–L4 × старт = 8 ячеек. Точки те же, что в основном эксперименте при тех же
    M, ρ, уровне и повторе; сравнение — с разнотипными ячейками основного эксперимента, где тип задачи
    от положения не зависит."""
    return [dict(level=lvl, M=8, rho=6, start=sm, hetero=True,
                 spec=InstanceSpec(field=main_field(2, SIDE, 8, 6, lvl, sigma_mode=SIGMA_MODE), n_agents=8,
                                   start_mode=sm, heterogeneous=True, type_purity=purity))
            for lvl, sm in product(levels, START_MODES)]


def skfu_check():
    """Сверка с СКФУ: куб 100×100×100, 100 равномерных точек, 5/10/15/20 аппаратов, одна база в углу.
    Разнотипность — как в коде симуляции авторов (github.com/BenJoice/Task-Allocation): 5 типов,
    у задачи один тип, у аппарата 1–3 типа, каждый тип умеют не меньше двух аппаратов.
    Расчёт — experiments/skfu_check.py."""
    return [dict(M=m, spec=InstanceSpec(field=FieldSpec(dim=3, side=100.0, n_tasks=100),
                                        n_agents=m, start_mode="base", base_pos="corner",
                                        heterogeneous=True, n_types=5, max_types_per_agent=3,
                                        caps_scheme="random", min_cover=2))
            for m in (5, 10, 15, 20)]
