"""Общая часть методов распределения: результат и подсчёт показателей."""
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..exact import route_lengths
from ..routing import reroute


@dataclass
class MethodResult:
    name: str
    assignment: np.ndarray                     # (N,) номер аппарата для каждой задачи
    routes: List[List[int]]                    # маршруты после единого решателя
    lengths: np.ndarray                        # длины путей после единого решателя
    raw_routes: Optional[List[List[int]]] = None   # маршруты «как выдал метод» (если он их строит)
    raw_lengths: Optional[np.ndarray] = None
    rounds: int = 0                            # раунды согласования (для децентрализованных)
    messages: int = 0                          # переданные сообщения (при полносвязной сети)
    wall: float = 0.0                          # время работы метода без единого решателя
    info: dict = field(default_factory=dict)

    @property
    def S(self) -> float:
        return float(self.lengths.sum())

    @property
    def Lam(self) -> float:
        return float(self.lengths.max())

    @property
    def raw_S(self) -> Optional[float]:
        return None if self.raw_lengths is None else float(self.raw_lengths.sum())

    @property
    def raw_Lam(self) -> Optional[float]:
        return None if self.raw_lengths is None else float(self.raw_lengths.max())

    @property
    def idle(self) -> int:
        return int(sum(1 for r in self.routes if not r))


def finalize(inst, name, assignment, raw_routes=None, rounds=0, messages=0, wall=0.0, info=None):
    assignment = np.asarray(assignment, int)
    check_assignment(inst, assignment)
    routes, lengths = reroute(inst, assignment)
    raw_lengths = None
    if raw_routes is not None:
        raw_lengths = route_lengths(inst, raw_routes)
    return MethodResult(name=name, assignment=assignment, routes=routes, lengths=lengths,
                        raw_routes=raw_routes, raw_lengths=raw_lengths, rounds=rounds,
                        messages=messages, wall=wall, info=info or {})


def check_assignment(inst, assignment):
    """Каждая задача у одного аппарата, умеющего её тип."""
    assignment = np.asarray(assignment)
    if assignment.shape != (inst.n_tasks,) or (assignment < 0).any() or \
            (assignment >= inst.n_agents).any():
        raise ValueError("назначение неполное или с неверными номерами аппаратов")
    if not inst.can_do()[assignment, np.arange(inst.n_tasks)].all():
        raise ValueError("задача назначена аппарату, который не умеет её тип")


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.wall = time.perf_counter() - self.t0
