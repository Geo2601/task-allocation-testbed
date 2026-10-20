"""Пример задачи распределения: поле задач, типы задач, умения аппаратов, старты.

Какие случайные составляющие от чего зависят (парный план):
- координаты точек — только от поля (форма, N, уровень скученности, m) и номера повтора;
  одинаковы для однородной и разнотипной группы и для обоих режимов старта;
- типы задач — от поля и номера повтора;
- умения аппаратов и рассредоточенные старты — от числа аппаратов, формы поля и номера повтора;
  одинаковы на всех уровнях скученности.
Благодаря этому сравнения «однородная — разнотипная», «база — рассредоточенный старт» и
«уровень k — уровень k+1» идут на одних и тех же случайных числах.
"""
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np

from .clustering import clustering_index
from .fields import FieldSpec, LEVELS, LEVELS_REL, sample_field
from .seeds import rng as make_rng


@dataclass(frozen=True)
class InstanceSpec:
    """Описание примера.

    n_agents     — число аппаратов M;
    start_mode   — 'base': все взлетают из одной точки и возвращаются в неё;
                   'scattered': случайные старты, без возврата;
    base_pos     — 'corner' (угол поля, начало координат) или 'center';
    heterogeneous — False: один тип задач, все умеют всё;
                   True: n_types типов, каждый аппарат умеет 1 или 2 типа,
                   каждый тип умеют не меньше min_cover аппаратов;
    type_purity  — доля задач, получающих «свой» тип скопления (0 — тип не зависит от
                   положения, как в основном эксперименте; > 0 — для проверки гипотезы Г4);
    caps_scheme  — 'balanced': ровно половина аппаратов (с округлением вниз) умеет
                   max_types_per_agent типов, остальные — один; средняя «универсальность»
                   одинакова при любом M. 'random': число умений у каждого аппарата
                   равновероятно 1…max; при малых M отбраковка делает группу универсальнее.
    """
    field: FieldSpec
    n_agents: int
    start_mode: str = "base"
    base_pos: str = "corner"
    heterogeneous: bool = False
    n_types: int = 3
    min_cover: int = 2
    max_types_per_agent: int = 2
    type_purity: float = 0.0
    caps_scheme: str = "balanced"

    def __post_init__(self):
        if self.n_agents < 1:
            raise ValueError("n_agents ≥ 1")
        if self.start_mode not in ("base", "scattered"):
            raise ValueError("start_mode: 'base' или 'scattered'")
        if self.base_pos not in ("corner", "center"):
            raise ValueError("base_pos: 'corner' или 'center'")
        if self.heterogeneous:
            if self.n_types < 2:
                raise ValueError("для разнотипной группы нужно n_types ≥ 2")
            if self.n_agents * self.max_types_per_agent < self.n_types * self.min_cover:
                raise ValueError("умений не хватает, чтобы каждый тип умели min_cover аппаратов")
            if not 0.0 <= self.type_purity <= 1.0:
                raise ValueError("type_purity в [0, 1]")
            if self.caps_scheme not in ("balanced", "random"):
                raise ValueError("caps_scheme: 'balanced' или 'random'")
            if self.caps_scheme == "balanced":
                n_multi = self.n_agents // 2
                total = n_multi * self.max_types_per_agent + (self.n_agents - n_multi)
                if total < self.n_types * self.min_cover:
                    raise ValueError("при caps_scheme='balanced' умений не хватает для min_cover")

    @property
    def effective_n_types(self) -> int:
        return self.n_types if self.heterogeneous else 1

    def to_dict(self):
        d = asdict(self)
        d["field"] = self.field.to_dict()
        return d


@dataclass
class Instance:
    spec: InstanceSpec
    rep: int
    base_seed: int
    points: np.ndarray          # (N, dim) координаты задач
    task_type: np.ndarray       # (N,) тип задачи, 0..T−1
    capabilities: np.ndarray    # (M, T) bool — аппарат k умеет тип t
    starts: np.ndarray          # (M, dim) стартовые точки
    returns: bool               # возвращаются ли аппараты в старт
    cluster_id: np.ndarray      # (N,) номер скопления, −1 у равномерного поля (служебное)
    R: float                    # показатель скученности
    meta: dict = field(default_factory=dict)

    @property
    def n_tasks(self) -> int:
        return len(self.points)

    @property
    def n_agents(self) -> int:
        return len(self.starts)

    def can_do(self) -> np.ndarray:
        """(M, N) bool — может ли аппарат k выполнить задачу j."""
        return self.capabilities[:, self.task_type]

    def to_dict(self) -> dict:
        return dict(
            spec=self.spec.to_dict(), rep=self.rep, base_seed=self.base_seed,
            points=self.points.tolist(), task_type=self.task_type.tolist(),
            capabilities=self.capabilities.astype(int).tolist(), starts=self.starts.tolist(),
            returns=self.returns, cluster_id=self.cluster_id.tolist(), R=self.R, meta=self.meta,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "Instance":
        sd = dict(d["spec"])
        sd["field"] = FieldSpec(**sd["field"])
        return cls(
            spec=InstanceSpec(**sd), rep=d["rep"], base_seed=d["base_seed"],
            points=np.asarray(d["points"], float), task_type=np.asarray(d["task_type"], int),
            capabilities=np.asarray(d["capabilities"], bool), starts=np.asarray(d["starts"], float),
            returns=bool(d["returns"]), cluster_id=np.asarray(d["cluster_id"], int),
            R=float(d["R"]), meta=d.get("meta", {}),
        )

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Instance":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def sample_capabilities(n_agents, n_types, min_cover, max_per_agent, r, scheme="balanced",
                        max_tries=100_000):
    """Умения аппаратов. Число умений у аппарата: при scheme='balanced' — у половины аппаратов
    max_per_agent, у остальных 1 (кому сколько — случайно); при 'random' — равновероятно
    от 1 до max_per_agent. Сами типы — случайно без повторов. Принимается набор, где каждый
    тип умеют не меньше min_cover аппаратов (выбор с отбраковкой)."""
    for attempt in range(1, max_tries + 1):
        if scheme == "balanced":
            n_multi = n_agents // 2
            counts = r.permutation(np.r_[np.full(n_multi, max_per_agent),
                                         np.ones(n_agents - n_multi, int)])
        else:
            counts = r.integers(1, max_per_agent + 1, size=n_agents)
        caps = np.zeros((n_agents, n_types), bool)
        for k in range(n_agents):
            caps[k, r.choice(n_types, size=counts[k], replace=False)] = True
        if caps.sum(axis=0).min() >= min_cover:
            return caps, attempt
    raise RuntimeError("не удалось подобрать умения аппаратов")


def make_instance(spec: InstanceSpec, rep: int, base_seed: int = 0) -> Instance:
    f = spec.field
    pts, cid = sample_field(f, rep, base_seed)
    n, m, d = f.n_tasks, spec.n_agents, f.dim
    field_key = (d, f.side, n, f.sigma_frac, f.n_centers, f.sigma_mode)
    meta = {}

    # Типы задач и умения.
    if spec.heterogeneous:
        t_rng = make_rng("types", base_seed, *field_key, spec.n_types, spec.type_purity, rep)
        types = t_rng.integers(0, spec.n_types, size=n)
        if spec.type_purity > 0 and not f.is_uniform:
            dominant = t_rng.integers(0, spec.n_types, size=f.n_centers)
            keep = t_rng.random(n) < spec.type_purity
            types = np.where(keep, dominant[cid], types)
        c_rng = make_rng("caps", base_seed, d, f.side, m, spec.n_types, spec.min_cover,
                         spec.max_types_per_agent, spec.caps_scheme, rep)
        caps, attempts = sample_capabilities(m, spec.n_types, spec.min_cover,
                                             spec.max_types_per_agent, c_rng, spec.caps_scheme)
        meta["caps_attempts"] = attempts
    else:
        types = np.zeros(n, int)
        caps = np.ones((m, 1), bool)

    # Старты.
    if spec.start_mode == "base":
        base = np.zeros(d) if spec.base_pos == "corner" else np.full(d, f.side / 2.0)
        starts = np.tile(base, (m, 1))
        returns = True
    else:
        s_rng = make_rng("starts", base_seed, d, f.side, m, rep)
        starts = s_rng.uniform(0.0, f.side, size=(m, d))
        returns = False

    return Instance(spec=spec, rep=rep, base_seed=base_seed, points=pts, task_type=types,
                    capabilities=caps, starts=starts, returns=returns, cluster_id=cid,
                    R=clustering_index(pts, f.side), meta=meta)


def main_field(dim: int, side: float, n_agents: int, tasks_per_agent: int, level: str,
               centers_per_agent: float = 1.0, sigma_mode: str = "rel") -> FieldSpec:
    """Поле основного эксперимента: N = M·ρ, m = M (или M·centers_per_agent), σ по уровню
    из LEVELS (sigma_mode='abs') или LEVELS_REL (sigma_mode='rel')."""
    frac = (LEVELS if sigma_mode == "abs" else LEVELS_REL)[level]
    m = max(1, int(round(n_agents * centers_per_agent)))
    return FieldSpec(dim=dim, side=side, n_tasks=n_agents * tasks_per_agent,
                     sigma_frac=frac, n_centers=None if frac is None else m, sigma_mode=sigma_mode)
