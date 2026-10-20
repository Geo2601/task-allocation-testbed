"""Точное решение (статья, раздел 2) на CP-SAT (OR-Tools).

Две независимые формулировки одной задачи (сверяются между собой, с полным перебором и с MILP):

1. «По аппаратам» (model='vehicle'). У каждого аппарата k свой граф: узел 0 — его старт,
   узлы 1…N — задачи. Ограничение AddCircuit на каждый аппарат: выбранные дуги образуют один цикл
   через старт или пустой цикл (аппарат не вылетает); петля на задаче — «задача не у k».

2. «Общий обход» (model='giant'). Один граф: M узлов-стартов и N узлов-задач, одно ограничение
   AddCircuit на все узлы. Обход идёт старт₁ → задачи аппарата 1 → старт₂ → … → старт_M → … →
   старт₁, порядок стартов зафиксирован (это сразу снимает перестановки аппаратов). Принадлежность
   задачи аппарату и накопленная длина пути передаются по дугам. Дуг в M раз меньше, чем в
   формулировке 1, поэтому она для больших групп.

Общее:
- каждая задача ровно у одного аппарата, и только у умеющего её тип;
- режим «база»: путь замыкается в старт, возврат стоит своё расстояние;
- рассредоточенный старт без возврата: возврат стоит 0, то есть цикл = открытый путь;
- L_k — длина пути аппарата k; S = Σ L_k; Λ = max L_k.

Расстояния переводятся в целые с шагом 1/SCALE (по умолчанию 0,01 условной единицы). Оптимальность
доказывается для округлённых расстояний; итоговые S и Λ пересчитываются по точным расстояниям
выбранных маршрутов. Ошибка округления не больше 0,005·(N + M) на суммарный путь.

Симметрия. Одинаковые аппараты (общий старт, одинаковые умения) можно переставлять без изменения
S и Λ. Внутри такой группы аппараты упорядочиваются по наименьшему номеру своей задачи
(symmetry='index'); запасной вариант — порядок длин L_a ≥ L_b (symmetry='length').

Усиление нижней оценки (допустимые неравенства, оптимум не отсекают):
- L_k ≥ (путь до задачи j и обратно)·[задача j у k] — аппарат с задачей j не пролетит меньше;
- при известной нижней оценке суммарного пути S_lb (например, из задачи на минимум S):
  M·Λ ≥ S_lb, так как S ≤ M·Λ. Без этого CP-SAT почти не сдвигает оценку для Λ.

Минимум S при общей базе с возвратом и одинаковых аппаратах сводится к обходу одним аппаратом
(single_tour_reducible): два замкнутых маршрута через базу склеиваются в один той же длины.
Так S* доказывается за секунды и при N = 48 (без сведения — не всегда и за 30 с на 2 потоках).

Взвешенный критерий (objective='W', принят 11.10.2026 для трёх предпочтений):
F_w = w·S/S° + (1 − w)·Λ/Λ°, где S° и Λ° — минимумы S и Λ (нормировка norm = (S°, Λ°)).
w = 1 — важен только путь, w = 0 — только время, w = 0,5 — поровну. В решатель идёт целая
функция a·S + b·Λ с a = round(K·w·Λ°/S°), b = round(K·(1 − w)), K = 10⁶; относительная
погрешность весов от округления не больше 10⁻⁴.
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from ortools.sat.python import cp_model

SCALE = 100
W_SCALE = 10 ** 6


def weighted_value(S, Lam, w, norm):
    """F_w = w·S/S° + (1 − w)·Λ/Λ°; norm = (S°, Λ°)."""
    return w * S / norm[0] + (1 - w) * Lam / norm[1]


def weight_coefs(w, norm):
    """Целые коэффициенты (a, b) для a·S + b·Λ ∝ F_w."""
    S0, L0 = norm
    return int(round(W_SCALE * w * L0 / S0)), int(round(W_SCALE * (1 - w)))


@dataclass
class ExactResult:
    objective: str                 # 'S', 'Lambda', 'S|L<=T', 'W', 'lex_Lambda', 'lex_S'
    status: str                    # OPTIMAL / FEASIBLE / INFEASIBLE / UNKNOWN
    S: Optional[float] = None
    Lam: Optional[float] = None
    lengths: Optional[np.ndarray] = None
    routes: Optional[List[List[int]]] = None   # номера задач в порядке обхода, по аппаратам
    bound: Optional[float] = None              # нижняя оценка решателя для оптимизируемого показателя
                                               # (для 'W' — в единицах F_w)
    gap: Optional[float] = None                # (значение − оценка) / значение
    wall: float = 0.0
    T: Optional[float] = None
    info: dict = field(default_factory=dict)

    @property
    def optimal(self) -> bool:
        return self.status == "OPTIMAL"


def distance_matrices(inst):
    """Для каждого аппарата матрица (N+1)×(N+1): узел 0 — старт, 1…N — задачи; возврат
    в старт стоит 0, если аппарат не возвращается."""
    P, S = inst.points, inst.starts
    dtt = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    mats = []
    for k in range(inst.n_agents):
        ds = np.linalg.norm(P - S[k], axis=1)
        d = np.zeros((inst.n_tasks + 1, inst.n_tasks + 1))
        d[1:, 1:] = dtt
        d[0, 1:] = ds
        d[1:, 0] = ds if inst.returns else 0.0
        mats.append(d)
    return mats


def route_lengths(inst, routes):
    """Точные длины путей по маршрутам (номера задач 0…N−1)."""
    out = np.zeros(inst.n_agents)
    for k, r in enumerate(routes):
        if not r:
            continue
        pts = np.vstack([inst.starts[k], inst.points[r]])
        if inst.returns:
            pts = np.vstack([pts, inst.starts[k]])
        out[k] = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    return out


def _identical_groups(inst):
    keys = {}
    for k in range(inst.n_agents):
        key = (tuple(np.round(inst.starts[k], 9)), tuple(inst.capabilities[k]))
        keys.setdefault(key, []).append(k)
    return [g for g in keys.values() if len(g) > 1]


class _BaseModel:
    """Общая часть: усиления, симметрия, S и Λ. Наследник задаёт self.L (длины) и self.y
    (литерал «задача j у аппарата k» или None, если аппарат не умеет её тип)."""

    def _init_common(self, inst, scale):
        self.inst = inst
        self.m = cp_model.CpModel()
        self.scale = scale
        self.D = [np.rint(d * scale).astype(np.int64) for d in distance_matrices(inst)]
        n = inst.n_tasks
        self.ub = int(sum(d.max() for d in self.D) * (n + 1))

    def _canonical(self, routes):
        """Маршруты подсказки, переставленные внутри групп одинаковых аппаратов так, чтобы
        выполнялось упорядочение против перестановок (иначе подсказка недопустима)."""
        routes = [list(r) for r in routes]
        for g in _identical_groups(self.inst):
            rs = [routes[k] for k in g]
            if self.symmetry == "index":
                rs.sort(key=lambda r: (len(r) == 0, min(r) if r else 0))
            elif self.symmetry == "length":
                rs.sort(key=lambda r: -self._scaled_len(g[0], r))
            for k, r in zip(g, rs):
                routes[k] = r
        return routes

    def _scaled_len(self, k, r):
        if not r:
            return 0
        seq = [0] + [j + 1 for j in r] + [0]
        return int(sum(self.D[k][a, b] for a, b in zip(seq, seq[1:])))

    def _hint_common(self, routes):
        for k, r in enumerate(routes):
            self.m.AddHint(self.L[k], self._scaled_len(k, r))

    def _finish(self, symmetry, S_lb):
        self.symmetry = symmetry
        inst, n, M = self.inst, self.inst.n_tasks, self.inst.n_agents
        for k in range(M):
            for j in range(n):
                if self.y[k, j] is not None:
                    reach = int(self.D[k][0, j + 1] + self.D[k][j + 1, 0])
                    if reach:
                        self.m.Add(self.L[k] >= reach).OnlyEnforceIf(self.y[k, j])
        if symmetry == "index":
            for g in _identical_groups(inst):
                for a, b in zip(g, g[1:]):
                    for j in range(n):
                        if self.y[b, j] is None:
                            continue
                        prev = [self.y[a, i] for i in range(j) if self.y[a, i] is not None]
                        if prev:
                            self.m.AddBoolOr(prev).OnlyEnforceIf(self.y[b, j])
                        else:
                            self.m.AddBoolOr([self.y[b, j].Not()])
        elif symmetry == "length":
            for g in _identical_groups(inst):
                for a, b in zip(g, g[1:]):
                    self.m.Add(self.L[a] >= self.L[b])
        self.Ssum = self.m.NewIntVar(0, self.ub * M, "S")
        self.m.Add(self.Ssum == sum(self.L))
        self.Lam = self.m.NewIntVar(0, self.ub, "Lam")
        self.m.AddMaxEquality(self.Lam, self.L)
        if S_lb is not None:
            lb = int(np.floor(S_lb * self.scale - 0.5 * (n + M)))
            self.m.Add(self.Ssum >= lb)
            self.m.Add(M * self.Lam >= lb)


class _VehicleModel(_BaseModel):
    def __init__(self, inst, scale=SCALE, symmetry="index", S_lb=None):
        self._init_common(inst, scale)
        n, M = inst.n_tasks, inst.n_agents
        can = inst.can_do()
        self.x, self.L, self.y = [], [], np.empty((M, n), dtype=object)
        self.idle, self.skip = [], {}
        for k in range(M):
            arcs, xk, terms = [], {}, []
            nodes = [0] + [j + 1 for j in range(n) if can[k, j]]
            idle = self.m.NewBoolVar(f"idle{k}")
            self.idle.append(idle)
            arcs.append((0, 0, idle))
            for j in range(n):
                if can[k, j]:
                    skip = self.m.NewBoolVar(f"skip{k}_{j}")
                    self.skip[k, j] = skip
                    arcs.append((j + 1, j + 1, skip))
                    self.y[k, j] = skip.Not()
                    self.m.AddImplication(idle, skip)
                else:
                    self.y[k, j] = None
            for i in nodes:
                for j in nodes:
                    if i == j:
                        continue
                    lit = self.m.NewBoolVar(f"x{k}_{i}_{j}")
                    arcs.append((i, j, lit))
                    xk[i, j] = lit
                    if self.D[k][i, j]:
                        terms.append(int(self.D[k][i, j]) * lit)
            self.m.AddCircuit(arcs)
            Lk = self.m.NewIntVar(0, self.ub, f"L{k}")
            self.m.Add(Lk == sum(terms))
            self.x.append(xk)
            self.L.append(Lk)
        for j in range(n):
            self.m.AddExactlyOne([self.y[k, j] for k in range(M) if self.y[k, j] is not None])
        self._finish(symmetry, S_lb)

    def routes(self, solver):
        out = []
        for xk in self.x:
            nxt = {i: j for (i, j), lit in xk.items() if solver.BooleanValue(lit)}
            r, cur = [], nxt.get(0)
            while cur is not None and cur != 0:
                r.append(cur - 1)
                cur = nxt.get(cur)
            out.append(r)
        return out

    def add_hint(self, routes):
        self.m.ClearHints()
        routes = self._canonical(routes)
        for k, r in enumerate(routes):
            seq = [0] + [j + 1 for j in r] + ([0] if r else [])
            used = set(zip(seq, seq[1:]))
            for (i, j), lit in self.x[k].items():
                self.m.AddHint(lit, (i, j) in used)
            self.m.AddHint(self.idle[k], not r)
            mine = set(r)
            for (kk, j), v in self.skip.items():
                if kk == k:
                    self.m.AddHint(v, j not in mine)
        self._hint_common(routes)


class _GiantModel(_BaseModel):
    def __init__(self, inst, scale=SCALE, symmetry="index", S_lb=None):
        self._init_common(inst, scale)
        n, M = inst.n_tasks, inst.n_agents
        can = inst.can_do()
        m, D = self.m, self.D
        dtt = D[0][1:, 1:]
        self.arcs = {}
        arcs, cost_terms = [], []

        def arc(u, v, c):
            lit = m.NewBoolVar(f"a{u}_{v}")
            arcs.append((u, v, lit))
            self.arcs[u, v] = lit
            if c:
                cost_terms.append(int(c) * lit)
            return lit

        lab = [m.NewIntVarFromDomain(cp_model.Domain.FromValues(
            [k for k in range(M) if can[k, j]]), f"lab{j}") for j in range(n)]
        acc = [m.NewIntVar(0, self.ub, f"acc{j}") for j in range(n)]
        self.lab, self.acc = lab, acc
        self.L = [m.NewIntVar(0, self.ub, f"L{k}") for k in range(M)]
        for k in range(M):
            nxt = (k + 1) % M
            if M > 1:
                lit = arc(k, nxt, 0)                          # аппарат k не вылетает
                m.Add(self.L[k] == 0).OnlyEnforceIf(lit)
            for j in range(n):
                if not can[k, j]:
                    continue
                lit = arc(k, M + j, D[k][0, j + 1])           # первая задача аппарата k
                m.Add(lab[j] == k).OnlyEnforceIf(lit)
                m.Add(acc[j] == int(D[k][0, j + 1])).OnlyEnforceIf(lit)
                lit = arc(M + j, nxt, D[k][j + 1, 0])         # последняя задача аппарата k
                m.Add(lab[j] == k).OnlyEnforceIf(lit)
                m.Add(self.L[k] == acc[j] + int(D[k][j + 1, 0])).OnlyEnforceIf(lit)
        for i in range(n):
            for j in range(n):
                if i != j and (can[:, i] & can[:, j]).any():
                    lit = arc(M + i, M + j, dtt[i, j])
                    m.Add(lab[j] == lab[i]).OnlyEnforceIf(lit)
                    m.Add(acc[j] == acc[i] + int(dtt[i, j])).OnlyEnforceIf(lit)
        m.AddCircuit(arcs)
        self.y = np.empty((M, n), dtype=object)
        for j in range(n):
            lits = []
            for k in range(M):
                if can[k, j]:
                    b = m.NewBoolVar(f"y{k}_{j}")
                    m.Add(lab[j] == k).OnlyEnforceIf(b)
                    m.Add(lab[j] != k).OnlyEnforceIf(b.Not())
                    self.y[k, j] = b
                    lits.append(b)
                else:
                    self.y[k, j] = None
            m.AddExactlyOne(lits)
        self._finish(symmetry, S_lb)
        m.Add(self.Ssum == sum(cost_terms))   # то же S через дуги — для линейной оценки

    def routes(self, solver):
        M = self.inst.n_agents
        nxt = {u: v for (u, v), lit in self.arcs.items() if solver.BooleanValue(lit)}
        out = [[] for _ in range(M)]
        k, cur = 0, nxt[0]
        for _ in range(len(nxt)):
            if cur < M:
                if cur == 0:
                    break
                k = cur
            else:
                out[k].append(cur - M)
            cur = nxt[cur]
        return out

    def add_hint(self, routes):
        M = self.inst.n_agents
        routes = self._canonical(routes)
        used = set()
        for k, r in enumerate(routes):
            seq = [k] + [M + j for j in r] + [(k + 1) % M]
            used |= set(zip(seq, seq[1:]))
        self.m.ClearHints()
        for (u, v), lit in self.arcs.items():
            self.m.AddHint(lit, (u, v) in used)
        dtt = self.D[0][1:, 1:]
        for k, r in enumerate(routes):
            a = None
            for i, j in enumerate(r):
                a = int(self.D[k][0, j + 1]) if i == 0 else a + int(dtt[r[i - 1], j])
                self.m.AddHint(self.lab[j], k)
                self.m.AddHint(self.acc[j], a)
                for kk in range(M):
                    if self.y[kk, j] is not None:
                        self.m.AddHint(self.y[kk, j], kk == k)
        self._hint_common(routes)


MODELS = {"vehicle": _VehicleModel, "giant": _GiantModel}


def _build(inst, model, symmetry, S_lb):
    return MODELS[model](inst, symmetry=symmetry, S_lb=S_lb)


def _solve(model, objective_var, time_limit, workers, hint=None, seed=0):
    model.m.Minimize(objective_var)
    if hint is not None:
        model.add_hint(hint)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = float(time_limit)
    s.parameters.num_search_workers = int(workers)
    s.parameters.random_seed = int(seed)
    t0 = time.perf_counter()
    st = s.Solve(model.m)
    wall = time.perf_counter() - t0
    return s, s.StatusName(st), wall


def _result(model, solver, status, wall, objective, var, T=None, bound_div=None):
    res = ExactResult(objective=objective, status=status, wall=wall, T=T)
    if status in ("OPTIMAL", "FEASIBLE"):
        routes = model.routes(solver)
        lengths = route_lengths(model.inst, routes)
        res.routes, res.lengths = routes, lengths
        res.S, res.Lam = float(lengths.sum()), float(lengths.max())
        val = solver.Value(var)
        bnd = solver.BestObjectiveBound()
        res.bound = bnd / (model.scale if bound_div is None else bound_div)
        res.gap = 0.0 if val == 0 else max(0.0, (val - bnd) / val)
        res.info = dict(conflicts=solver.NumConflicts(), branches=solver.NumBranches())
    return res


def _solve_single(inst, objective, T, time_limit, workers, hint, seed, symmetry, S_lb, model,
                  S_max=None, Lam_max=None, w=None, norm=None):
    mdl = _build(inst, model, symmetry, S_lb)
    slack = 0.5 * (inst.n_tasks + inst.n_agents)   # округление дуг
    if S_max is not None:
        mdl.m.Add(mdl.Ssum <= int(np.ceil(S_max * mdl.scale + slack)))
    if Lam_max is not None:
        mdl.m.Add(mdl.Lam <= int(np.ceil(Lam_max * mdl.scale + slack)))
    if objective == "S":
        var = mdl.Ssum
    elif objective == "Lambda":
        var = mdl.Lam
    elif objective == "S|L<=T":
        for Lk in mdl.L:
            mdl.m.Add(Lk <= int(np.floor(T * mdl.scale)))
        var = mdl.Ssum
    elif objective == "W":
        a, b = weight_coefs(w, norm)
        M = inst.n_agents
        var = mdl.m.NewIntVar(0, a * mdl.ub * M + b * mdl.ub, "F")
        mdl.m.Add(var == a * mdl.Ssum + b * mdl.Lam)
    else:
        raise ValueError(objective)
    s, st, wall = _solve(mdl, var, time_limit, workers, hint, seed)
    div = W_SCALE * norm[1] * mdl.scale if objective == "W" else None
    res = _result(mdl, s, st, wall, objective, var, T, bound_div=div)
    res.info["model"] = model
    if objective == "W":
        res.info.update(w=w, norm=tuple(norm))
        if res.S is not None:
            res.info["F"] = weighted_value(res.S, res.Lam, w, norm)
    return res


def solve(inst, objective="S", T=None, time_limit=600, workers=8, hint=None, seed=0,
          symmetry="index", S_lb=None, model="portfolio", S_max=None, Lam_max=None,
          w=None, norm=None):
    """Одна задача: objective = 'S' (минимум суммарного пути), 'Lambda' (минимум времени
    выполнения), 'S|L<=T' (минимум S при L_k ≤ T для всех k) или 'W' (минимум
    F_w = w·S/S° + (1 − w)·Λ/Λ° при нормировке norm = (S°, Λ°)).
    S_lb — известная нижняя оценка суммарного пути (усиливает оценку для Λ).

    model='portfolio': сначала одна формулировка на половину лимита, при недоказанном оптимуме —
    другая на оставшееся время с подсказкой от лучшего найденного решения. Берётся лучшее
    решение и наибольшая из нижних оценок (обе оценки допустимы). В пробном расчёте предела точного решения «общий обход» быстрее
    доказывает однородные задачи, «по аппаратам» — разнотипные; первой идёт подходящая.

    Подсказка (hint) — допустимое решение, обычно от маршрутного решателя. Если решатель за лимит
    не нашёл ничего лучше, возвращается подсказка (с нижней оценкой решателя): результат никогда
    не хуже подсказки."""
    res = _solve_core(inst, objective, T, time_limit, workers, hint, seed, symmetry, S_lb, model,
                      S_max, Lam_max, w, norm)
    cand = _hint_result(inst, objective, hint, T, S_max, Lam_max, w, norm)
    if cand is None:
        return res
    key = _objective_key(objective, w, norm)
    if res.routes is not None and key(cand) >= key(res) - 1e-9:
        return res
    res.S, res.Lam, res.lengths, res.routes = cand.S, cand.Lam, cand.lengths, cand.routes
    res.info = dict(res.info, from_hint=True)
    if objective == "W":
        res.info["F"] = key(cand)
    val = key(cand)
    if res.bound is not None:
        tol = 0.005 * (inst.n_tasks + inst.n_agents)
        if objective == "W":
            tol = tol * (w / norm[0] + (1 - w) / norm[1]) + 1e-4 * val
        res.gap = 0.0 if val <= res.bound + tol else max(0.0, (val - res.bound) / val)
        res.status = "OPTIMAL" if res.gap == 0.0 else "FEASIBLE"
    else:
        res.status = "FEASIBLE"
    return res


def _objective_key(objective, w=None, norm=None):
    if objective == "Lambda":
        return lambda r: r.Lam
    if objective == "W":
        return lambda r: weighted_value(r.S, r.Lam, w, norm)
    return lambda r: r.S


def _hint_result(inst, objective, hint, T, S_max, Lam_max, w, norm):
    """Подсказка как решение, если она допустима для этой задачи; иначе None."""
    if hint is None:
        return None
    routes = [list(map(int, r)) for r in hint]
    if len(routes) != inst.n_agents or sorted(j for r in routes for j in r) != list(range(inst.n_tasks)):
        return None
    can = inst.can_do()
    if not all(can[k, j] for k, r in enumerate(routes) for j in r):
        return None
    L = route_lengths(inst, routes)
    tol = 0.005 * (inst.n_tasks + inst.n_agents)
    if objective == "S|L<=T" and L.max() > T + 1e-9:
        return None
    if S_max is not None and L.sum() > S_max + tol:
        return None
    if Lam_max is not None and L.max() > Lam_max + tol:
        return None
    return ExactResult(objective=objective, status="FEASIBLE", S=float(L.sum()), Lam=float(L.max()),
                       lengths=L, routes=routes, T=T, info=dict(model="hint"))


def single_tour_reducible(inst):
    """Общая база с возвратом и одинаковые умения: минимум S достигается одним аппаратом
    (два замкнутых маршрута через базу склеиваются в один той же длины), поэтому S* — это
    кратчайший обход базы и всех задач одним аппаратом."""
    return bool(inst.returns and np.allclose(inst.starts, inst.starts[0])
                and (inst.capabilities == inst.capabilities[0]).all())


def _solve_single_tour(inst, time_limit, workers, hint, seed):
    """Минимум S сведением к одному аппарату (см. single_tour_reducible)."""
    import dataclasses
    one = dataclasses.replace(inst, starts=inst.starts[:1], capabilities=inst.capabilities[:1])
    h1 = None
    if hint is not None:
        h1 = [max(hint, key=len)] if sum(1 for r in hint if r) <= 1 else None
    r = _solve_single(one, "S", None, time_limit, workers, h1, seed, "index", None, "vehicle")
    if r.routes is not None:
        routes = r.routes + [[] for _ in range(inst.n_agents - 1)]
        r.routes = routes
        r.lengths = route_lengths(inst, routes)
        r.S, r.Lam = float(r.lengths.sum()), float(r.lengths.max())
    r.info = dict(r.info, model="single_tour")
    return r


def _solve_core(inst, objective="S", T=None, time_limit=600, workers=8, hint=None, seed=0,
          symmetry="index", S_lb=None, model="portfolio", S_max=None, Lam_max=None,
          w=None, norm=None):
    """Решение без проверки подсказки (см. solve)."""
    if objective == "S" and S_max is None and Lam_max is None and single_tour_reducible(inst):
        return _solve_single_tour(inst, time_limit, workers, hint, seed)
    if objective == "W" and (w is None or norm is None or not 0 < w < 1):
        raise ValueError("для 'W' нужны 0 < w < 1 и norm = (S°, Λ°)")
    extra = dict(S_max=S_max, Lam_max=Lam_max, w=w, norm=norm)
    if model != "portfolio":
        return _solve_single(inst, objective, T, time_limit, workers, hint, seed, symmetry, S_lb,
                             model, **extra)
    homogeneous = bool((inst.capabilities == inst.capabilities[0]).all())
    first, second = ("giant", "vehicle") if homogeneous else ("vehicle", "giant")
    a = _solve_single(inst, objective, T, time_limit / 2, workers, hint, seed, symmetry, S_lb,
                      first, **extra)
    if a.optimal:
        return a
    b = _solve_single(inst, objective, T, time_limit - a.wall, workers,
                      a.routes if a.routes is not None else hint, seed, symmetry, S_lb, second,
                      **extra)
    sols = [r for r in (a, b) if r.routes is not None]
    if not sols:
        return b
    rnd = 0.005 * (inst.n_tasks + inst.n_agents)   # округление дуг, в единицах расстояния
    if objective == "Lambda":
        key, tol_of = (lambda r: r.Lam), (lambda v: rnd)
    elif objective == "W":
        key = lambda r: weighted_value(r.S, r.Lam, w, norm)
        tol_of = lambda v: rnd * (w / norm[0] + (1 - w) / norm[1]) + 1e-4 * v
    else:
        key, tol_of = (lambda r: r.S), (lambda v: rnd)
    best = min(sols, key=key)
    bounds = [r.bound for r in (a, b) if r.bound is not None]
    out = ExactResult(objective=objective, status=best.status, S=best.S, Lam=best.Lam,
                      lengths=best.lengths, routes=best.routes, wall=a.wall + b.wall, T=T,
                      info=dict(model="portfolio", best_from=best.info.get("model")))
    if objective == "W":
        out.info.update(w=w, norm=tuple(norm), F=key(best))
    if bounds:
        out.bound = max(bounds)
        val = key(best)
        tol = tol_of(val)
        out.gap = 0.0 if val <= out.bound + tol else max(0.0, (val - out.bound) / val)
        if b.optimal or out.gap == 0.0:
            out.status = "OPTIMAL"
    return out


def solve_lex(inst, first="Lambda", time_limit=600, workers=8, hint=None, seed=0,
              symmetry="index", S_lb=None, model="portfolio"):
    """Лексикографически: сначала минимум first, затем при найденном значении — минимум второго.
    Дают крайние точки фронта. Время делится поровну между этапами; оба этапа — через solve(),
    то есть по умолчанию портфелем формулировок. Точка считается точной, только если доказаны
    оба этапа."""
    kw = dict(workers=workers, seed=seed, symmetry=symmetry, S_lb=S_lb, model=model)
    second = "S" if first == "Lambda" else "Lambda"
    st1 = solve(inst, first, time_limit=time_limit / 2, hint=hint, **kw)
    if st1.routes is None:
        st1.objective = f"lex_{first}"
        return st1
    if first == "S" and st1.optimal and single_tour_reducible(inst):
        # общая база, одинаковые аппараты: минимум S — один обход; в общем положении другого
        # решения с той же длиной нет, поэтому второй этап не нужен (Λ = S*)
        return ExactResult(objective="lex_S", status="OPTIMAL", S=st1.S, Lam=st1.Lam,
                           lengths=st1.lengths, routes=st1.routes, bound=st1.Lam, gap=0.0,
                           wall=st1.wall, info=dict(stage1_status=st1.status, stage1_bound=st1.bound,
                                                    stage1_gap=st1.gap, stage2_status="single_tour"))
    v1 = st1.Lam if first == "Lambda" else st1.S
    bound_kw = dict(Lam_max=v1) if first == "Lambda" else dict(S_max=v1)
    st2 = solve(inst, second, time_limit=time_limit / 2, hint=st1.routes, **kw, **bound_kw)
    key = (lambda r: (r.Lam, r.S)) if first == "Lambda" else (lambda r: (r.S, r.Lam))
    best = min([r for r in (st1, st2) if r.routes is not None], key=key)
    return ExactResult(objective=f"lex_{first}",
                       status="OPTIMAL" if (st1.optimal and st2.optimal) else "FEASIBLE",
                       S=best.S, Lam=best.Lam, lengths=best.lengths, routes=best.routes,
                       bound=st2.bound, gap=st2.gap, wall=st1.wall + st2.wall,
                       info=dict(stage1_status=st1.status, stage1_bound=st1.bound,
                                 stage1_gap=st1.gap, stage2_status=st2.status))


WEIGHTS = (0.25, 0.5, 0.75)


def pareto_front(inst, n_inner=5, time_limit=600, workers=8, seed=0, symmetry="index",
                 model="portfolio", heuristic_share=0.1, weights=WEIGHTS, lean=False):
    """Фронт «суммарный путь S — время выполнения Λ»: две крайние точки, n_inner точек
    методом ε-ограничений между ними и минимумы взвешенного критерия F_w для w из weights
    (нормировка — крайние точки). Возвращает (недоминируемые точки, все решения).

    Порядок: (1) маршрутный решатель OR-Tools даёт хорошие решения на минимум S и на минимум Λ
    за долю heuristic_share лимита — они идут подсказками в CP-SAT; (2) лексикографический
    минимум S — его нижняя оценка усиливает задачи с ограничением на Λ; (3) лексикографический
    минимум Λ; (4) точки ε-ограничений; (5) минимумы F_w — эталон для предпочтений «поровну» и
    проверки чувствительности к весу. Для каждой точки сохраняется нижняя оценка решателя.

    lean=True — облегчённый эталон для основного эксперимента (по пилоту 12.10.2026): только то, что нужно
    показателям g_S, g_Λ, g_w — минимум S, минимум Λ (каждый с полным лимитом) и минимумы F_w;
    без точек ε-ограничений и без вторых этапов лексикографических задач. Полный фронт считается
    на подвыборке."""
    from .best_known import solve_routing
    kw = dict(workers=workers, seed=seed, symmetry=symmetry)
    h_time = max(1.0, heuristic_share * time_limit)
    hS = solve_routing(inst, "S", h_time) if heuristic_share > 0 else None
    hL = solve_routing(inst, "Lambda", h_time) if heuristic_share > 0 else None
    if lean:
        b = solve(inst, "S", time_limit=time_limit, hint=hS.routes if hS else None, model=model,
                  **kw)
        S_lb = b.bound
        a = solve(inst, "Lambda", time_limit=time_limit, S_lb=S_lb,
                  hint=hL.routes if hL else None, model=model, **kw)
    else:
        b = solve_lex(inst, "S", time_limit, hint=hS.routes if hS else None, model=model, **kw)
        S_lb = b.info.get("stage1_bound")
        a = solve_lex(inst, "Lambda", time_limit, S_lb=S_lb, hint=hL.routes if hL else None,
                      model=model, **kw)
    sols = [a, b]
    if not lean and a.Lam is not None and b.Lam is not None and b.Lam > a.Lam * (1 + 1e-9):
        for T in np.linspace(a.Lam, b.Lam, n_inner + 2)[1:-1]:
            # решение с минимальным Λ допустимо при любом T ≥ Λ* — подсказка решателю
            sols.append(solve(inst, "S|L<=T", T=T, time_limit=time_limit, hint=a.routes,
                              S_lb=S_lb, model=model, **kw))
    if a.Lam is not None and b.S is not None:
        norm = (b.S, a.Lam)
        for w in weights:
            cand = [r for r in sols if r.routes is not None]
            hint = min(cand, key=lambda r: weighted_value(r.S, r.Lam, w, norm)).routes
            sols.append(solve(inst, "W", w=w, norm=norm, time_limit=time_limit, hint=hint,
                              S_lb=S_lb, model=model, **kw))
    pts = [(r.S, r.Lam) for r in sols if r.S is not None]
    for h in (hS, hL):   # решения эвристики тоже допустимы — могут оказаться на фронте
        if h is not None and h.S is not None:
            pts.append((h.S, h.Lam))
    return nondominated(pts), sols


def nondominated(points, tol=1e-9):
    """Недоминируемые точки (S, Λ), по возрастанию Λ."""
    pts = sorted(set((round(s, 9), round(l, 9)) for s, l in points), key=lambda p: (p[1], p[0]))
    out = []
    for s, l in pts:
        if not out or s < out[-1][0] - tol:
            out.append((s, l))
    return out


def eps_indicator(S, Lam, front):
    """Мультипликативный ε-индикатор: насколько (в долях) нужно ухудшить точки фронта,
    чтобы ни одна не была лучше (S, Λ) по обоим показателям. 0 — точка на фронте."""
    return min(max(S / s, Lam / l) for s, l in front) - 1.0
