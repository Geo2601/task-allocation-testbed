"""Анализ (статья, раздел 4): сравнение методов внутри ячеек и правило выбора метода.

Правило выбора — «дерево политики»: по признакам, известным до распределения (скученность R,
задач на аппарат ρ, число аппаратов M, общая база или нет, разнотипная группа или нет), дерево
глубиной 2–3 выбирает метод. Дерево строится не на угадывание «лучшего метода» (классификация),
а сразу на наименьшее сожаление: в каждом листе выбирается метод с наименьшим средним разрывом,
разбиения — те, что сильнее всего уменьшают суммарный разрыв. Так дерево не тратит ветви на
различение методов, которые почти равны.

Разбиение берётся, только если снижает средний разрыв по всей обучающей выборке хотя бы на min_gain
(по умолчанию 0,005, то есть полпроцента): иначе дерево обрастает ветвями, которые почти ничего не
дают читателю (так было на первых 3000 примерах основного эксперимента).

Сожаление политики = среднее по примерам (разрыв выбранного метода − наименьший разрыв среди
методов на этом примере). Сравнение — с правилами «всегда один метод».
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from .stats import bootstrap_median, compare_methods

FEATURES = ("R", "rho", "M", "base", "hetero")
FEATURE_NAMES = {"R": "скученность R", "rho": "задач на аппарат ρ", "M": "аппаратов M",
                 "base": "общая база", "hetero": "разнотипная группа"}


def features(inst_table: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=inst_table.index)
    X["R"] = inst_table["R"].astype(float)
    X["rho"] = inst_table["rho"].astype(float)
    X["M"] = inst_table["M"].astype(float)
    X["base"] = (inst_table["start"] == "base").astype(float)
    X["hetero"] = inst_table["hetero"].astype(bool).astype(float)
    return X


def loss_matrix(d: pd.DataFrame, pref: str, methods) -> pd.DataFrame:
    """Примеры × методы: разрыв по выбранному предпочтению."""
    G = d[d.method.isin(methods)].pivot_table(index="key", columns="method", values=pref)
    return G[[m for m in methods if m in G.columns]].dropna()


@dataclass
class _Node:
    method: str
    n: int
    feature: Optional[str] = None
    threshold: Optional[float] = None
    left: Optional["_Node"] = None     # признак ≤ порога
    right: Optional["_Node"] = None


@dataclass
class PolicyTree:
    max_depth: int = 3
    min_leaf: int = 20
    max_thresholds: int = 24
    min_gain: float = 0.0      # разбиение берётся, если снижает средний разрыв по всей выборке хотя бы на столько
    root: Optional[_Node] = None
    methods: List[str] = field(default_factory=list)

    def fit(self, X: pd.DataFrame, G: pd.DataFrame):
        self.methods = list(G.columns)
        self._n_total = len(G)
        self.root = self._grow(X.to_numpy(float), G.to_numpy(float), list(X.columns), 0)
        return self

    def _leaf(self, G):
        return self.methods[int(np.argmin(G.mean(axis=0)))]

    def _grow(self, X, G, names, depth):
        node = _Node(method=self._leaf(G), n=len(G))
        if depth >= self.max_depth or len(G) < 2 * self.min_leaf:
            return node
        best_cost = G.sum(axis=0).min()
        best = None
        for f in range(X.shape[1]):
            vals = np.unique(X[:, f])
            if len(vals) < 2:
                continue
            cuts = (vals[:-1] + vals[1:]) / 2
            if len(cuts) > self.max_thresholds:
                cuts = np.unique(np.quantile(X[:, f], np.linspace(0, 1, self.max_thresholds + 2)[1:-1]))
            for c in cuts:
                m = X[:, f] <= c
                if m.sum() < self.min_leaf or (~m).sum() < self.min_leaf:
                    continue
                cost = G[m].sum(axis=0).min() + G[~m].sum(axis=0).min()
                if cost < best_cost - 1e-12:
                    best_cost, best = cost, (f, float(c), m)
        if best is None or (G.sum(axis=0).min() - best_cost) < self.min_gain * self._n_total:
            return node
        f, c, m = best
        node.feature, node.threshold = names[f], c
        node.left = self._grow(X[m], G[m], names, depth + 1)
        node.right = self._grow(X[~m], G[~m], names, depth + 1)
        if node.left.feature is None and node.right.feature is None and \
                node.left.method == node.right.method:
            node.feature = node.threshold = node.left = node.right = None   # бесполезное разбиение
        return node

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        out = []
        for _, row in X.iterrows():
            node = self.root
            while node.feature is not None:
                node = node.left if row[node.feature] <= node.threshold else node.right
            out.append(node.method)
        return np.array(out)

    def rules(self) -> str:
        lines = []

        def fmt(f, c, le):
            name = FEATURE_NAMES.get(f, f)
            if f in ("base", "hetero"):
                return f"{name}: {'нет' if le else 'да'}"
            return f"{name} {'≤' if le else '>'} {c:.3g}"

        def walk(node, pre):
            if node.feature is None:
                lines.append(f"{pre}→ {node.method} ({node.n} прим.)")
                return
            lines.append(f"{pre}если {fmt(node.feature, node.threshold, True)}:")
            walk(node.left, pre + "    ")
            lines.append(f"{pre}если {fmt(node.feature, node.threshold, False)}:")
            walk(node.right, pre + "    ")
        walk(self.root, "")
        return "\n".join(lines)


def regret(choice: np.ndarray, G: pd.DataFrame) -> float:
    vals = G.to_numpy(float)
    idx = np.array([G.columns.get_loc(c) for c in choice])
    return float(np.mean(vals[np.arange(len(vals)), idx] - vals.min(axis=1)))


def evaluate_policy(d: pd.DataFrame, pref: str, methods, train_mask_fn, max_depth=3, min_leaf=20,
                    min_gain=0.005):
    """Обучение дерева на основном эксперименте (train), проверка на отложенной части основного эксперимента и
    на проверках устойчивости. Возвращает (дерево, таблица сожалений)."""
    inst = d.drop_duplicates("key").set_index("key")
    G = loss_matrix(d, pref, methods)
    inst = inst.loc[G.index]
    X = features(inst)
    is_main = inst["prefix"] == "main"
    tr = is_main & train_mask_fn(inst)
    tree = PolicyTree(max_depth=max_depth, min_leaf=min_leaf, min_gain=min_gain).fit(X[tr], G[tr])
    parts = {"обучение": tr, "отложенные (основной эксперимент)": is_main & ~tr}
    for p in sorted(set(inst["prefix"]) - {"main"}):
        parts[f"проверка: {p}"] = inst["prefix"] == p
    rows = []
    for name, mask in parts.items():
        if mask.sum() == 0:
            continue
        row = dict(часть=name, примеров=int(mask.sum()),
                   дерево=regret(tree.predict(X[mask]), G[mask]))
        for m in G.columns:
            row[f"всегда {m}"] = regret(np.full(mask.sum(), m), G[mask])
        rows.append(row)
    return tree, pd.DataFrame(rows)


def cell_comparison(d: pd.DataFrame, pref: str, methods) -> pd.DataFrame:
    """Внутри каждой ячейки: медианы с бутстреп-интервалами, Фридман, лучший метод и значимость
    его отрыва от второго (Уилкоксон, Холм)."""
    cells = ["prefix", "level", "M", "rho", "start", "hetero"]
    rows = []
    for key, g in d[d.method.isin(methods)].groupby(cells):
        piv = g.pivot_table(index="rep", columns="method", values=pref).dropna()
        if len(piv) < 5:
            continue
        cmp = compare_methods({m: piv[m].to_numpy() for m in piv.columns})
        med = piv.median()
        best, second = med.nsmallest(2).index
        pair = next(p for p in cmp["pairs"] if {p["a"], p["b"]} == {best, second})
        row = dict(zip(cells, key), n=len(piv), friedman_p=cmp.get("friedman_p"), best=best,
                   second=second, p_best_vs_second=pair["p_holm"])
        for m in piv.columns:
            lo, hi = bootstrap_median(piv[m].to_numpy(), n_boot=2000)
            row[f"{m}_median"], row[f"{m}_ci"] = float(med[m]), f"{lo:.3f}–{hi:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)
