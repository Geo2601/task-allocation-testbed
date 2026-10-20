"""Дерево политики находит порог, при котором меняется лучший метод, и даёт нулевое сожаление
там, где выбор очевиден; сожаление «всегда одного метода» считается верно."""
import numpy as np
import pandas as pd

from allocsim.analysis import PolicyTree, evaluate_policy, regret


def _synthetic(n=400, seed=0):
    r = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        R = r.uniform(0.2, 1.0)
        start = "base" if i % 2 else "scattered"
        # при R < 0,6 лучше «a», иначе «b»; при общей базе «c» всегда хуже
        ga = 0.1 if R < 0.6 else 0.3
        gb = 0.3 if R < 0.6 else 0.1
        gc = 0.5 if start == "base" else 0.2
        for m, g in (("a", ga), ("b", gb), ("c", gc)):
            rows.append(dict(key=f"k{i}", prefix="main", rep=i, level="L0", M=8, rho=6,
                             start=start, hetero=False, R=R, method=m, g_S=g + r.normal(0, 0.005)))
    return pd.DataFrame(rows)


def test_tree_finds_threshold():
    d = _synthetic()
    tree, reg = evaluate_policy(d, "g_S", ("a", "b", "c"), lambda inst: inst["rep"] % 10 < 7,
                                max_depth=2, min_leaf=10)
    assert tree.root.feature == "R" and abs(tree.root.threshold - 0.6) < 0.05
    test = reg[reg["часть"].str.startswith("отложенные")].iloc[0]
    assert test["дерево"] < 0.01 < test["всегда a"]


def test_regret_simple():
    G = pd.DataFrame({"a": [0.1, 0.3], "b": [0.2, 0.1]})
    assert abs(regret(np.array(["a", "a"]), G) - 0.1) < 1e-12
    assert abs(regret(np.array(["a", "b"]), G)) < 1e-12


def test_tree_no_useless_split():
    X = pd.DataFrame({"R": np.linspace(0, 1, 100)})
    G = pd.DataFrame({"a": np.full(100, 0.1), "b": np.full(100, 0.2)})
    t = PolicyTree(max_depth=3, min_leaf=10).fit(X, G)
    assert t.root.feature is None and t.root.method == "a"
