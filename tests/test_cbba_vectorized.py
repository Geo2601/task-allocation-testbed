"""Матричный расчёт ставок CBBA совпадает с расчётом по одной задаче (_insertion, _tdr)."""
import importlib

import numpy as np
import pytest

from allocsim.instance import InstanceSpec, main_field, make_instance

C = importlib.import_module("allocsim.methods.cbba")


@pytest.mark.parametrize("start", ["base", "scattered"])
@pytest.mark.parametrize("score", ["marginal", "tdr"])
def test_all_bids_match_scalar(start, score):
    inst = make_instance(InstanceSpec(field=main_field(2, 1000.0, 4, 6, "L2"), n_agents=4,
                                      start_mode=start), 0)
    P = inst.points
    DT = np.linalg.norm(P[:, None] - P[None], axis=2)
    DS = np.linalg.norm(inst.starts[:, None] - P[None], axis=2)
    lam = 1.0 / 1000.0
    r = np.random.default_rng(1)
    for path in ([], [3], [5, 1, 7], list(r.permutation(24)[:9])):
        rest = [j for j in range(24) if j not in path]
        val, pos = C._all_bids(DT, DS[2], path, inst.returns, score, lam)
        for j in rest:
            if score == "tdr":
                b, p = C._tdr(inst, 2, path, j, lam)
                assert np.exp(-lam * val[j]) == pytest.approx(b, rel=1e-9)
            else:
                b, p = C._insertion(inst, 2, path, j)
                assert val[j] == pytest.approx(b, abs=1e-6)
            assert pos[j] == p


def test_tdr_marginal_matches_path_value():
    """Матричный прирост ценности пути совпадает с прямым пересчётом Σ exp(−λ t) до и после вставки."""
    inst = make_instance(InstanceSpec(field=main_field(2, 1000.0, 4, 6, "L2"), n_agents=4,
                                      start_mode="scattered"), 0)
    P = inst.points
    DT = np.linalg.norm(P[:, None] - P[None], axis=2)
    DS = np.linalg.norm(inst.starts[:, None] - P[None], axis=2)
    lam = 1 / 300.0

    def value(path):
        t, v, prev = 0.0, 0.0, None
        for j in path:
            t += DS[1, j] if prev is None else DT[prev, j]
            v += np.exp(-lam * t)
            prev = j
        return v

    for path in ([], [4], [2, 9, 5], [0, 7, 3, 11, 6]):
        val, pos = C._all_bids(DT, DS[1], path, False, "tdr_marginal", lam)
        for j in [x for x in range(24) if x not in path]:
            best = max(value(path[:q] + [j] + path[q:]) - value(path) for q in range(len(path) + 1))
            assert -val[j] == pytest.approx(best, rel=1e-9, abs=1e-12)
            q = pos[j]
            assert value(path[:q] + [j] + path[q:]) - value(path) == pytest.approx(best, abs=1e-12)
