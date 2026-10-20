"""Подсказки CP-SAT: перестановка маршрутов под упорядочение одинаковых аппаратов и правило
«результат не хуже подсказки»."""
import numpy as np
import pytest

from allocsim.best_known import solve_routing
from allocsim.exact import _VehicleModel, _GiantModel, solve
from allocsim.instance import InstanceSpec, main_field, make_instance


def _base_inst(M=4, rho=6, rep=0):
    return make_instance(InstanceSpec(field=main_field(2, 1000.0, M, rho, "L2"), n_agents=M,
                                      start_mode="base"), rep)


@pytest.mark.parametrize("Model", [_VehicleModel, _GiantModel])
def test_canonical_routes_satisfy_index_order(Model):
    inst = _base_inst()
    mdl = Model(inst, symmetry="index")
    routes = [[], [7, 3], [0, 9], [5]]            # нарушает упорядочение
    can = mdl._canonical(routes)
    mins = [min(r) for r in can if r]
    assert mins == sorted(mins) and can[-1] == []  # пустые — в конце, остальные по наименьшей задаче
    assert sorted(map(tuple, can)) == sorted(map(tuple, routes))


@pytest.mark.parametrize("model", ["vehicle", "giant", "portfolio"])
@pytest.mark.parametrize("obj", ["S", "Lambda"])
def test_result_never_worse_than_hint(model, obj):
    inst = _base_inst(M=4, rho=6, rep=1)
    h = solve_routing(inst, obj, time_limit=2)
    # нарочно переставим маршруты, чтобы подсказка нарушала упорядочение одинаковых аппаратов
    hint = h.routes[::-1]
    r = solve(inst, obj, time_limit=0.2, workers=1, hint=hint, model=model)
    key = (lambda x: x.S) if obj == "S" else (lambda x: x.Lam)
    assert r.routes is not None
    assert key(r) <= key(h) + 1e-6
    assert sorted(j for x in r.routes for j in x) == list(range(inst.n_tasks))


def test_hint_with_symmetry_is_accepted_quickly():
    """На однородной группе с общей базой подсказка от маршрутного решателя после перестановки
    допустима: решатель стартует с неё (раньше в этом случае CP-SAT мог закончить хуже подсказки)."""
    inst = _base_inst(M=8, rho=6, rep=0)
    h = solve_routing(inst, "Lambda", time_limit=3)
    r = solve(inst, "Lambda", time_limit=5, workers=2, hint=h.routes, model="giant")
    assert r.Lam <= h.Lam + 1e-6


def test_single_tour_reduction_matches_full_model():
    from allocsim.exact import single_tour_reducible
    from allocsim.verify import brute_force
    for rep in range(3):
        inst = make_instance(InstanceSpec(field=main_field(2, 1000.0, 3, 2, "L2"), n_agents=3,
                                          start_mode="base"), rep)
        assert single_tour_reducible(inst)
        r = solve(inst, "S", time_limit=20, workers=2)
        assert r.optimal and r.info["model"] == "single_tour"
        assert abs(r.S - brute_force(inst)["S"]) < 0.05
    het = make_instance(InstanceSpec(field=main_field(2, 1000.0, 4, 3, "L2"), n_agents=4,
                                     start_mode="base", heterogeneous=True), 0)
    sc = make_instance(InstanceSpec(field=main_field(2, 1000.0, 4, 3, "L2"), n_agents=4,
                                    start_mode="scattered"), 0)
    assert not single_tour_reducible(het) and not single_tour_reducible(sc)
