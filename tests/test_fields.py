"""Проверки генератора полей и показателя скученности. Запуск: python -m pytest -q"""
import json

import numpy as np
import pytest

from allocsim.clustering import (clark_evans_expected, clustering_index, donnelly_expected,
                                 mean_nn_distance, uniform_reference)
from allocsim.fields import FieldSpec, LEVELS, reflect_into_box, sample_field
from allocsim.instance import Instance, InstanceSpec, make_instance, main_field
from allocsim.plan import center_base_check, main_plan, skfu_check, volume_check


# --- отражение внутрь поля -------------------------------------------------------------

def test_reflect_into_box():
    x = np.array([-30.0, 0.0, 50.0, 100.0, 130.0, 260.0, -250.0])
    assert np.allclose(reflect_into_box(x, 100.0), [30, 0, 50, 100, 70, 60, 50])


# --- поля -------------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("level", list(LEVELS))
def test_points_inside_field(dim, level):
    f = main_field(dim, 1000.0, 8, 12, level)
    for rep in range(20):
        pts, cid = sample_field(f, rep)
        assert pts.shape == (96, dim)
        assert pts.min() >= 0.0 and pts.max() <= 1000.0
        assert (cid == -1).all() if f.is_uniform else ((cid >= 0) & (cid < 8)).all()


def test_reproducible_and_rep_dependent():
    f = main_field(2, 1000.0, 8, 6, "L2")
    a, _ = sample_field(f, 3)
    b, _ = sample_field(f, 3)
    c, _ = sample_field(f, 4)
    assert np.array_equal(a, b)
    assert not np.allclose(a, c)


def test_base_seed_changes_field():
    f = main_field(2, 1000.0, 8, 6, "L2")
    assert not np.allclose(sample_field(f, 0, base_seed=0)[0], sample_field(f, 0, base_seed=1)[0])


def test_common_random_numbers_across_levels():
    """Для одного повтора уровни отличаются только σ: точки = отражение(центр + σ·z)
    с одними и теми же центрами, выбором центра и z на всех уровнях."""
    from allocsim.seeds import rng as make_rng
    rep, m, n = 7, 8, 48
    r = make_rng("thomas", 0, 2, 1000.0, n, m, rep)
    centers = r.uniform(0.0, 1000.0, size=(m, 2))
    assign = r.integers(0, m, size=n)
    z = r.standard_normal(size=(n, 2))
    for lvl in ("L1", "L2", "L3", "L4"):
        f = main_field(2, 1000.0, m, 6, lvl)
        pts, cid = sample_field(f, rep)
        assert np.array_equal(cid, assign)
        assert np.allclose(pts, reflect_into_box(centers[assign] + f.sigma * z, 1000.0))


def test_sigma_modes():
    a = FieldSpec(dim=2, side=1000, n_tasks=48, sigma_frac=0.1, n_centers=16, sigma_mode="abs")
    r = FieldSpec(dim=2, side=1000, n_tasks=48, sigma_frac=0.1, n_centers=16, sigma_mode="rel")
    assert a.sigma == pytest.approx(100.0)
    assert r.sigma == pytest.approx(1000 / 4 * 0.1)


def test_bad_specs():
    with pytest.raises(ValueError):
        FieldSpec(dim=4)
    with pytest.raises(ValueError):
        FieldSpec(sigma_frac=0.1, n_centers=None)
    with pytest.raises(ValueError):
        InstanceSpec(field=FieldSpec(n_tasks=12), n_agents=2, heterogeneous=True)  # 2·2 < 3·2


# --- показатель скученности -------------------------------------------------------------

def test_mean_nn_distance_simple():
    pts = np.array([[0.0, 0.0], [3.0, 4.0], [10.0, 0.0]])
    # ближайшие: 5, 5, √(49+16)=8,06
    assert mean_nn_distance(pts) == pytest.approx((5 + 5 + np.sqrt(65)) / 3)


@pytest.mark.parametrize("n", [24, 48, 96, 192])
def test_reference_matches_donnelly(n):
    """Таблица нормировки совпадает с формулой Donnelly (1978) для квадрата с краями в пределах 2 %."""
    mean, _ = uniform_reference(n, 2)
    assert mean == pytest.approx(donnelly_expected(n, 1.0), rel=0.02)


@pytest.mark.parametrize("dim", [2, 3])
def test_reference_above_clark_evans(dim):
    """Края удлиняют расстояния до соседа: таблица всегда выше формулы без краёв,
    и разница уменьшается с ростом N."""
    gaps = [uniform_reference(n, dim)[0] / clark_evans_expected(n, 1.0, dim) - 1 for n in (12, 48, 192)]
    assert all(g > 0 for g in gaps) and gaps[0] > gaps[1] > gaps[2]


@pytest.mark.parametrize("dim,n", [(2, 12), (2, 96), (3, 48)])
def test_uniform_R_is_one_on_average(dim, n):
    f = FieldSpec(dim=dim, side=1000.0, n_tasks=n)
    rs = [clustering_index(sample_field(f, rep, base_seed=99)[0], 1000.0) for rep in range(400)]
    assert np.mean(rs) == pytest.approx(1.0, abs=0.03)


def test_R_scale_invariant():
    f = FieldSpec(dim=2, side=1000.0, n_tasks=48)
    pts, _ = sample_field(f, 0)
    assert clustering_index(pts, 1000.0) == pytest.approx(clustering_index(pts / 10.0, 100.0))


def test_R_decreases_with_level():
    med = []
    for lvl in LEVELS:
        f = main_field(2, 1000.0, 8, 12, lvl)
        med.append(np.median([clustering_index(sample_field(f, r)[0], 1000.0) for r in range(100)]))
    assert all(a > b for a, b in zip(med, med[1:])), med


# --- примеры -------------------------------------------------------------------------

def _spec(**kw):
    base = dict(field=main_field(2, 1000.0, 8, 6, "L3"), n_agents=8)
    base.update(kw)
    return InstanceSpec(**base)


def test_same_points_across_types_and_starts():
    variants = [_spec(), _spec(heterogeneous=True), _spec(start_mode="scattered"),
                _spec(heterogeneous=True, start_mode="scattered"), _spec(base_pos="center")]
    pts = [make_instance(s, 5).points for s in variants]
    for p in pts[1:]:
        assert np.array_equal(p, pts[0])


def test_same_caps_and_starts_across_levels():
    a = make_instance(_spec(field=main_field(2, 1000.0, 8, 6, "L1"), heterogeneous=True,
                            start_mode="scattered"), 2)
    b = make_instance(_spec(field=main_field(2, 1000.0, 8, 6, "L4"), heterogeneous=True,
                            start_mode="scattered"), 2)
    assert np.array_equal(a.capabilities, b.capabilities)
    assert np.array_equal(a.starts, b.starts)


@pytest.mark.parametrize("m", [4, 8, 16])
def test_capabilities_constraints(m):
    for rep in range(50):
        inst = make_instance(_spec(field=main_field(2, 1000.0, m, 3, "L2"), n_agents=m,
                                   heterogeneous=True), rep)
        caps = inst.capabilities
        assert caps.shape == (m, 3)
        assert ((caps.sum(1) >= 1) & (caps.sum(1) <= 2)).all()
        assert (caps.sum(0) >= 2).all()
        assert ((inst.task_type >= 0) & (inst.task_type < 3)).all()
        assert inst.can_do().any(axis=0).all()  # каждую задачу кто-то может выполнить


@pytest.mark.parametrize("m", [4, 8, 16])
def test_balanced_versatility_same_for_all_group_sizes(m):
    """При caps_scheme='balanced' у половины аппаратов 2 умения, у половины 1 — при любом M."""
    for rep in range(30):
        caps = make_instance(_spec(field=main_field(2, 1000.0, m, 3, "L2"), n_agents=m,
                                   heterogeneous=True), rep).capabilities
        assert sorted(caps.sum(1)) == [1] * (m // 2) + [2] * (m // 2)


def test_random_scheme_allowed():
    inst = make_instance(_spec(heterogeneous=True, caps_scheme="random"), 0)
    assert (inst.capabilities.sum(0) >= 2).all()


def test_homogeneous_everyone_can_do_everything():
    inst = make_instance(_spec(), 0)
    assert inst.capabilities.shape == (8, 1) and inst.can_do().all()
    assert (inst.task_type == 0).all()


def test_type_purity():
    """При type_purity = 1 все задачи одного скопления одного типа; при 0 типы перемешаны."""
    pure = make_instance(_spec(heterogeneous=True, type_purity=1.0), 1)
    for k in np.unique(pure.cluster_id):
        assert len(np.unique(pure.task_type[pure.cluster_id == k])) == 1
    mixed = [make_instance(_spec(heterogeneous=True), r) for r in range(20)]
    n_mixed = sum(len(np.unique(i.task_type[i.cluster_id == k])) > 1
                  for i in mixed for k in np.unique(i.cluster_id) if (i.cluster_id == k).sum() >= 3)
    assert n_mixed > 0


def test_starts():
    corner = make_instance(_spec(), 0)
    assert np.allclose(corner.starts, 0.0) and corner.returns
    center = make_instance(_spec(base_pos="center"), 0)
    assert np.allclose(center.starts, 500.0)
    sc = make_instance(_spec(start_mode="scattered"), 0)
    assert not sc.returns and len(np.unique(sc.starts[:, 0])) == 8
    assert sc.starts.min() >= 0 and sc.starts.max() <= 1000


def test_json_roundtrip(tmp_path):
    inst = make_instance(_spec(heterogeneous=True, start_mode="scattered"), 11)
    path = tmp_path / "inst.json"
    inst.save(path)
    back = Instance.load(path)
    assert back.spec == inst.spec and back.rep == inst.rep
    for name in ("points", "task_type", "capabilities", "starts", "cluster_id"):
        assert np.array_equal(getattr(back, name), getattr(inst, name))
    assert back.R == inst.R
    json.loads(path.read_text(encoding="utf-8"))


# --- план экспериментов --------------------------------------------------------------------------

def test_plan_sizes():
    assert len(main_plan()) == 180
    assert len(volume_check()) == 20
    assert len(center_base_check()) == 10
    assert len(skfu_check()) == 4
    ns = {c["spec"].field.n_tasks for c in main_plan()}
    assert ns == {12, 24, 48, 96, 192}


def test_extra_checks():
    from allocsim.instance import make_instance
    from allocsim.plan import clusters_double_check, clusters_half_check, purity_check
    half, double, pur = clusters_half_check(), clusters_double_check(), purity_check()
    assert len(half) == len(double) == 16 and len(pur) == 8
    assert {c["spec"].field.n_centers for c in half} == {4}
    assert {c["spec"].field.n_centers for c in double} == {16}
    assert all(c["level"] != "L0" for c in half + double + pur)
    # чистота типов: в каждом скоплении один тип; точки и умения — как в основном эксперименте
    c = pur[0]
    i = make_instance(c["spec"], 3)
    for k in np.unique(i.cluster_id):
        assert len(np.unique(i.task_type[i.cluster_id == k])) == 1
    m = next(x for x in main_plan() if x["level"] == c["level"] and x["M"] == 8 and x["rho"] == 6
             and x["start"] == c["start"] and x["hetero"])
    j = make_instance(m["spec"], 3)
    assert np.allclose(i.points, j.points) and (i.capabilities == j.capabilities).all()
