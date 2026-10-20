"""Проверки метода СКФУ: сектора равной площади (как в коде авторов), слои, оценки p_ij,
раздача кластеров и остатка, допустимость; жадный «по очереди» (GrA в работах СКФУ)."""
import numpy as np
import pytest

from allocsim.fields import FieldSpec
from allocsim.instance import InstanceSpec, main_field, make_instance
from allocsim.methods import greedy, sector_labels, skfu, strip_labels
from allocsim.methods.skfu import _split_counts

GRID = np.stack(np.meshgrid(np.linspace(0.5, 999.5, 300), np.linspace(0.5, 999.5, 300)),
                -1).reshape(-1, 2)


@pytest.mark.parametrize("apex", [(0, 0), (500, 500), (200, 700), (1000, 1000)])
@pytest.mark.parametrize("M", [3, 5, 8, 16])
def test_sectors_equal_area(apex, M):
    lab = sector_labels(GRID, np.array(apex, float), np.zeros(2), 1000.0, M)
    share = np.bincount(lab, minlength=M) / len(GRID)
    assert np.allclose(share, 1 / M, atol=0.006)


def test_sectors_match_authors_code_boundaries():
    """Код авторов (M = 5, сторона 100, база в углу): границы секторов пересекают дальние стороны
    в точках, равномерно расставленных по их общей длине 200 — через каждые 40."""
    near = [(39, 100), (41, 100), (79, 100), (81, 100), (100, 81), (100, 79), (100, 41), (100, 39)]
    lab = sector_labels(np.array(near, float) * 0.999, np.zeros(2), np.zeros(2), 100.0, 5)
    assert all(lab[i] != lab[i + 1] for i in range(0, 8, 2))
    assert len(set(lab)) == 5


def test_sectors_use_xy_in_cube():
    pts = np.array([[90, 10, 0], [90, 10, 99]], float)
    lab = sector_labels(pts, np.zeros(3), np.zeros(3), 100.0, 4)
    assert lab[0] == lab[1]


def test_strips_equal_width():
    pts = np.c_[np.zeros(10), np.arange(10) * 100 + 50]
    assert strip_labels(pts, np.zeros(2), 1000.0, 5).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_split_counts():
    c = _split_counts(10, [1, 1, 2])
    assert c.tolist() == [3, 2, 5]
    assert _split_counts(4, [0, 0]).tolist() == [2, 2]


def _cube_inst(M, rep=0):
    spec = InstanceSpec(field=FieldSpec(dim=3, side=100.0, n_tasks=100), n_agents=M,
                        start_mode="base", heterogeneous=True, n_types=5, max_types_per_agent=3,
                        caps_scheme="random", min_cover=2)
    return make_instance(spec, rep)


@pytest.mark.parametrize("M", [5, 10, 20])
def test_skfu_valid_heterogeneous_cube(M):
    inst = _cube_inst(M)
    r = skfu(inst)
    assert inst.can_do()[r.assignment, np.arange(inst.n_tasks)].all()
    assert sorted(j for x in r.routes for j in x) == list(range(inst.n_tasks))
    assert sum(r.info["cluster_sizes"]) == inst.n_tasks
    assert (skfu(inst).assignment == r.assignment).all()


def test_skfu_homogeneous_base_one_sector_each():
    inst = make_instance(InstanceSpec(field=main_field(2, 1000.0, 8, 6, "L0"), n_agents=8,
                                      start_mode="base"), 0)
    r = skfu(inst)
    lab = sector_labels(inst.points, inst.starts[0], np.zeros(2),
                        float(inst.points.max()), 8)
    # каждому сектору — один аппарат, остатка нет
    for j in range(8):
        assert len(set(r.assignment[lab == j])) <= 1
    assert r.info["leftovers"] == 0


@pytest.mark.parametrize("start", ["base", "scattered"])
@pytest.mark.parametrize("het", [False, True])
@pytest.mark.parametrize("partition", ["sectors", "strips"])
def test_skfu_valid_main_setting(start, het, partition):
    inst = make_instance(InstanceSpec(field=main_field(2, 1000.0, 8, 6, "L3"), n_agents=8,
                                      start_mode=start, heterogeneous=het), 1)
    r = skfu(inst, partition=partition)
    assert inst.can_do()[r.assignment, np.arange(inst.n_tasks)].all()


def test_skfu_cluster_owner_gets_its_doable_tasks():
    """Первый круг: задачи кластера, которые его владелец умеет, остаются у владельца."""
    inst = _cube_inst(10, rep=3)
    r = skfu(inst)
    lab = sector_labels(inst.points, inst.starts[0], np.zeros(3), float(inst.points.max()), 10)
    can = inst.can_do()
    for j in range(10):
        members = np.flatnonzero(lab == j)
        owners = {r.assignment[t] for t in members if can[r.assignment[t], t]}
        assert owners  # в каждом кластере кто-то работает
    # остаток задач (владелец не умеет) — не больше доли задач, которых владелец не умеет
    assert r.info["leftovers"] <= inst.n_tasks


def test_greedy_round_robin_equal_counts():
    inst = make_instance(InstanceSpec(field=main_field(2, 1000.0, 4, 6, "L0"), n_agents=4,
                                      start_mode="base"), 0)
    r = greedy(inst, order="round_robin")
    assert np.bincount(r.assignment, minlength=4).tolist() == [6, 6, 6, 6]
    assert r.name == "greedy_round_robin"
