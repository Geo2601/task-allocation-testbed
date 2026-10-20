"""Проверка прогона: пример сохраняется, повторный запуск его пропускает, таблица собирается."""
import json

from allocsim.instance import InstanceSpec, main_field
from allocsim.runner import RunConfig, collect, run_instance


def _cell(M=4, rho=3, start="base", het=False):
    spec = InstanceSpec(field=main_field(2, 1000.0, M, rho, "L4"), n_agents=M, start_mode=start,
                        heterogeneous=het)
    return dict(level="L4", M=M, rho=rho, start=start, hetero=het, spec=spec)


def test_run_instance_exact_and_collect(tmp_path):
    cfg = RunConfig(exact_max_n=12, exact_time=20, bk_time=1, n_inner=2, workers=2)
    p = run_instance(_cell(), 0, cfg, tmp_path, "main")
    assert p is not None and p.exists()
    rec = json.loads(p.read_text())
    assert rec["reference"]["kind"] == "exact"
    assert {"greedy", "cbba", "kmeans_hungarian", "gaca", "skfu", "cbba_tdr"} <= set(rec["methods"])
    m = rec["methods"]["cbba"]
    assert m["g_S"] >= -1e-6 and m["g_Lam"] >= -1e-6 and m["g_w50"] >= -1e-6
    # повторный запуск пропускает готовый пример
    assert run_instance(_cell(), 0, cfg, tmp_path, "main") is None
    d = collect(tmp_path)
    assert len(d) == len(rec["methods"]) and set(d.method) == set(rec["methods"])


def test_run_instance_best_known(tmp_path):
    cfg = RunConfig(exact_max_n=12, exact_s_max_n=12, bk_time=1, n_inner=1, extra=False)
    p = run_instance(_cell(M=4, rho=6, start="scattered", het=True), 1, cfg, tmp_path, "main")
    rec = json.loads(p.read_text())
    assert rec["reference"]["kind"] == "best_known"
    assert set(rec["methods"]) == {"greedy", "cbba", "kmeans_hungarian", "gaca", "skfu"}


def test_run_instance_best_known_with_exact_S(tmp_path):
    cfg = RunConfig(exact_max_n=12, exact_s_max_n=48, exact_time=30, workers=2, bk_time=1,
                    n_inner=1, extra=False)
    # общая база, однородная группа: минимум S сводится к одному обходу и доказывается быстро
    p = run_instance(_cell(M=8, rho=6, start="base", het=False), 2, cfg, tmp_path, "main")
    rec = json.loads(p.read_text())
    assert rec["reference"]["kind"] == "best_known+exact_S"
    # минимум S на эталонном фронте не больше доказанного минимума
    sol = [x for x in rec["reference"]["solves"] if x["objective"] == "S"][0]
    assert min(s for s, _ in rec["reference"]["front"]) <= sol["S"] + 1e-6


def test_lean_reference_and_refine(tmp_path):
    from allocsim.runner import refine_exact_S
    cfg = RunConfig(exact_max_n=12, exact_time=20, bk_time=1, n_inner=2, workers=2,
                    full_front_reps=0, extra=False)
    p = run_instance(_cell(M=4, rho=3, start="scattered", het=True), 3, cfg, tmp_path, "main")
    rec = json.loads(p.read_text())
    assert rec["reference"]["kind"].endswith("_lean")
    objs = [x["objective"] for x in rec["reference"]["solves"]]
    assert objs.count("W") == 3 and "S|L<=T" not in objs and "S" in objs and "Lambda" in objs
    # эталон «лучшее известное» без S*, затем уточнение
    cfg2 = RunConfig(exact_max_n=12, exact_s_max_n=0, bk_time=1, n_inner=1, extra=False)
    p2 = run_instance(_cell(M=8, rho=3, start="base", het=False), 4, cfg2, tmp_path, "main")
    assert json.loads(p2.read_text())["reference"]["kind"] == "best_known"
    assert refine_exact_S(p2, time_limit=20, workers=2) == "OPTIMAL"
    rec2 = json.loads(p2.read_text())
    assert rec2["reference"]["kind"] == "best_known+exact_S"
    assert all(m["g_S"] >= -1e-6 for m in rec2["methods"].values())
    assert refine_exact_S(p2, time_limit=20, workers=2) is None


def test_update_methods_adds_and_rescores(tmp_path):
    from allocsim.runner import update_methods
    cfg = RunConfig(exact_max_n=12, exact_time=20, bk_time=1, n_inner=1, workers=2, extra=False)
    p = run_instance(_cell(M=4, rho=3, start="scattered", het=False), 5, cfg, tmp_path, "main")
    before = json.loads(p.read_text())
    assert "cbba_tdr_marginal" not in before["methods"]
    update_methods(p, ["gaca", "cbba_tdr_marginal"])
    after = json.loads(p.read_text())
    assert "cbba_tdr_marginal" in after["methods"]
    assert after["methods"]["cbba"]["S"] == before["methods"]["cbba"]["S"]
    assert all(m["g_S"] >= -1e-6 and m["g_w50"] >= -1e-6 for m in after["methods"].values())
