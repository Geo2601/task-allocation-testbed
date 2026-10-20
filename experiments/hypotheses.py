"""Проверка гипотез Г1–Г5 (статья, разделы 4–5) по таблице основного эксперимента.

Г1: выигрыш методов с кластеризацией над CBBA растёт со скученностью. Проверка: для каждого
    предпочтения, старта и типа группы — корреляция Спирмена разности разрывов (метод − CBBA)
    со скученностью R; медианы разности по уровням. Г1 верна, если корреляция положительна
    (чем равномернее поле, тем хуже метод относительно CBBA).
Г2: выигрыш GACA над CBBA на равномерных полях при малом ρ — для обеих настроек CBBA.
Г3: метод СКФУ проигрывает k-means тем сильнее, чем выше скученность (время, общая база).
Г4: разнотипность уменьшает выигрыш кластеризации (k-means − CBBA, однородная против разнотипной).
Г5: при общей базе экономящие путь методы проигрывают по времени; число простаивающих аппаратов.

Запуск:  python -m experiments.hypotheses [--src data/all.csv.gz] [--out results/hypotheses.txt]
Результат: печать и текстовый файл.
"""
import argparse
import io
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from allocsim.stats import holm, paired_wilcoxon


def h1(d, inst, group_methods=("kmeans_hungarian", "skfu", "gaca"), prefix="main"):
    rows = []
    d = d[d.prefix == prefix]
    for pref in ("g_S", "g_Lam", "g_w50"):
        p = d.pivot_table(index="key", columns="method", values=pref)
        for meth in group_methods:
            diff = (p[meth] - p["cbba"]).rename("diff").to_frame().join(inst)
            for (st, het), g in diff.groupby(["start", "hetero"]):
                rho, pv = spearmanr(g.R, g["diff"])
                med = g.groupby("level")["diff"].median()
                rows.append(dict(показатель=pref, метод=meth, старт=st, разнотипная=het, n=len(g),
                                 спирмен=round(rho, 2), p=pv, L0=round(med.get("L0", float("nan")), 3),
                                 L2=round(med.get("L2", float("nan")), 3),
                                 L4=round(med.get("L4", float("nan")), 3)))
    r = pd.DataFrame(rows)
    r["p_холм"] = holm(r.p.to_numpy())
    r["p"] = r.p.map(lambda x: f"{x:.1e}")
    r["p_холм"] = r.p_холм.map(lambda x: f"{x:.1e}")
    return r


def h1_checks(d, inst, group_methods=("kmeans_hungarian", "gaca")):
    """Г1 на проверках устойчивости: та же корреляция, по каждой проверке целиком."""
    rows = []
    for prefix in sorted(set(inst.prefix) - {"main"}):
        dd = d[d.prefix == prefix]
        for pref in ("g_S", "g_Lam", "g_w50"):
            p = dd.pivot_table(index="key", columns="method", values=pref)
            for meth in group_methods:
                diff = (p[meth] - p["cbba"]).rename("diff").to_frame().join(inst)
                for st, g in diff.groupby("start"):
                    rho, pv = spearmanr(g.R, g["diff"])
                    rows.append(dict(проверка=prefix, показатель=pref, метод=meth, старт=st, n=len(g),
                                     спирмен=round(rho, 2), p=pv))
    if not rows:
        return pd.DataFrame()
    r = pd.DataFrame(rows)
    r["p_холм"] = holm(r.p.to_numpy())
    r["p"] = r.p.map(lambda x: f"{x:.1e}")
    r["p_холм"] = r.p_холм.map(lambda x: f"{x:.1e}")
    return r


def h4_purity(d, inst):
    """Г4 напрямую: разнотипная группа, типы перемешаны в скоплениях (основной эксперимент, M = 8, ρ = 6)
    против «в скоплении один тип» (проверка purity). Те же точки и умения, пары по (уровень, старт,
    повтор). Разность (k-means − CBBA): чем она меньше, тем больше выигрыш кластеризации."""
    if "purity" not in set(inst.prefix):
        return pd.DataFrame()
    rows = []
    for pref in ("g_S", "g_Lam", "g_w50"):
        p = d.pivot_table(index="key", columns="method", values=pref)
        diff = (p["kmeans_hungarian"] - p["cbba"]).rename("diff").to_frame().join(inst)
        mixed = diff[(diff.prefix == "main") & diff.hetero & (diff.M == 8) & (diff.rho == 6)
                     & (diff.level != "L0")]
        pure = diff[diff.prefix == "purity"]
        a = mixed.set_index(["level", "start", "rep"])["diff"]
        b = pure.set_index(["level", "start", "rep"])["diff"]
        both = pd.concat([a.rename("перемешаны"), b.rename("один_тип")], axis=1).dropna()
        for (lvl, st), g in both.groupby(level=["level", "start"]):
            w = paired_wilcoxon(g["один_тип"], g["перемешаны"])
            rows.append(dict(показатель=pref, уровень=lvl, старт=st, n=len(g),
                             перемешаны=round(g["перемешаны"].median(), 3),
                             один_тип=round(g["один_тип"].median(), 3),
                             разность=round(w["median_diff"], 3), p=w["p"]))
    r = pd.DataFrame(rows)
    r["p_холм"] = holm(r.p.to_numpy())
    r["p"] = r.p.map(lambda x: f"{x:.1e}")
    r["p_холм"] = r.p_холм.map(lambda x: f"{x:.1e}")
    return r


def h2(d, inst):
    rows = []
    for (M, rho), g in inst[(inst.start == "scattered") & (~inst.hetero) & (inst.level == "L0")
                            & (inst.prefix == "main")].groupby(["M", "rho"]):
        p = d[d.key.isin(g.index)].pivot_table(index="key", columns="method", values="g_S")
        w1 = paired_wilcoxon(p["gaca"], p["cbba"])
        row = dict(M=M, rho=rho, n=len(p), GACA=round(p.gaca.median(), 3), CBBA=round(p.cbba.median(), 3),
                   разность=round(w1["median_diff"], 3), p=f"{w1['p']:.1e}")
        if "cbba_tdr_marginal" in p:
            w2 = paired_wilcoxon(p["gaca"], p["cbba_tdr_marginal"])
            row.update(CBBA_авторов=round(p.cbba_tdr_marginal.median(), 3),
                       разность_с_авторским=round(w2["median_diff"], 3), p2=f"{w2['p']:.1e}")
        rows.append(row)
    return pd.DataFrame(rows)


def h3(d, inst):
    k = inst[(inst.start == "base") & (inst.prefix == "main")].index
    p = d[d.key.isin(k)].pivot_table(index="key", columns="method", values="g_Lam").join(inst["level"])
    out = p.groupby("level")[["skfu", "kmeans_hungarian"]].median().round(3)
    out["доля_СКФУ_лучше"] = p.assign(b=p.skfu < p.kmeans_hungarian).groupby("level").b.mean().round(2)
    return out


def h4(d, inst):
    p = d.pivot_table(index="key", columns="method", values="g_Lam")
    diff = (p["kmeans_hungarian"] - p["cbba"]).rename("diff").to_frame().join(inst)
    return diff[diff.prefix == "main"].pivot_table(index=["start", "level"], columns="hetero",
                                                   values="diff", aggfunc="median").round(3)


def h5(d, inst):
    m = d[d.method.isin(["cbba", "gaca", "kmeans_hungarian", "skfu", "greedy"]) & (d.prefix == "main")]
    return m.pivot_table(index=["start", "hetero"], columns="method",
                         values=["g_Lam", "idle"], aggfunc="median").round(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/all.csv.gz")
    ap.add_argument("--out", default="results/hypotheses.txt")
    args = ap.parse_args()
    d = pd.read_csv(args.src)
    inst = d.drop_duplicates("key").set_index("key")[["R", "start", "hetero", "level", "prefix", "M", "rho", "N",
                                                       "rep"]]
    buf = io.StringIO()
    with redirect_stdout(buf):
        print("Г1. Разность разрывов (метод − CBBA); спирмен < 0 — выигрыш метода над CBBA тем меньше, "
              "чем сильнее скопления (Г1 опровергается); > 0 — Г1 подтверждается.")
        print(h1(d, inst).to_string(index=False))
        r = h1_checks(d, inst)
        if len(r):
            print("\nГ1 на проверках устойчивости:")
            print(r.to_string(index=False))
        print("\nГ2. GACA против CBBA, путь, равномерное поле, рассредоточенный старт, однородная группа:")
        print(h2(d, inst).to_string(index=False))
        print("\nГ3. Время, общая база: медианы g_Λ метода СКФУ и k-means по уровням:")
        print(h3(d, inst).to_string())
        print("\nГ4. Время: медиана (k-means − CBBA) по старту и уровню, однородная / разнотипная группа:")
        print(h4(d, inst).to_string())
        r = h4_purity(d, inst)
        if len(r):
            print("\nГ4 напрямую: (k-means − CBBA), типы перемешаны в скоплениях / в скоплении один тип:")
            print(r.to_string(index=False))
        print("\nГ5. Медианы g_Λ и числа простаивающих аппаратов:")
        print(h5(d, inst).to_string())
    text = buf.getvalue()
    print(text)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
