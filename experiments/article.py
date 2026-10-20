"""Воспроизведение таблиц 1–5 и рисунков 1–4 статьи и сверка чисел текста с пересчётом.

Всё, кроме табл. 3, считается из готовых данных в папке data/ (итог основного эксперимента
и проверок; как получить эти данные заново — в README). Табл. 3 пересчитывается с нуля: методы
прогоняются на тех же примерах, это несколько минут.

Запуск:  python -m experiments.article [--data data] [--out results/article] [--jobs 4]
         [--no-mechanism]   — табл. 3 взять из data/mechanism.csv, не пересчитывая

Результат в results/article/:
  table1_reference.csv … table5_gaca.csv   таблицы статьи;
  fig1_fields.png … fig4_bestmap.png       рисунки статьи;
  numbers.csv, report.md                   каждое число из текста рядом с пересчётом
                                           и отметкой о совпадении.
Код возврата 1, если хотя бы одно число не совпало.

Числа в «в_статье» записаны так, как они напечатаны (десятичная запятая, без единиц); пересчёт
округляется до того же знака.
"""
import argparse
import io
import json
import os
import re
import shutil
import sys
from contextlib import redirect_stdout
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from allocsim.analysis import cell_comparison, evaluate_policy, features, loss_matrix
from experiments import figures_main, hypotheses, mechanism
from experiments.skfu_check import PAPER_S

METHODS = ("greedy", "cbba", "kmeans_hungarian", "gaca", "skfu")
NAMES = {"greedy": "жадный", "cbba": "CBBA", "kmeans_hungarian": "k-means", "gaca": "GACA", "skfu": "секторы"}
LEVELS = ("L0", "L1", "L2", "L3", "L4")
STARTS = {"base": "общая база", "scattered": "рассредоточенный старт"}
def train_mask(inst):
    """Обучающая часть основного эксперимента — повторы 0–6 каждого десятка (70 %)."""
    return (inst["rep"] % 10) < 7


# ---------- форматирование и журнал сверки ----------

def f(x, nd=1):
    """Число как в статье: округление «половина вверх», десятичная запятая, знак минус «−»."""
    q = Decimal(repr(round(float(x), 10))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    if q == 0:
        q = abs(q)
    return f"{q:.{nd}f}".replace(".", ",").replace("-", "−")


def hround(x, nd):
    """Округление «половина вверх» для таблиц (как в тексте статьи); NaN остаётся NaN."""
    return x if pd.isna(x) else float(f(x, nd).replace(",", ".").replace("−", "-"))


def pct(x, nd=0):
    return f(100 * x, nd)


def rng(a, b, nd=(0, 0), scale=1.0):
    """Диапазон «a–b» как в статье; если концы после округления равны — одно число."""
    nd = nd if isinstance(nd, tuple) else (nd, nd)
    lo, hi = f(scale * a, nd[0]), f(scale * b, nd[1])
    return lo if lo == hi else f"{lo}–{hi}"


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, sec, what, article, computed, note="", tol=None):
        """tol — допустимое расхождение каждого числа (для строк таблиц, объединяющих два случая)."""
        computed = str(computed)
        ok = _norm(article) == _norm(computed)
        if not ok and tol is not None:
            a, c = _nums(article), _nums(computed)
            ok = len(a) == len(c) and all(abs(x - y) <= tol + 1e-9 for x, y in zip(a, c))
        self.rows.append(dict(раздел=sec, что=what, в_статье=article, пересчёт=computed, совпадает=ok,
                              примечание=note))

    def frame(self):
        return pd.DataFrame(self.rows)


def _nums(s):
    return [float(x.replace("−", "-").replace(",", ".")) for x in re.findall(r"[−-]?\d+(?:,\d+)?", str(s))]


def _norm(s):
    return str(s).replace("−", "-").replace(" ", "").replace(" ", "").replace("\xa0", "")


def yes(b):
    return "да" if bool(b) else "нет"


# ---------- данные ----------

def load(data):
    d = pd.read_csv(data / "all.csv.gz")
    inst = d.drop_duplicates("key").set_index("key")[["R", "start", "hetero", "level", "prefix", "M", "rho",
                                                      "N", "rep"]]
    return d, inst


# ---------- аннотация и раздел 4 ----------

def sec_plan(d, inst, C):
    m = inst[inst.prefix == "main"]
    C.add("Аннотация", "всего тестовых примеров", "12 500", f"{len(inst):,}".replace(",", " "))
    C.add("Аннотация", "число аппаратов", "4–16", f"{m.M.min()}–{m.M.max()}")
    C.add("Аннотация", "число задач", "12–192", f"{m.N.min()}–{m.N.max()}")
    C.add("5.4", "ячеек основного эксперимента", "180",
          str(m.groupby(["level", "M", "rho", "start", "hetero"]).ngroups))
    C.add("Рис. 3", "примеров на уровень и режим старта", "900",
          str(int(m.groupby(["level", "start"]).size().median())))
    for p in ("volume", "center", "clusters_half", "clusters_double", "purity"):
        C.add("4", f"примеров в проверке {p}", {"volume": "1000", "center": "500", "clusters_half": "800",
                                                 "clusters_double": "800", "purity": "400"}[p],
              str(int((inst.prefix == p).sum())))


# ---------- раздел 2: эталон, табл. 1 ----------

def sec_reference(data, out, C):
    k = pd.read_csv(data / "exact_limits.csv")
    C.add("2", "примеров в пробном расчёте предела точного решения", "65",
          str(len(k.drop_duplicates(["start", "hetero", "level", "M", "rho", "N", "rep"]))))
    s = k[k.objective == "S"]
    C.add("2", "минимум S доказан во всех случаях при N ≤ 48", "да", yes((s.status == "OPTIMAL").all()))
    lam = k[k.objective == "Lambda"]
    l24, l48 = lam[lam.N == 24], lam[lam.N == 48]
    C.add("2", "минимум Λ при N = 24 доказан за 60 с, %", "91",
          pct(((l24.status == "OPTIMAL") & (l24.wall <= 60)).mean()))
    C.add("2", "минимум Λ при N = 48 доказан за 600 с, %", "41", pct((l48.status == "OPTIMAL").mean()))

    q = pd.read_csv(data / "reference_quality.csv.gz")
    proven = lambda s: s.eq(True).mean()   # noqa: E731
    m = q[q.prefix == "main"]
    rows = []
    for N in (12, 24):
        e = m[m.N == N]
        rows.append(dict(N=str(N), примеров=len(e), минимум_S=proven(e.S_proven), минимум_Λ=proven(e.Lam_proven),
                         все_три_F_w=proven(e.W_proven)))
    n48 = m[(m.N == 48) & (m.rep < 5)]          # подвыборка: повторы 0–4 каждой ячейки
    rows.append(dict(N="48 (подвыборка)", примеров=len(n48), минимум_S=proven(n48.S_proven)))
    t1 = pd.DataFrame(rows)
    for c in ("минимум_S", "минимум_Λ", "все_три_F_w"):
        t1[c] = (100 * t1[c]).map(lambda v: hround(v, 1))     # доля примеров, где минимум доказан, %
    t1.to_csv(out / "table1_reference.csv", index=False)
    art = {"12": ("1000", "100", "100", "100"), "24": ("2000", "99,3", "81,8", "67,2")}
    for r in rows[:2]:
        a = art[r["N"]]
        C.add("Табл. 1", f"N = {r['N']}: примеров", a[0], str(r["примеров"]))
        nd = 0 if r["N"] == "12" else 1
        C.add("Табл. 1", f"N = {r['N']}: минимум S доказан, %", a[1], pct(r["минимум_S"], nd))
        C.add("Табл. 1", f"N = {r['N']}: минимум Λ доказан, %", a[2], pct(r["минимум_Λ"], nd))
        C.add("Табл. 1", f"N = {r['N']}: все три минимума F_w доказаны, %", a[3], pct(r["все_три_F_w"], nd))
    C.add("Табл. 1", "N = 48 (подвыборка): примеров", "300", str(len(n48)))
    C.add("Табл. 1", "N = 48 (подвыборка): минимум S доказан, %", "68", pct(proven(n48.S_proven)))
    e24 = m[(m.N == 24) & (m.start == "base") & (~m.hetero.astype(bool))]
    C.add("2", "минимум Λ при N = 24, общая база, однородная группа, %", "66,8", pct(proven(e24.Lam_proven), 1))
    g = m[m.N <= 24].Lam_gap.dropna()
    C.add("2", "недоказанные Λ: медиана расхождения с нижней оценкой, %", "3,4", pct(g.median(), 1))
    C.add("2", "недоказанные Λ: 90 % случаев не более, %", "14,7", pct(g.quantile(0.9), 1))
    b = m[m.N == 48].bk_S_over_opt.dropna()
    C.add("2", "N = 48: лучшее известное S выше доказанного, медиана, %", "0", pct(b.median()))
    C.add("2", "N = 48: то же, 90 % примеров не более, %", "1,4", pct(b.quantile(0.9), 1))
    C.add("2", "N = 48: то же, худший случай, %", "5,3", pct(b.max(), 1))


# ---------- раздел 5.1: пример фронта, рис. 2 ----------

def sec_front(data, out, C):
    rec = json.loads((data / "front_example.json").read_text(encoding="utf-8"))
    front = rec["reference"].get("ref_front") or rec["reference"]["front"]
    S0, L0 = min(s for s, _ in front), min(l for _, l in front)
    g = {m: (rec["methods"][m]["S"] / S0 - 1, rec["methods"][m]["Lam"] / L0 - 1) for m in METHODS}
    C.add("5.1", "пример: M, N, уровень, старт, разнотипная", "4, 24, L0, base, да",
          f"{rec['M']}, {rec['N']}, {rec['level']}, {rec['start']}, {yes(rec['hetero'])}")
    C.add("5.1", "CBBA: путь равен минимуму", "да", yes(abs(g["cbba"][0]) < 1e-6))
    C.add("5.1", "CBBA: время больше минимума, %", "47", pct(g["cbba"][1]))
    C.add("5.1", "GACA: путь длиннее минимума, %", "18", pct(g["gaca"][0]))
    C.add("5.1", "GACA: время больше минимума, %", "12", pct(g["gaca"][1]))
    km, sk = g["kmeans_hungarian"], g["skfu"]
    C.add("5.1", "k-means и секторы: время больше минимума, %", "12–13",
          rng(min(km[1], sk[1]), max(km[1], sk[1]), 0, 100))
    C.add("5.1", "k-means и секторы: путь длиннее минимума, %", "43–50",
          rng(min(km[0], sk[0]), max(km[0], sk[0]), 0, 100))
    figures_main.fig_front(out, data / "front_example.json")
    shutil.move(out / "fig_front.png", out / "fig2_front.png")


# ---------- раздел 5.2: разрывы по уровням, табл. 2 ----------

def sec_levels(d, inst, out, C):
    m = d[(d.prefix == "main") & d.method.isin(METHODS)]
    med = {p: m.groupby(["start", "level", "method"])[p].median().unstack() for p in ("g_S", "g_Lam", "g_w50")}
    gS, gL, gW = med["g_S"], med["g_Lam"], med["g_w50"]
    C.add("5.2", "по пути на всех уровнях лучший CBBA", "да", yes((gS.idxmin(axis=1) == "cbba").all()))
    C.add("5.2", "медиана g_S у CBBA по уровням и стартам, %", "0,3–4",
          rng(gS.cbba.min(), gS.cbba.max(), (1, 0), 100))
    ratio = 1 + gS.loc["base", ["kmeans_hungarian", "skfu"]]
    C.add("5.2", "общая база: путь k-means и секторов длиннее минимума, раз", "2,3–3",
          rng(ratio.min().min(), ratio.max().max(), (1, 0)))
    b = m[(m.start == "base") & (~m.hetero.astype(bool))]
    act = (b.M - b.idle).groupby(b.method).median()
    share_idle = (b.idle / b.M).groupby(b.method).median()
    C.add("5.2", "Г5: общая база, однородная группа: CBBA работает аппаратов (медиана)", "1", f(act["cbba"], 0))
    C.add("5.2", "Г5: GACA оставляет без дела долю группы (медиана)", "0,75", f(share_idle["gaca"], 2),
          "в статье «три четверти группы»")
    bl = b.groupby("method").g_Lam.median()
    C.add("5.2", "Г5: g_Λ CBBA, GACA, k-means, секторы, %", "81, 51, 2, 1",
          ", ".join(pct(bl[x]) for x in ("cbba", "gaca", "kmeans_hungarian", "skfu")))
    C.add("5.2", "Г3: общая база, L0: g_Λ секторы против k-means, %", "3,3 против 10,4",
          f"{pct(gL.loc[('base', 'L0'), 'skfu'], 1)} против {pct(gL.loc[('base', 'L0'), 'kmeans_hungarian'], 1)}")
    C.add("5.2", "Г3: общая база, L4: g_Λ секторы против k-means, %", "3,1 против 1,8",
          f"{pct(gL.loc[('base', 'L4'), 'skfu'], 1)} против {pct(gL.loc[('base', 'L4'), 'kmeans_hungarian'], 1)}")
    C.add("5.2", "рассредоточенный старт: по времени лучший k-means на всех уровнях", "да",
          yes((gL.loc["scattered"].idxmin(axis=1) == "kmeans_hungarian").all()))
    auction = {"cbba", "gaca"}
    C.add("5.2", "равный вес, общая база: лучше аукционные алгоритмы на всех уровнях", "да",
          yes(gW.loc["base"].idxmin(axis=1).isin(auction).all()))
    C.add("5.2", "равный вес, общая база: g_0,5 CBBA на L0 и L4, %", "22 и 3",
          f"{pct(gW.loc[('base', 'L0'), 'cbba'])} и {pct(gW.loc[('base', 'L4'), 'cbba'])}")
    bs = gW.loc["scattered"].idxmin(axis=1)
    C.add("5.2", "равный вес, рассредоточенный: L0–L2 лучший k-means, L3–L4 аукционные", "да",
          yes((bs[["L0", "L1", "L2"]] == "kmeans_hungarian").all() and bs[["L3", "L4"]].isin(auction).all()))
    C.add("5.2", "равный вес, рассредоточенный, L4: GACA против k-means, %", "18 против 34",
          f"{pct(gW.loc[('scattered', 'L4'), 'gaca'])} против "
          f"{pct(gW.loc[('scattered', 'L4'), 'kmeans_hungarian'])}")

    h1 = hypotheses.h1(d, inst)
    h1.to_csv(out / "hypothesis1_all.csv", index=False)
    km = h1[h1.метод == "kmeans_hungarian"]
    C.add("5.2", "Г1: k-means − CBBA, корреляция отрицательна во всех 12 сочетаниях", "да",
          yes((km.спирмен < 0).all() and len(km) == 12))
    cell = lambda pref, st, het, lvl: km[(km.показатель == pref) & (km.старт == st)   # noqa: E731
                                         & (km.разнотипная == het)][lvl].iloc[0]
    C.add("5.2", "время, база, однородная: k-means лучше CBBA на L0 и L4, п. п.", "115 и 53",
          f"{pct(-cell('g_Lam', 'base', False, 'L0'))} и {pct(-cell('g_Lam', 'base', False, 'L4'))}")
    C.add("5.2", "равный вес, рассредоточенный, однородная: L0 лучше на, L4 хуже на, п. п.", "24 и 15",
          f"{pct(-cell('g_w50', 'scattered', False, 'L0'))} и {pct(cell('g_w50', 'scattered', False, 'L4'))}")
    sk = h1[h1.метод == "skfu"]
    p = sk.p_холм.astype(float)
    C.add("5.2", "секторы − CBBA: корреляция отрицательна и значима, сочетаний из 12", "11",
          str(int(((sk.спирмен < 0) & (p < 0.05)).sum())))
    ga = h1[(h1.метод == "gaca") & (h1.старт == "base") & h1.показатель.isin(["g_Lam", "g_w50"])]
    C.add("5.2", "GACA − CBBA при общей базе (время, равный вес): корреляция отрицательна", "да",
          yes((ga.спирмен < 0).all()))

    t2 = km.pivot_table(index="показатель", columns=["старт", "разнотипная"], values="спирмен")
    t2 = t2.loc[["g_S", "g_Lam", "g_w50"], [("base", False), ("base", True), ("scattered", False),
                                            ("scattered", True)]]
    t2.map(lambda v: hround(v, 2)).to_csv(out / "table2_spearman.csv")
    art = {"g_S": ("−0,26", "−0,11", "−0,56", "−0,15"), "g_Lam": ("−0,48", "−0,36", "−0,36", "−0,24"),
           "g_w50": ("−0,52", "−0,36", "−0,51", "−0,28")}
    cols = ("база, однородная", "база, разнотипная", "рассредоточенный, однородная", "рассредоточенный, разнотипная")
    for pref, vals in art.items():
        for c, a, v in zip(cols, vals, t2.loc[pref]):
            C.add("Табл. 2", f"{pref}, {c}", a, f(v, 2))
    C.add("Табл. 2", "примеров в каждом столбце", "2250", str(int(km.n.median())))
    C.add("Табл. 2", "все значения значимы: p после поправки Холма < 2·10⁻⁶", "да",
          yes((km.p_холм.astype(float) < 2e-6).all()))

    figures_main.fig_levels(out, d)
    shutil.move(out / "fig_levels.png", out / "fig3_levels.png")
    (out / "fig_levels_types.png").unlink(missing_ok=True)
    return h1


# ---------- раздел 5.3: механизм, табл. 3 ----------

def sec_mechanism(data, out, C, jobs, recompute):
    if recompute:
        med = mechanism.medians(mechanism.measure(mechanism.cells("main"), 50, jobs))
        pur = mechanism.medians(mechanism.measure(mechanism.cells("purity"), 50, jobs))
    else:
        med = pd.read_csv(data / "mechanism.csv", index_col=[0, 1, 2])
        pur = pd.read_csv(data / "mechanism_purity.csv", index_col=[0, 1, 2])
    med.to_csv(out / "mechanism.csv")          # полная точность: те же файлы лежат в data/
    pur.to_csv(out / "mechanism_purity.csv")
    rows = [("Целостность скоплений", "integrity", "base", "cbba", "— 1,00 1,00 1,00 1,00"),
            ("Целостность скоплений", "integrity", "scattered", "cbba", "— 0,81 0,98 1,00 1,00"),
            ("Целостность скоплений", "integrity", "scattered", "gaca", "— 0,77 0,95 1,00 1,00"),
            ("Целостность скоплений", "integrity", "base", "kmeans_hungarian", "— 0,71 0,88 0,98 1,00"),
            ("Целостность скоплений", "integrity", "scattered", "kmeans_hungarian", "— 0,71 0,88 0,98 1,00"),
            ("Целостность скоплений", "integrity", "base", "skfu", "— 0,60 0,75 0,87 0,94"),
            ("Целостность скоплений", "integrity", "scattered", "skfu", "— 0,60 0,75 0,87 0,94"),
            ("Неравномерность загрузки", "cv_len", "base", "kmeans_hungarian", "0,29 0,26 0,30 0,32 0,34"),
            ("Неравномерность загрузки", "cv_len", "scattered", "kmeans_hungarian", "0,31 0,32 0,32 0,37 0,43"),
            ("Неравномерность загрузки", "cv_len", "base", "skfu", "0,15 0,32 0,43 0,59 0,64"),
            ("Простаивающих из 8", "idle", "scattered", "cbba", "0 2 3 4 4")]
    table = []
    for title, col, st, meth, art in rows:
        vals = [med.loc[(st, meth, lvl), col] for lvl in LEVELS]
        nd = 0 if col == "idle" else 2
        comp = " ".join("—" if pd.isna(v) else f(v, nd) for v in vals)
        table.append(dict(величина=title, старт=STARTS[st], метод=NAMES[meth],
                          **{lvl: hround(v, nd) for lvl, v in zip(LEVELS, vals)}))
        both = meth in ("kmeans_hungarian", "skfu") and col == "integrity"
        C.add("Табл. 3", f"{title}, {STARTS[st]}, {NAMES[meth]}, L0–L4", art, comp,
              "в статье одна строка «оба» на два режима старта; допуск 0,01" if both else "",
              tol=0.01 if both else None)
    pd.DataFrame(table).to_csv(out / "table3_mechanism.csv", index=False)
    i = med.integrity
    C.add("5.3", "L2: целостность у CBBA не ниже, чем у k-means, при обоих стартах", "да",
          yes(all(i[(s, "cbba", "L2")] >= i[(s, "kmeans_hungarian", "L2")] for s in ("base", "scattered"))))
    strong = pur.integrity[pur.integrity.index.get_level_values("level").isin(["L3", "L4"])]
    pk = strong.xs("kmeans_hungarian", level="method")
    C.add("5.6", "«один тип в скоплении», L3–L4: целостность у k-means", "0,84–0,85",
          rng(pk.min(), pk.max(), 2), "медианы по уровню и старту, mechanism_purity.csv")
    pa = pd.concat([strong.xs(x, level="method") for x in ("cbba", "gaca")])
    C.add("5.6", "«один тип в скоплении», L3–L4: целостность у CBBA и GACA", "1,00",
          rng(pa.min(), pa.max(), 2), "медианы по уровню и старту, mechanism_purity.csv")


# ---------- раздел 5.4: лучшие методы по ячейкам, рис. 4 ----------

def sec_cells(d, out, C):
    m = d[d.prefix == "main"]
    res = {}
    for pref in ("g_S", "g_Lam", "g_w50"):
        c = cell_comparison(m, pref, METHODS)
        c.to_csv(out / f"cells_{pref}.csv", index=False)
        res[pref] = c
    s, l, w = res["g_S"], res["g_Lam"], res["g_w50"]
    cnt = lambda c: c.best.value_counts()   # noqa: E731
    C.add("5.4", "путь: лучший CBBA / GACA, ячеек", "141 / 39",
          f"{cnt(s).get('cbba', 0)} / {cnt(s).get('gaca', 0)}")
    gs = s[s.best == "gaca"]
    C.add("5.4", "путь: из ячеек GACA — разнотипная группа, рассредоточенный старт", "34",
          str(int(((gs.start == "scattered") & gs.hetero.astype(bool)).sum())))
    C.add("5.4", "время: лучший k-means / секторы / жадный, ячеек", "113 / 49 / 13",
          " / ".join(str(cnt(l).get(x, 0)) for x in ("kmeans_hungarian", "skfu", "greedy")))
    C.add("5.4", "время: секторы лучшие только при общей базе", "да", yes((l[l.best == "skfu"].start == "base").all()))
    C.add("5.4", "время: жадный лучший только в разнотипной группе", "да",
          yes(l[l.best == "greedy"].hetero.astype(bool).all()))
    C.add("5.4", "равный вес: лучший GACA / CBBA / k-means, ячеек", "76 / 52 / 51",
          " / ".join(str(cnt(w).get(x, 0)) for x in ("gaca", "cbba", "kmeans_hungarian")))
    sig = lambda c: (c.p_best_vs_second.astype(float) < 0.05).mean()   # noqa: E731
    C.add("5.4", "отрыв лучшего от второго значим: путь / время / равный вес, % ячеек", "69 / 58 / 42",
          f"{pct(sig(s))} / {pct(sig(l))} / {pct(sig(w))}")
    figures_main.fig_bestmap(out, d)
    shutil.move(out / "fig_bestmap.png", out / "fig4_bestmap.png")


# ---------- раздел 5.5: правило выбора, табл. 4; раздел 5.6: перенос на проверки ----------

def _leaves(node, path=()):
    if node.feature is None:
        return [(path, node.method)]
    return (_leaves(node.left, path + ((node.feature, "≤", node.threshold),))
            + _leaves(node.right, path + ((node.feature, ">", node.threshold),)))


def rule_text(tree):
    """Правило дерева в виде строки как в табл. 4: «условие → метод; …»."""
    out = []
    for path, meth in _leaves(tree.root):
        cond = [("общая база" if fe == "base" and op == ">" else "рассредоточенный старт" if fe == "base"
                 else f"R {op} {f(th, 2)}" if fe == "R" else f"{fe} {op} {th:g}") for fe, op, th in path]
        out.append(f"{', '.join(cond)} → {NAMES[meth]}" if cond else NAMES[meth])
    return "; ".join(out)


def sec_rule(d, out, C):
    prefs = ("g_S", "g_Lam", "g_w50", "g_w25", "g_w75")
    trees, regs, rows = {}, {}, []
    for pref in prefs:
        tree, reg = evaluate_policy(d, pref, METHODS, train_mask, max_depth=3, min_leaf=40, min_gain=0.005)
        trees[pref], regs[pref] = tree, reg.set_index("часть")
        held = regs[pref].loc["отложенные (основной эксперимент)"]
        const = held[[f"всегда {m}" for m in METHODS]].sort_values()
        rows.append(dict(предпочтение=pref, правило=rule_text(tree),
                         сожаление_правила=100 * held["дерево"], лучший_постоянный=const.index[0][7:],
                         его_сожаление=100 * const.iloc[0], второй_постоянный=const.index[1][7:],
                         сожаление_второго=100 * const.iloc[1], примеров=int(held["примеров"])))
        with open(out / f"rule_{pref}.txt", "w", encoding="utf-8") as fh:
            fh.write(f"{tree.rules()}\n\n{reg.round(4).to_string(index=False)}\n")
    t4 = pd.DataFrame(rows).set_index("предпочтение")
    t4.map(lambda v: hround(v, 1) if isinstance(v, float) else v).to_csv(out / "table4_rule.csv")
    C.add("Табл. 4", "отложенных примеров", "2700", str(int(t4.примеров.iloc[0])))
    art = {"g_S": ("2,0", "cbba", None), "g_Lam": ("7,7", "kmeans_hungarian", None),
           "g_w50": ("5,4", "gaca", "9,0"), "g_w25": ("8,1", "kmeans_hungarian", "13,0"),
           "g_w75": ("3,4", "cbba", "4,0")}
    for pref, (rg, best, best_rg) in art.items():
        r = t4.loc[pref]
        C.add("Табл. 4", f"{pref}: сожаление правила, п. п.", rg, f(r.сожаление_правила, 1))
        C.add("Табл. 4", f"{pref}: лучший постоянный выбор", NAMES[best], NAMES[r.лучший_постоянный])
        if best_rg:
            C.add("Табл. 4", f"{pref}: сожаление лучшего постоянного выбора, п. п.", best_rg, f(r.его_сожаление, 1))
    for pref, meth in (("g_S", "cbba"), ("g_Lam", "kmeans_hungarian")):
        C.add("Табл. 4", f"{pref}: правило — один метод", NAMES[meth],
              ", ".join(sorted({NAMES[mm] for _, mm in _leaves(trees[pref].root)})))
    r = t4.loc["g_Lam"]
    C.add("Табл. 4", "g_Lam: следующий постоянный выбор и его сожаление", "секторы, 36,4",
          f"{NAMES[r.второй_постоянный]}, {f(r.сожаление_второго, 1)}")

    def leaves_text(pref):
        return rule_text(trees[pref])
    C.add("Табл. 4", "g_w50: правило",
          "рассредоточенный старт, R ≤ 0,49 → GACA; рассредоточенный старт, R > 0,49 → k-means; "
          "общая база, R ≤ 0,43 → CBBA; общая база, R > 0,43 → GACA", leaves_text("g_w50"))
    C.add("Табл. 4", "g_w25: правило",
          "рассредоточенный старт → k-means; общая база, R ≤ 0,82 → GACA; общая база, R > 0,82 → k-means",
          leaves_text("g_w25"))
    C.add("Табл. 4", "g_w75: правило", "рассредоточенный старт → GACA; общая база → CBBA", leaves_text("g_w75"))
    w = t4.loc["g_w50"]
    C.add("5.5", "равный вес: правило лучше постоянного GACA, раз", "1,7",
          f(w.его_сожаление / w.сожаление_правила, 1))

    # сожаление правила для равного веса по размеру задачи (отложенные примеры)
    G = loss_matrix(d, "g_w50", METHODS)
    inst = d.drop_duplicates("key").set_index("key").loc[G.index]
    held = (inst.prefix == "main") & ~train_mask(inst)
    X = features(inst)
    choice = pd.Series(trees["g_w50"].predict(X[held]), index=G[held].index)
    vals = G[held]
    reg = vals.to_numpy()[np.arange(len(vals)), [vals.columns.get_loc(c) for c in choice]] - vals.min(axis=1)
    N = inst.loc[held, "N"]
    C.add("5.5", "равный вес: сожаление правила при N ≤ 24 / N = 48 / N ≥ 96, п. п.", "4,9 / 5,9 / 5,5",
          " / ".join(f(100 * reg[mask].mean(), 1) for mask in (N <= 24, N == 48, N >= 96)))

    # 5.6: перенос правил на проверки устойчивости
    checks = ("volume", "center", "clusters_half", "clusters_double", "purity")
    part = lambda pref, p: regs[pref].loc[f"проверка: {p}"]   # noqa: E731

    def best_const(pref, p):
        c = part(pref, p)[[f"всегда {m}" for m in METHODS]].astype(float)
        return c.idxmin()[7:], c.min()
    for pref, meth, art_rng, nd in (("g_S", "cbba", "0,4–2,6", (1, 1)), ("g_Lam", "kmeans_hungarian", "5,9–17", (1, 0))):
        rg = [part(pref, p)["дерево"] for p in checks]
        C.add("5.6", f"{pref}: сожаление правила на проверках, п. п.", art_rng, rng(min(rg), max(rg), nd, 100))
        C.add("5.6", f"{pref}: правило — лучший постоянный выбор в каждой проверке", "да",
              yes(all(best_const(pref, p)[0] == meth for p in checks)))
    tw = lambda p: part("g_w50", p)["дерево"]   # noqa: E731
    C.add("5.6", "g_w50, объём: правило против лучшего постоянного, п. п.", "9,6 против 10,9",
          f"{pct(tw('volume'), 1)} против {pct(best_const('g_w50', 'volume')[1], 1)}")
    C.add("5.6", "g_w50, другое число скоплений (m = M/2 и m = 2M): правило, п. п.", "4,6 и 5,9",
          f"{pct(tw('clusters_half'), 1)} и {pct(tw('clusters_double'), 1)}")
    C.add("5.6", "g_w50, другое число скоплений (m = M/2 и m = 2M): лучший постоянный, п. п.", "7,0 и 9,0",
          f"{pct(best_const('g_w50', 'clusters_half')[1], 1)} и {pct(best_const('g_w50', 'clusters_double')[1], 1)}")
    bp = best_const("g_w50", "purity")
    C.add("5.6", "g_w50, «один тип в скоплении»: правило против постоянного GACA, п. п.", "6,7 против 5,1",
          f"{pct(tw('purity'), 1)} против {pct(part('g_w50', 'purity')['всегда gaca'], 1)}")
    C.add("5.6", "g_w50, «один тип в скоплении»: лучший постоянный", "GACA", NAMES[bp[0]])
    C.add("5.6", "g_w50, база в центре: правило против постоянного GACA, п. п.", "18,7 против 3,4",
          f"{pct(tw('center'), 1)} против {pct(part('g_w50', 'center')['всегда gaca'], 1)}")
    C.add("5.6", "g_w25, объём: лучше всегда k-means", "да",
          yes(part("g_w25", "volume")["всегда kmeans_hungarian"] < part("g_w25", "volume")["дерево"]))
    C.add("5.6", "g_w75, база в центре: лучше всегда GACA", "да",
          yes(part("g_w75", "center")["всегда gaca"] < part("g_w75", "center")["дерево"]))
    C.add("Аннотация", "равный вес: сожаление правила и постоянного GACA, п. п.", "5,4 и 9,0",
          f"{f(w.сожаление_правила, 1)} и {f(w.его_сожаление, 1)}")


# ---------- раздел 5.6: Г1 и Г4 на проверках ----------

def sec_checks(d, inst, out, C):
    r = hypotheses.h1_checks(d, inst)
    r.to_csv(out / "hypothesis1_checks.csv", index=False)
    km = r[r.метод == "kmeans_hungarian"]
    C.add("5.6", "Г1 на проверках: сочетаний (k-means − CBBA)", "27", str(len(km)))
    C.add("5.6", "Г1 на проверках: корреляция отрицательна во всех", "да", yes((km.спирмен < 0).all()))
    C.add("5.6", "Г1 на проверках: значима после поправки Холма", "25", str(int((km.p_холм.astype(float) < 0.05).sum())))
    m = d[d.method.isin(METHODS)]
    half = m[(m.prefix == "clusters_half") & (m.start == "scattered") & (m.level == "L4")]
    main = m[(m.prefix == "main") & (m.M == 8) & (m.rho == 6) & (m.start == "scattered") & (m.level == "L4")]
    gl = lambda dd, meth, p: dd[dd.method == meth][p].median()   # noqa: E731
    C.add("5.6", "m = M/2, рассредоточенный, L4: g_Λ k-means при m = M/2 и m = M, %", "53 и 33",
          f"{pct(gl(half, 'kmeans_hungarian', 'g_Lam'))} и {pct(gl(main, 'kmeans_hungarian', 'g_Lam'))}")
    C.add("5.6", "m = M/2, рассредоточенный, L4: g_0,5 k-means и CBBA, %", "60 и 14",
          f"{pct(gl(half, 'kmeans_hungarian', 'g_w50'))} и {pct(gl(half, 'cbba', 'g_w50'))}")
    h4 = hypotheses.h4(d, inst)
    C.add("5.6", "Г4: время, база, L0: k-means лучше CBBA, разнотипная и однородная, п. п.", "56 и 115",
          f"{pct(-h4.loc[('base', 'L0'), True])} и {pct(-h4.loc[('base', 'L0'), False])}")
    p = hypotheses.h4_purity(d, inst)
    p.to_csv(out / "hypothesis4_purity.csv", index=False)
    pv = p.p_холм.astype(float)
    s = p[p.показатель == "g_S"]
    C.add("5.6", "Г4 напрямую, путь: преимущество сократилось / значимо, ячеек из 8", "8 / 7",
          f"{int((s.разность > 0).sum())} / {int(((s.разность > 0) & (pv[s.index] < 0.05)).sum())}")
    w = p[p.показатель == "g_w50"]
    C.add("5.6", "Г4 напрямую, равный вес: преимущество сократилось, ячеек из 8", "8", str(int((w.разность > 0).sum())))
    lam = p[(p.показатель == "g_Lam") & (pv < 0.05) & (p.разность > 0)]
    C.add("5.6", "Г4 напрямую, время: значимо только при рассредоточенном старте на L3–L4", "да",
          yes(set(zip(lam.уровень, lam.старт)) == {("L3", "scattered"), ("L4", "scattered")}))


# ---------- раздел 5.7: сверки в условиях авторов, табл. 5 ----------

def sec_authors(d, inst, data, out, C):
    h2 = hypotheses.h2(d, inst)
    h2.to_csv(out / "hypothesis2_main.csv", index=False)
    C.add("5.7", "основной эксперимент: GACA и CBBA значимо не различаются (p > 0,05) ни в одной ячейке", "да",
          yes((h2.p.astype(float) > 0.05).all()), "по ячейкам M × ρ, равномерное поле, рассредоточенный старт, "
                                                  "однородная группа")
    C.add("5.7", "основной эксперимент: CBBA в настройке [9] уступает GACA по g_S, п. п.", "15–24",
          rng(-h2.разность_с_авторским.max(), -h2.разность_с_авторским.min(), 0, 100))

    k = pd.read_csv(data / "gaca_check.csv.gz")
    t5 = 100 * k.pivot_table(index="method", columns="M", values="opt_raw", aggfunc="median")
    t5 = t5.loc[["gaca", "cbba_tdr_marginal_author", "cbba"]]
    t5.map(lambda v: hround(v, 1)).to_csv(out / "table5_gaca.csv")
    art = {"gaca": "97,3 97,0 97,0 96,5", "cbba_tdr_marginal_author": "83,4 81,5 79,9 81,2",
           "cbba": "98,4 97,3 97,3 97,3"}
    lab = {"gaca": "GACA", "cbba_tdr_marginal_author": "CBBA в настройке [9]", "cbba": "CBBA со ставкой по приросту пути"}
    for meth, a in art.items():
        C.add("Табл. 5", f"{lab[meth]}, M = 5, 10, 15, 20", a, " ".join(f(v, 1) for v in t5.loc[meth]))
    C.add("Табл. 5", "полей на каждое число аппаратов", "100", str(int(k.groupby(["M", "method"]).size().median())))
    pp = k.pivot_table(index=["M", "rep"], columns="method", values="opt_raw")
    diff = 100 * (pp.gaca - pp.cbba_tdr_marginal_author).groupby("M").median()
    C.add("5.7", "разница GACA и CBBA в настройке [9], п. п.", "13–17", rng(diff.min(), diff.max(), 0),
          "медиана парной разности по полям при каждом M")

    s = pd.read_csv(data / "skfu_check.csv")
    p = s.pivot_table(index=["M", "rep"], columns="method", values="S_closed")
    gain = lambda g: (1 - p.skfu / p[g]).groupby("M").median()   # noqa: E731
    a, b = gain("greedy"), gain("greedy_round_robin")
    C.add("5.7", "секторы короче жадного (общий решатель маршрутов), %", "−0,3–2,1", rng(a.min(), a.max(), 1, 100),
          "медиана по картам; путь с возвратом на базу")
    C.add("5.7", "секторы короче жадного «по очереди», %", "0,9–2,3", rng(b.min(), b.max(), 1, 100),
          "медиана по картам; путь с возвратом на базу")
    mm = s.groupby(["M", "method"]).mean(numeric_only=True)
    ours = np.array([1 - mm.loc[(M, "skfu")].S_open / PAPER_S["DFCM"][M] for M in (5, 10, 15, 20)])
    C.add("5.7", "путь секторов короче, чем в [7, 8], %", "14–21", rng(ours.min(), ours.max(), 0, 100),
          "средние по картам, путь без возврата, как в [7, 8]")
    gr = np.array([PAPER_S["GrA"][M] / mm.loc[(M, "greedy")].S_raw_open for M in (5, 10, 15, 20)])
    C.add("5.7", "путь жадного в [7, 8] длиннее нашего, раз", "1,6–1,9", rng(gr.min(), gr.max(), 1),
          "средние по картам, путь без возврата, маршруты в порядке выбора задач")


# ---------- рисунок 1 ----------

def fig_fields(out):
    figures_main.fig_fields(out)
    shutil.move(out / "fig_fields.png", out / "fig1_fields.png")


def report(C, out):
    t = C.frame()
    t.to_csv(out / "numbers.csv", index=False)
    bad = t[~t.совпадает]
    lines = ["# Сверка чисел статьи с пересчётом", "",
             f"Проверено чисел и утверждений: {len(t)}; совпало: {int(t.совпадает.sum())}; "
             f"не совпало: {len(bad)}.", ""]
    if len(bad):
        lines += ["## Не совпало", "", "| Раздел | Что | В статье | Пересчёт | Примечание |", "|---|---|---|---|---|"]
        lines += [f"| {r.раздел} | {r.что} | {r.в_статье} | {r.пересчёт} | {r.примечание} |" for r in bad.itertuples()]
        lines.append("")
    lines += ["## Все проверки", "", "| Раздел | Что | В статье | Пересчёт | ✓ | Примечание |", "|---|---|---|---|---|---|"]
    lines += [f"| {r.раздел} | {r.что} | {r.в_статье} | {r.пересчёт} | {'✓' if r.совпадает else '✗'} | {r.примечание} |"
              for r in t.itertuples()]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/article")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--no-mechanism", action="store_true", help="табл. 3 из data/, без пересчёта")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    C = Checks()
    d, inst = load(data)
    steps = [("план эксперимента", lambda: sec_plan(d, inst, C)),
             ("табл. 1, раздел 2", lambda: sec_reference(data, out, C)),
             ("рис. 1", lambda: fig_fields(out)),
             ("рис. 2, раздел 5.1", lambda: sec_front(data, out, C)),
             ("рис. 3, табл. 2, раздел 5.2", lambda: sec_levels(d, inst, out, C)),
             ("табл. 3, раздел 5.3", lambda: sec_mechanism(data, out, C, args.jobs, not args.no_mechanism)),
             ("рис. 4, раздел 5.4", lambda: sec_cells(d, out, C)),
             ("табл. 4, разделы 5.5–5.6", lambda: sec_rule(d, out, C)),
             ("раздел 5.6", lambda: sec_checks(d, inst, out, C)),
             ("табл. 5, раздел 5.7", lambda: sec_authors(d, inst, data, out, C))]
    for name, fn in steps:
        print(f"… {name}", flush=True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            fn()
    t = report(C, out)
    bad = t[~t.совпадает]
    print(f"\nПроверено {len(t)}, совпало {int(t.совпадает.sum())}, не совпало {len(bad)}.")
    if len(bad):
        print(bad[["раздел", "что", "в_статье", "пересчёт"]].to_string(index=False))
    print("→", out)
    return 1 if len(bad) else 0


if __name__ == "__main__":
    sys.exit(main())
