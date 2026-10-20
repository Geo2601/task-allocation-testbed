"""Качество эталона основного эксперимента (таблица для статьи).

Для N ≤ 24 (точный эталон): доля примеров, где доказан минимум S, минимум Λ и все три минимума F_w;
у недоказанных минимумов Λ — разрыв до нижней оценки (только облегчённый эталон: там задача на минимум Λ
решается одна, и её оценка относится именно к Λ). Для N = 48: доля примеров с доказанным S* и
отставание лучшего известного S от доказанного (по примерам, где S* уточнялся).

Запуск:  python -m experiments.reference_quality --out results/experiment
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _group(obj):
    if obj in ("S", "lex_S"):
        return "S"
    if obj in ("Lambda", "lex_Lambda"):
        return "Lam"
    if obj == "W":
        return "W"
    return "eps"


def collect(out):
    rows = []
    for f in sorted(Path(out).glob("*.json")):
        rec = json.loads(f.read_text())
        ref = rec["reference"]
        row = dict(key=rec["key"], prefix=rec["prefix"], N=rec["N"], start=rec["start"], hetero=rec["hetero"],
                   kind=ref["kind"], rep=rec["rep"])
        by = {}
        for s in ref["solves"]:
            by.setdefault(_group(s["objective"]), []).append(s)
        for g in ("S", "Lam", "W"):
            if g in by:
                row[f"{g}_proven"] = all(s["status"] == "OPTIMAL" for s in by[g])
        if "W" in by:
            row["W_share"] = np.mean([s["status"] == "OPTIMAL" for s in by["W"]])
        lam = [s for s in by.get("Lam", []) if s["objective"] == "Lambda"]
        if lam and lam[0]["status"] != "OPTIMAL" and lam[0]["bound"]:
            row["Lam_gap"] = lam[0]["Lam"] / lam[0]["bound"] - 1
        if rec["N"] > 24:
            sS = [s for s in ref["solves"] if s["objective"] == "S" and s["status"] == "OPTIMAL"]
            if sS:
                heur = [s["S"] for s in ref["solves"] if s["objective"] != "S" and s["S"] is not None]
                row["bk_S_over_opt"] = min(heur) / sS[0]["S"] - 1 if heur else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/experiment")
    args = ap.parse_args()
    d = collect(args.out)
    for c in ("S_proven", "Lam_proven", "W_proven", "W_share", "Lam_gap", "bk_S_over_opt"):
        if c not in d:
            d[c] = np.nan
    ex = d[d.N <= 24]
    if len(ex):
        _exact(ex)
    _n48(d[d.N == 48])
    d.to_csv(Path(args.out) / "reference_quality.csv", index=False)


def _exact(ex):
    print("Точный эталон (N ≤ 24), доля доказанных:")
    cols = ["S_proven", "Lam_proven", "W_proven", "W_share"]
    t = ex.groupby(["N", "start", "hetero"])[cols].mean().round(3)
    print(t.to_string())
    print(ex.groupby("N")[cols].mean().round(3).to_string())
    print("(W_proven — доказаны все три минимума F_w; W_share — доля доказанных задач F_w)")
    g = ex.Lam_gap.dropna()
    if len(g):
        print(f"Недоказанные минимумы Λ (облегчённый эталон, {len(g)}): разрыв до оценки — медиана "
              f"{g.median():.3f}, 90 % — {g.quantile(0.9):.3f}, максимум {g.max():.3f}")


def _n48(n48):
    if not len(n48):
        return
    att = n48[n48.rep < 5]
    print(f"\nN = 48: S* доказан в {n48.S_proven.fillna(False).sum()} из {len(n48)} примеров "
          f"(подвыборка повторов 0–4: {att.S_proven.fillna(False).mean():.2f})")
    b = n48.bk_S_over_opt.dropna()
    if len(b):
        print(f"Лучшее известное S против доказанного S* ({len(b)}): медиана {b.median():.4f}, "
              f"90 % — {b.quantile(0.9):.4f}, максимум {b.max():.4f}, совпадение (≤ 0,1 %) — "
              f"{(b <= 0.001).mean():.2f}")


if __name__ == "__main__":
    main()
