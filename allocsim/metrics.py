"""Показатели качества метода относительно эталона (статья, раздел 1).

Три предпочтения (приняты 11.10.2026):
- важен суммарный путь — g_S = S/S° − 1;
- важно время выполнения — g_Λ = Λ/Λ° − 1;
- важно и то и другое поровну — g_w50 = F_0,5 / F_0,5° − 1,
где F_w = w·S/S° + (1 − w)·Λ/Λ°, S° и Λ° — минимумы по эталону, F_w° — минимум F_w по эталону.
Проверка чувствительности к весу — g_w25 и g_w75. Дополнительно: ε-индикатор до фронта и число
простаивающих аппаратов (без них «всё одному аппарату» выглядит идеальным по ε).
"""
from .exact import WEIGHTS, eps_indicator, nondominated, weighted_value


def _wkey(w):
    return f"g_w{int(round(100 * w)):02d}"


def gaps(result, S_ref, Lam_ref, front, weights=WEIGHTS):
    """g_S, g_Λ — разрывы до эталонных минимумов; g_wXX — разрыв по F_w; ε — до фронта."""
    norm = (S_ref, Lam_ref)
    out = dict(
        S=result.S, Lam=result.Lam,
        g_S=result.S / S_ref - 1.0,
        g_Lam=result.Lam / Lam_ref - 1.0,
    )
    for w in weights:
        F_ref = min(weighted_value(s, l, w, norm) for s, l in front)
        out[_wkey(w)] = weighted_value(result.S, result.Lam, w, norm) / F_ref - 1.0
    out.update(
        eps=eps_indicator(result.S, result.Lam, front),
        idle=result.idle, rounds=result.rounds, messages=result.messages, wall=result.wall,
        raw_S=result.raw_S, raw_Lam=result.raw_Lam,
    )
    return out


def reference_front(front, results):
    """Эталонный фронт: точный или лучший известный, дополненный решениями методов
    (статья, раздел 2: метод может найти точку лучше эвристического эталона).
    Для точного фронта ничего не меняется, кроме погрешности округления."""
    pts = list(front) + [(r.S, r.Lam) for r in results]
    return nondominated(pts)


def score_all(results, front, weights=WEIGHTS):
    """Показатели всех методов относительно общего эталона."""
    ref = reference_front(front, results)
    S_ref = min(s for s, _ in ref)
    L_ref = min(l for _, l in ref)
    return {r.name: gaps(r, S_ref, L_ref, ref, weights) for r in results}, ref
