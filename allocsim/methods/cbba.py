"""CBBA — consensus-based bundle algorithm (Choi, Brunet, How, IEEE T-RO, 2009, 25(4), 912–926).

Синхронный вариант при полносвязной сети (как в постановке статьи и у GACA).

Фаза 1 (пакет). Каждый аппарат по очереди добавляет в свой пакет задачу с наибольшей оценкой,
которую он может перебить. Оценка задачи — постоянная награда минус прирост длины пути при
лучшей вставке в текущий путь аппарата (в режиме «база» — с возвратом). Задачи чужого типа
оценки не получают. Прирост длины от вставки не обладает свойством убывающей отдачи, поэтому
ставка ограничивается сверху предыдущей ставкой в пакете («bid warping», Johnson, Choi, Ponda,
How, 2012) — это сохраняет гарантию сходимости CBBA.

Фаза 2 (согласование). Все видят все ставки; задача достаётся наибольшей ставке (при равенстве —
меньшему номеру аппарата). Аппарат, проигравший задачу из пакета, снимает её и все задачи,
добавленные после неё; ставки на снятые после неё задачи обнуляются.

Фазы повторяются, пока назначение не перестанет меняться. Раунд = фаза 1 + фаза 2;
сообщений за раунд — M·(M − 1) (каждый передаёт свои ставки каждому).

Вариант score='tdr_marginal' — исходная постановка CBBA (Choi, Brunet, How, 2009) с наградой,
убывающей со временем: ценность пути Σ_j λ^{t_j}, ставка — прирост ценности пути при лучшей вставке
(новая задача добавляет λ^{t_j}, задачи после неё теряют из-за задержки). Так был настроен CBBA у
авторов GACA (уточнено у авторов 12.10.2026): t — путь от старта до задачи, делённый на 10,
λ = 0,9, длина пакета не ограничена (lam_abs = ln(1/0,9)/10 на единицу расстояния).

Вариант score='tdr' — оценка, как в распространённой открытой реализации CBBA-Python
(github.com/zehuilu/CBBA-Python, lib/CBBA.py): ставка — ценность самой задачи, убывающая со
временем прибытия, exp(−λ·t_j), при лучшей позиции вставки; прирост пути для других задач не
учитывается и ставка не ограничивается предыдущей. λ задаётся долей стороны поля:
λ = discount / side (по умолчанию discount = 1). Нужен для сверки с GACA в условиях авторов: проверить, не объясняется ли
разрыв GACA и CBBA в статье GACA тем, какой оценкой пользовался CBBA.
"""
import numpy as np

from .base import Timer, finalize


def _insertion(inst, k, path, j):
    """Наименьший прирост длины пути аппарата k при вставке задачи j и позиция вставки."""
    P = inst.points
    nodes = [inst.starts[k]] + [P[t] for t in path]
    end = inst.starts[k] if inst.returns else None
    best, best_pos = np.inf, 0
    pj = P[j]
    for pos in range(len(path) + 1):
        a = nodes[pos]
        if pos < len(path):
            b = P[path[pos]]
            inc = np.linalg.norm(a - pj) + np.linalg.norm(pj - b) - np.linalg.norm(a - b)
        elif end is not None:
            inc = np.linalg.norm(a - pj) + np.linalg.norm(pj - end) - np.linalg.norm(a - end)
        else:
            inc = np.linalg.norm(a - pj)
        if inc < best - 1e-12:
            best, best_pos = inc, pos
    return best, best_pos


def _tdr(inst, k, path, j, lam):
    """Наибольшая ценность exp(−λ·t_j) по позициям вставки задачи j в путь аппарата k."""
    P = inst.points
    nodes = [inst.starts[k]] + [P[t] for t in path]
    best, best_pos, t_prev = -np.inf, 0, 0.0
    for pos in range(len(path) + 1):
        if pos > 0:
            t_prev += float(np.linalg.norm(nodes[pos] - nodes[pos - 1]))
        val = float(np.exp(-lam * (t_prev + np.linalg.norm(nodes[pos] - P[j]))))
        if val > best + 1e-15:
            best, best_pos = val, pos
    return best, best_pos


LAM_AUTHOR = float(-np.log(0.9) / 10.0)   # λ = 0,9 за 10 единиц пути (настройка авторов GACA)


def _all_bids(DT, ds_k, path, returns, score, lam):
    """Для всех задач сразу: наилучшая позиция вставки в путь и её оценка (то же, что _insertion
    и _tdr по одной задаче, но матрицами). Возвращает (прирост длины или время прибытия, позиция)."""
    if score == "tdr_marginal":
        # прирост ценности пути Σ exp(−λ t) при вставке; возвращается со знаком минус (меньше — лучше)
        if not path:
            gain = np.exp(-lam * ds_k)[None, :]
        else:
            p = np.asarray(path)
            frm = np.vstack([ds_k[None, :], DT[p, :]])                         # узел позиции → задача
            seg = np.r_[ds_k[p[0]], DT[p[:-1], p[1:]]] if len(p) > 1 else np.r_[ds_k[p[0]]]
            t_node = np.r_[0.0, np.cumsum(seg)]                                # время в узле позиции
            w_task = np.exp(-lam * t_node[1:])                                 # ценность задач пути
            tail = np.r_[np.cumsum(w_task[::-1])[::-1], 0.0]                   # сумма ценности после позиции
            to = np.vstack([DT[p, :], np.zeros_like(ds_k)[None, :]])           # задача → следующий узел
            delay = frm + to - np.r_[seg, 0.0][:, None]                        # задержка задач после вставки
            delay[-1, :] = 0.0
            gain = np.exp(-lam * (t_node[:, None] + frm)) + (np.exp(-lam * delay) - 1.0) * tail[:, None]
        pos = np.argmax(gain, axis=0)
        return -gain[pos, np.arange(gain.shape[1])], pos
    if path:
        p = np.asarray(path)
        frm = np.vstack([ds_k[None, :], DT[p, :]])                 # от узла позиции до задачи
        if score == "tdr":
            seg = np.r_[ds_k[p[0]], DT[p[:-1], p[1:]]] if len(p) > 1 else np.r_[ds_k[p[0]]]
            t_node = np.r_[0.0, np.cumsum(seg)]                     # время в узле позиции
            val = t_node[:, None] + frm
        else:
            to = np.vstack([DT[p, :], (ds_k if returns else np.zeros_like(ds_k))[None, :]])
            base = np.r_[ds_k[p[0]], DT[p[:-1], p[1:]] if len(p) > 1 else np.empty(0),
                         ds_k[p[-1]] if returns else 0.0]
            val = frm + to - base[:, None]
    else:
        val = (ds_k + (ds_k if returns and score != "tdr" else 0.0))[None, :]
    pos = np.argmin(val, axis=0)
    return val[pos, np.arange(val.shape[1])], pos


def cbba(inst, max_bundle=None, max_rounds=None, score="marginal", discount=1.0, lam_abs=None):
    n, M = inst.n_tasks, inst.n_agents
    can = inst.can_do()
    L_t = n if max_bundle is None else max_bundle
    max_rounds = 20 * n if max_rounds is None else max_rounds
    diam = float(np.linalg.norm(np.ptp(np.vstack([inst.points, inst.starts]), axis=0)))
    reward = 4.0 * diam   # больше любого прироста длины при вставке одной задачи
    lam = discount / float(np.max(np.ptp(np.vstack([inst.points, inst.starts]), axis=0)))
    if lam_abs is not None:
        lam = float(lam_abs)
    with Timer() as t:
        P = inst.points
        DT = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        DS = np.linalg.norm(inst.starts[:, None, :] - P[None, :, :], axis=2)
        y = np.zeros(n)                      # наибольшая ставка по задаче
        z = np.full(n, -1)                   # победитель
        bundles = [[] for _ in range(M)]     # (задача, ставка) в порядке добавления
        paths = [[] for _ in range(M)]
        rounds = 0
        prev = None
        seen, cycled = set(), False
        while rounds < max_rounds:
            rounds += 1
            # фаза 1: все строят пакеты по сведениям с конца прошлого раунда (синхронно)
            y0, z0 = y.copy(), z.copy()
            for k in range(M):
                in_b = np.zeros(n, bool)
                in_b[[j for j, _ in bundles[k]]] = True
                while len(bundles[k]) < L_t:
                    last_bid = bundles[k][-1][1] if bundles[k] else np.inf
                    val, pos = _all_bids(DT, DS[k], paths[k], inst.returns, score, lam)
                    if score == "tdr":
                        bid = np.exp(-lam * val)
                    elif score == "tdr_marginal":
                        bid = -val
                    else:
                        bid = np.minimum(reward - val, last_bid)
                    ok = can[k] & ~in_b & ((bid > y0 + 1e-9) |
                                           ((np.abs(bid - y0) <= 1e-9) & ((z0 < 0) | (z0 > k))))
                    if not ok.any():
                        break
                    cand = np.flatnonzero(ok)
                    j = int(cand[np.argmax(bid[cand])])   # при равенстве — меньший номер
                    bundles[k].append((j, float(bid[j])))
                    paths[k].insert(int(pos[j]), j)
                    in_b[j] = True
            # фаза 2: по каждой задаче — наибольшая ставка среди пакетов
            claims = {}
            for k in range(M):
                for j, bid in bundles[k]:
                    cur = claims.get(j)
                    if cur is None or bid > cur[0] + 1e-9 or (abs(bid - cur[0]) <= 1e-9 and k < cur[1]):
                        claims[j] = (bid, k)
            y[:] = 0.0
            z[:] = -1
            for j, (bid, k) in claims.items():
                y[j], z[j] = bid, k
            for k in range(M):
                lost = next((i for i, (j, _) in enumerate(bundles[k]) if z[j] != k), None)
                if lost is None:
                    continue
                for j, _ in bundles[k][lost + 1:]:
                    if z[j] == k:
                        y[j], z[j] = 0.0, -1
                removed = {j for j, _ in bundles[k][lost:]}
                bundles[k] = bundles[k][:lost]
                paths[k] = [j for j in paths[k] if j not in removed]
            state = tuple(z)
            if state == prev and (z >= 0).all():
                break
            full = (state, tuple(np.round(y, 9)), tuple(tuple(j for j, _ in b) for b in bundles))
            if full in seen:          # состояние повторилось — торг зациклился (без убывающей отдачи)
                cycled = True
                break
            seen.add(full)
            prev = state
    converged = (z >= 0).all() and state == prev and not cycled
    if not (z >= 0).all():   # не сошлось — остаток ближайшему умеющему (фиксируется в info)
        for j in np.flatnonzero(z < 0):
            ks = np.flatnonzero(can[:, j])
            z[j] = int(ks[np.argmin(np.linalg.norm(inst.starts[ks] - inst.points[j], axis=1))])
            paths[z[j]].append(int(j))
    name = "cbba" if score == "marginal" else f"cbba_{score}"
    return finalize(inst, name, z, raw_routes=paths, rounds=rounds,
                    messages=rounds * M * (M - 1), wall=t.wall,
                    info=dict(converged=bool(converged), cycled=cycled, max_bundle=L_t, score=score))
