"""GACA — Grouping Auction-Consensus Algorithm (Rodriguez, Koenig, Dong, Lu; arXiv 2608.15884,
препринт, v1 от 16.08.2026). Реализация по тексту статьи; кода авторы не публиковали.

Идея: аппараты торгуются не за отдельные задачи, как в CBBA, а за группы близких задач. Группы —
цепочки из графа ближайших соседей; маршрут аппарата — цепочки, пристыкованные одна за другой.

1. Группы (одинаково у всех аппаратов, без обмена сообщениями).
   - Каждая задача соединяется с ближайшей (повторы рёбер склеиваются).
   - Пока есть задача с тремя и более рёбрами, удаляется самое длинное ребро при таких задачах;
     если задача осталась без рёбер, она пристыковывается к ближайшему концу цепочки (задаче
     с одним ребром).
   - В каждом цикле удаляется самое длинное ребро.
   - Связные части — группы-цепочки.

2. Фаза 1 (каждый аппарат предлагает не больше одного действия). Хвост τ_i — последняя задача
   маршрута аппарата i (или старт). Кандидаты:
   - взять свободную группу целиком: вход с ближнего к τ_i конца r, ставка b = d(τ_i, r);
   - взять часть свободной группы от внутренней задачи r до одного из концов: b = d(τ_i, r) + c_split,
     остаток пристыковывается к ближайшему концу другой свободной группы;
   - забрать чужую группу целиком: b = d(τ_i, r), если b меньше ставки владельца на эту группу;
   - забрать хвост чужой группы от внутренней задачи r до её конца (по ходу владельца):
     b = d(τ_i, r), если b меньше ребра, которым владелец входит в r.
   Выбирается действие с наименьшей ставкой.

3. Фаза 2 (согласование; связь полносвязная, раунды синхронные). Предложения, затрагивающие одну
   группу или маршрут другого предлагающего, конфликтуют; в каждой связной части конфликтов
   побеждает наименьшее Φ_k (формула 8 статьи): сумма по спорным группам «длина цепочки + ставка
   на вход». Аппарат, потерявший группу или её хвост, освобождает группы, пристыкованные после
   неё: их ставки считались от прежнего хвоста.

Раунды повторяются, пока никто ничего не предлагает. Раунд = фаза 1 + фаза 2; сообщений за раунд
M·(M − 1), как у CBBA.

Уточнено у авторов (12.10.2026):
- равенства при выборе ближайшего соседа — в пользу меньшего номера; если две задачи ближайшие к
  одной и той же, обе соединяются с ней; размер группы не ограничен;
- «ближайшая задача степени один» — ближайшая к оставшейся без рёбер задаче вершина с одним ребром;
- ставка «забрать чужую группу целиком» — как «взять свободную группу целиком»: расстояние от хвоста
  до ближнего конца группы;
- разрезать свободную группу: у задачи входа r два ребра — разрывается одно из них; ненужная часть
  пристыковывается одним из своих двух концов к ближайшей задаче степени один (концу цепочки из
  двух и более задач) другой свободной группы. Из 4 вариантов (какое ребро разорвать × каким концом
  пристыковать) выбирается вариант с наименьшим c_split = (длина нового ребра) − (длина разорванного
  ребра); ставка b = d(τ_i, r) + c_split. Если пристыковать не к чему, разрезать нельзя.
  (split_rule='author'; прежнее наше прочтение — пристыковка только концом у разреза, к любому концу
  свободной группы, включая одиночные, — split_rule='cut_end');
- забрать хвост чужой группы: ставка — расстояние от хвоста до задачи r, допустима, только если
  меньше длины разрываемого ребра; c_split здесь нет — как у нас;
- устаревшие сведения (метки времени) при полносвязной сети не нужны — авторы их не проверяли.
Осталось выбранным нами:
- по каким группам суммируется Φ_k: по спорным (phi='groups', по умолчанию) или по всему маршруту
  после действия (phi='route'). Во втором прочтении владелец, который сам ничего не предлагает,
  тоже участвует в споре со своим текущим маршрутом, и забрать у него можно, только если маршрут
  после этого дешевле его маршрута. Сверка в условиях авторов (5–20 аппаратов, 2–4 задачи на
  аппарат, 400 полей; статья, табл. 5): 'groups' — медиана 96,5–97,3 % от лучшего известного
  суммарного пути, как у авторов (≈ 97 %); 'route' — 91,9–93,5 %, перекос к выравниванию нагрузки;
- освобождённые группы остаются свободными цепочками в прежнем порядке;
- если состояние повторилось (обмен группами зациклился), дальше разрешены только свободные
  группы; номер раунда пишется в info['cycle_at'];
- равенства — в пользу меньшего номера аппарата, задачи, группы.
Варианты для проверки, что даёт выигрыш: phi='route', split=False, steal=False.

Наше приспособление к разнотипной группе (в статье агенты однородны): группы строятся внутри
каждого типа отдельно, аппарат предлагает действия только с группами своих типов.

Режим «база» с возвратом: ставки — как в статье (без учёта обратного пути); единый решатель
маршрутов после метода считает путь с возвратом, как для всех методов.
"""
import numpy as np

from .base import Timer, finalize

EPS = 1e-9


def nn_chains(D):
    """Группы-цепочки по матрице расстояний D (n×n) между задачами одного типа.
    Возвращает список цепочек (списков локальных номеров) в порядке обхода."""
    n = len(D)
    if n == 1:
        return [[0]]
    W = D.astype(float).copy()
    np.fill_diagonal(W, np.inf)
    adj = [set() for _ in range(n)]
    for u in range(n):
        v = int(np.argmin(W[u]))           # при равенстве — меньший номер
        adj[u].add(v)
        adj[v].add(u)

    def edge_key(u, v):                    # самое длинное, при равенстве — с меньшими номерами
        return (W[u, v], -min(u, v), -max(u, v))

    while True:
        conflict = [u for u in range(n) if len(adj[u]) > 2]
        if not conflict:
            break
        _, u, v = max((edge_key(u, v), u, v) for u in conflict for v in adj[u])
        adj[u].discard(v)
        adj[v].discard(u)
        for x in (u, v):
            if not adj[x]:
                ends = [w for w in range(n) if w != x and len(adj[w]) == 1]
                if ends:
                    w = min(ends, key=lambda w: (W[x, w], w))
                    adj[x].add(w)
                    adj[w].add(x)
    # связные части; цикл (рёбер столько же, сколько вершин) размыкается по самому длинному ребру
    seen, chains = set(), []
    for s in range(n):
        if s in seen:
            continue
        comp, stack = [], [s]
        seen.add(s)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        edges = {(min(u, v), max(u, v)) for u in comp for v in adj[u]}
        if len(comp) > 1 and len(edges) == len(comp):
            _, u, v = max((edge_key(u, v), u, v) for u, v in edges)
            adj[u].discard(v)
            adj[v].discard(u)
        ends = sorted(u for u in comp if len(adj[u]) <= 1)
        start = ends[0]
        chain, prev, cur = [start], None, start
        while True:
            nxt = [v for v in adj[cur] if v != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            chain.append(cur)
        chains.append(chain)
    return chains


class _Seg:
    __slots__ = ("tasks", "owner", "bid", "typ")

    def __init__(self, tasks, typ, owner=-1, bid=np.inf):
        self.tasks, self.typ, self.owner, self.bid = list(tasks), typ, owner, bid


def gaca(inst, phi="groups", split=True, steal=True, max_rounds=None, split_rule="author"):
    n, M = inst.n_tasks, inst.n_agents
    can = inst.can_do()
    P, S0 = inst.points, inst.starts
    DT = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    DS = np.linalg.norm(S0[:, None, :] - P[None, :, :], axis=2)
    max_rounds = 20 * n + 20 if max_rounds is None else max_rounds

    with Timer() as t:
        segs, nid = {}, 0
        for typ in np.unique(inst.task_type):
            idx = np.flatnonzero(inst.task_type == typ)
            for ch in nn_chains(DT[np.ix_(idx, idx)]):
                segs[nid] = _Seg(idx[ch], int(typ))
                nid += 1
        n_groups = len(segs)
        type_ok = {typ: can[:, np.flatnonzero(inst.task_type == typ)[0]]
                   for typ in np.unique(inst.task_type)}
        routes = [[] for _ in range(M)]

        def chain_len(tasks):
            return float(sum(DT[a, b] for a, b in zip(tasks, tasks[1:])))

        def d_from(i, j):
            """Расстояние от хвоста аппарата i до задачи j."""
            if routes[i]:
                return float(DT[segs[routes[i][-1]].tasks[-1], j])
            return float(DS[i, j])

        def total(i):
            return sum(chain_len(segs[s].tasks) + segs[s].bid for s in routes[i])

        def free_ends(typ, exclude):
            """Концы других свободных групп того же типа. По ответу автора — только «задачи степени
            один», то есть концы цепочек из двух и более задач; в прочтении 'cut_end' — любые."""
            min_len = 2 if split_rule == "author" else 1
            return [(e, s) for s, sg in segs.items() if sg.owner < 0 and sg.typ == typ
                    and s != exclude and len(sg.tasks) >= min_len
                    for e in {sg.tasks[0], sg.tasks[-1]}]

        def propose(i):
            best = None

            def offer(b, act):
                nonlocal best
                key = (b, act["sid"], act["kind"], tuple(act["take"]))
                if best is None or key < best[0]:
                    best = (key, act)

            for sid, sg in segs.items():
                if not type_ok[sg.typ][i] or sg.owner == i:
                    continue
                T, L = sg.tasks, len(sg.tasks)
                if sg.owner < 0:
                    for take in (T, T[::-1]):
                        offer(d_from(i, take[0]), dict(kind="claim", sid=sid, take=take, rest=None,
                                                       bid=d_from(i, take[0])))
                    if split and L >= 3:
                        ends = free_ends(sg.typ, sid)
                        if ends:
                            for q in range(1, L - 1):
                                for take, rest in ((T[q:], T[:q]), (T[:q + 1][::-1], T[q + 1:][::-1])):
                                    r, r1 = take[0], rest[-1]       # r1 — сосед r в остатке
                                    tips = (r1, rest[0]) if split_rule == "author" else (r1,)
                                    c, e, es, tip = min((DT[t, x[0]] - DT[r1, r], x[0], x[1], t)
                                                        for t in tips for x in ends)
                                    b = d_from(i, r) + c
                                    offer(b, dict(kind="split", sid=sid, take=list(take),
                                                  rest=list(rest), join=(es, e), tip=tip, bid=b))
                elif steal and steals_on:
                    # забрать можно, если ставка ниже ставки владельца; при phi='route' ещё и
                    # маршрут после этого дешевле текущего маршрута владельца
                    tot_i, tot_k = totals[i], totals[sg.owner]
                    for take in (T, T[::-1]):
                        b = d_from(i, take[0])
                        if b < sg.bid - EPS and (phi != "route" or tot_i + b + chain_len(take) < tot_k - EPS):
                            offer(b, dict(kind="steal", sid=sid, take=list(take), rest=None, bid=b))
                    for q in range(1, L):
                        b = d_from(i, T[q])
                        if b < DT[T[q - 1], T[q]] - EPS and (phi != "route" or tot_i + b + chain_len(T[q:]) < tot_k - EPS):
                            offer(b, dict(kind="steal_tail", sid=sid, take=T[q:], rest=T[:q], bid=b))
            if best is None:
                return None
            act = best[1]
            act["agent"] = i
            act["phi"] = act["bid"] + chain_len(act["take"]) + (totals[i] if phi == "route" else 0.0)
            touched = {act["sid"]}
            if act["kind"] == "split":
                touched.add(act["join"][0])
            act["touched"] = touched
            act["victim"] = segs[act["sid"]].owner if act["kind"] in ("steal", "steal_tail") else -1
            return act

        def release_after(k, sid):
            """Освободить группы аппарата k после группы sid (их ставки считались от прежнего хвоста)."""
            pos = routes[k].index(sid)
            for s in routes[k][pos + 1:]:
                segs[s].owner, segs[s].bid = -1, np.inf
            routes[k] = routes[k][:pos + 1]

        def apply(act):
            nonlocal nid
            i, sid, sg = act["agent"], act["sid"], segs[act["sid"]]
            kind = act["kind"]
            if kind == "claim":
                sg.tasks, sg.owner, sg.bid = list(act["take"]), i, act["bid"]
                routes[i].append(sid)
            elif kind == "split":
                es, e = act["join"]
                other = segs[es]
                rest = act["rest"]
                ends_at_tip = rest if rest[-1] == act["tip"] else rest[::-1]   # остаток кончается стыкуемым концом
                other.tasks = (ends_at_tip + other.tasks) if other.tasks[0] == e \
                    else (other.tasks + ends_at_tip[::-1])
                sg.tasks, sg.owner, sg.bid = list(act["take"]), i, act["bid"]
                routes[i].append(sid)
            elif kind == "steal":
                k = sg.owner
                release_after(k, sid)
                routes[k].remove(sid)
                sg.tasks, sg.owner, sg.bid = list(act["take"]), i, act["bid"]
                routes[i].append(sid)
            elif kind == "steal_tail":
                k = sg.owner
                release_after(k, sid)
                sg.tasks = list(act["rest"])                      # голова остаётся у владельца
                segs[nid] = _Seg(act["take"], sg.typ, owner=i, bid=act["bid"])
                routes[i].append(nid)
                nid += 1

        rounds, converged, steals_on, seen, cycle_at = 0, False, True, set(), None
        while rounds < max_rounds:
            rounds += 1
            sig = tuple((s, sg.owner, tuple(sg.tasks)) for s, sg in sorted(segs.items())) + \
                tuple(tuple(r) for r in routes)
            if sig in seen and steals_on:
                # состояние повторилось — обмен группами зациклился; дальше только свободные группы
                steals_on, cycle_at = False, rounds
            seen.add(sig)
            totals = [total(k) for k in range(M)]
            props = [p for p in (propose(i) for i in range(M)) if p is not None]
            if not props:
                converged = True
                break
            # конфликты: общая группа или действие против маршрута другого предлагающего
            parent = list(range(len(props)))

            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a

            for a in range(len(props)):
                for b in range(a + 1, len(props)):
                    pa, pb = props[a], props[b]
                    clash = bool(pa["touched"] & pb["touched"])
                    clash |= pa["victim"] == pb["agent"] or pb["victim"] == pa["agent"]
                    clash |= pa["victim"] >= 0 and pa["victim"] == pb["victim"]
                    if clash:
                        parent[find(a)] = find(b)
            groups = {}
            for a, p in enumerate(props):
                groups.setdefault(find(a), []).append(p)
            winners = [min(g, key=lambda p: (p["phi"], p["agent"])) for g in groups.values()]
            for act in sorted(winners, key=lambda p: p["agent"]):
                apply(act)

        assignment = np.full(n, -1)
        raw = []
        for k in range(M):
            r = [j for s in routes[k] for j in segs[s].tasks]
            raw.append(r)
            assignment[r] = k
    unassigned = int((assignment < 0).sum())
    if unassigned:   # не сошлось за max_rounds — остаток жадно ближайшему умеющему (фиксируется в info)
        for j in np.flatnonzero(assignment < 0):
            ks = np.flatnonzero(can[:, j])
            k = int(ks[np.argmin(DS[ks, j])])
            assignment[j] = k
            raw[k].append(int(j))
    name = "gaca" + ("" if phi == "groups" else f"_{phi}") + ("" if split else "_nosplit") + \
        ("" if split_rule == "author" else f"_{split_rule}") + \
        ("" if steal else "_nosteal")
    return finalize(inst, name,
                    assignment, raw_routes=raw, rounds=rounds, messages=rounds * M * (M - 1),
                    wall=t.wall, info=dict(converged=converged, groups=n_groups,
                                           unassigned_at_stop=unassigned, cycle_at=cycle_at))
