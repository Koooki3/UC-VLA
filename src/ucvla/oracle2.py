"""Exact-enumeration posterior oracle for SODH v2 (status: ``implemented``, symbolic).

The oracle sees only what a policy could see (observed ticks, task parameters, the clock) and
computes the exact posterior over the hidden state under the task's *declared* generative model.
Ticks that are not in the retained context are treated as unknown (same as sensor dropout), so
the oracle also measures what a *compressed* context can support (Memory Sufficiency). It shares
no code or rules with any heuristic decoder; it shares only the public environment model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

TICK_NS = 100_000_000
Front = Mapping[str, str]


def history_len(current_ns: int) -> int:
    """Number of past ticks given the current tick time (tick i sits at (i+1)*TICK_NS)."""
    return round(current_ns / TICK_NS) - 1


def tick_index(time_ns: int) -> int:
    return round(time_ns / TICK_NS) - 1


def _mix(dist: list[float], m: float) -> list[float]:
    n = len(dist)
    return [d * (1 - m) + (1 - d) * m / (n - 1) for d in dist]


def _track(params: Mapping[str, str], kept: Mapping[int, Front], n_hist: int) -> list[float]:
    n = int(params["n_slots"])
    eps = float(params["sensor_eps"])
    rate = float(params["swap_rate"])
    m = min(1.0, 2 * rate / n)  # P(ball moves) on a tick whose swap is hidden
    dist = [1.0 / n] * n
    seen = False
    for i in range(n_hist):
        tick = kept.get(i)
        if tick is None:  # unknown tick: swap-or-not unknown
            if seen:
                dist = _mix(dist, m)
            continue
        reveal = tick.get("reveal", "none")
        if reveal.startswith("slot:"):
            r = int(reveal[5:])
            dist = [(1 - eps) if s == r else eps / (n - 1) for s in range(n)]
            seen = True
        sw = tick.get("swap", "none")
        if sw == "?":
            if seen:
                dist = _mix(dist, m)
        elif sw != "none":
            a, b = (int(x) for x in sw.split("-"))
            dist[a], dist[b] = dist[b], dist[a]
    return dist


def _events(params: Mapping[str, str], kept: Mapping[int, Front], n_hist: int,
            current: Front) -> dict[int, float]:
    """Distribution over the number of accepted triggers (DP over (contact, count))."""
    jam = float(params["jam"])
    p_up, p_hold = float(params["p_attempt"]), float(params["p_hold"])
    state: dict[tuple[int, int], float] = {(0, 0): 1.0}
    ticks: list[Front | None] = [kept.get(i) for i in range(n_hist)] + [current]
    for tick in ticks:
        cs = (0, 1) if tick is None or tick["contact"] == "?" else (
            (1,) if tick["contact"] == "down" else (0,))
        ls = (0, 1) if tick is None or tick["lamp"] == "?" else (
            (1,) if tick["lamp"] == "on" else (0,))
        new: dict[tuple[int, int], float] = {}
        for (cp, cnt), p in state.items():
            for c in cs:
                pc = (p_up if c else 1 - p_up) if cp == 0 else (p_hold if c else 1 - p_hold)
                edge = cp == 0 and c == 1
                for lamp in ls:
                    pl = ((1 - jam) if lamp else jam) if edge else (0.0 if lamp else 1.0)
                    w = p * pc * pl
                    if w > 0:
                        key = (c, cnt + lamp)
                        new[key] = new.get(key, 0.0) + w
        z = sum(new.values())
        state = {k: v / z for k, v in new.items()}
    out: dict[int, float] = {}
    for (_, cnt), p in state.items():
        out[cnt] = out.get(cnt, 0.0) + p
    return out


def _resume(params: Mapping[str, str], kept: Mapping[int, Front], n_hist: int) -> dict[str, float]:
    items = params["items"].split(",")
    f, q, rem = float(params["fail"]), float(params["p_attempt"]), float(params["removal"])
    m = len(items)
    placed = dict.fromkeys(items, 0.0)
    for i in range(n_hist):
        tick = kept.get(i)
        if tick is None:  # whole tick unknown
            for it in items:
                placed[it] += (1 - placed[it]) * q * (1 - f) / m
                placed[it] *= 1 - rem / m
            continue
        a, r, rm = tick.get("attempt", "none"), tick.get("result", "none"), tick.get("removed", "none")
        if a in placed:
            if r == "ok":
                placed[a] = 1.0
            elif r == "?":
                placed[a] += (1 - placed[a]) * (1 - f)
        if rm in placed:
            placed[rm] = 0.0
    return placed


def posterior(task: str, params: Mapping[str, str], options: Sequence[str],
              kept: Mapping[int, Front], current: Front, current_ns: int) -> dict[str, float]:
    """Posterior probability that each option is correct (normalised), given the retained context."""
    n_hist = history_len(current_ns)
    if task == "track_ball":
        d = _track(params, kept, n_hist)
        p = {f"pick_slot_{k}": d[k] for k in range(len(d))}
    elif task == "button_events":
        k_target = int(params["target_count"])
        dist = _events(params, kept, n_hist, current)
        below = sum(v for c, v in dist.items() if c < k_target)
        p = {"press": below, "stop": 1.0 - below}
    elif task == "resume_items":
        placed = _resume(params, kept, n_hist)
        p = {f"place_{it}": 1.0 - v for it, v in placed.items()}
    elif task == "pick_flagged":
        p = {f"pick_{current['flag']}": 1.0}
    else:
        raise ValueError(f"no oracle for task {task!r}")
    z = sum(p.get(o, 0.0) for o in options) or 1.0
    return {o: p.get(o, 0.0) / z for o in options}


def optimal(post: Mapping[str, float], options: Sequence[str]) -> tuple[str, float]:
    """(argmax with option-order tie-break, margin = best - second)."""
    ranked = sorted(options, key=lambda o: (-post[o], options.index(o)))
    second = post[ranked[1]] if len(ranked) > 1 else 0.0
    return ranked[0], post[ranked[0]] - second
