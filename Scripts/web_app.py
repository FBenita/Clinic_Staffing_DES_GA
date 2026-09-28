"""
Walk-in Clinic Staffing — Robust Decision Support System
=========================================================

Interactive companion to the paper's robust staffing study. A five-stage walk-in clinic (reception -> 
nurse -> doctor -> pharmacy -> reception) is simulated for all 625 staffing plans in {1..5}^4 under
a nominal demand scenario (theta0) and a surge scenario (theta1).
A plan is *robustly feasible* when it clears the service-level floor tau under both scenarios;
its waiting cost and patients served are then judged at their worst case across the two.

Install and launch
------------------
    pip install streamlit plotly pandas numpy scipy
    streamlit run app.py

    Needs Streamlit 1.35+ (click-to-select on charts) and Plotly 5+; the app adapts to newer
    Streamlit releases (width="stretch" API) and to Streamlit's light and dark themes.

Optional extras:
    python app.py                  # launches Streamlit with the slate theme locked in
    python app.py --benchmark      # headless run: timings + the paper's headline numbers
    python app.py --export out/    # headless run: writes the notebook's CSV tables to out/

Faithfulness to the paper
-------------------------
The engine is the notebook's `simulate_clinic`, restructured for speed but consuming the random
number stream in exactly the same order (arrivals are pre-drawn in bulk and the generator state is
restored; reception draws use the same underlying uniform variate). With the paper's defaults
it reproduces the notebook's replication outputs, the 625-configuration tables and both Pareto
fronts bit for bit, and it runs about 2x faster per replication. The same common-random-number
seeds are used (replication i uses default_rng(i + 1)) for every plan and scenario.

Performance
-----------
* The 625 plans x 2 scenarios x R replications are spread over all CPU cores with a
  `ProcessPoolExecutor` that uses the "spawn" start method on every OS (Windows/macOS behaviour
  everywhere, and no fork-in-a-threaded-server hazards on Linux).
* Simulated scenarios are stored in `st.cache_data` (persisted to disk), keyed only by the
  inputs that change the simulation: arrival mean, service times, shift length and replications.
  Wages, the service-level floor and the waiting-cost valuation are *pricing* inputs: changing
  them re-prices cached simulations instantly instead of re-simulating. Changing only theta1
  re-simulates only the surge scenario.
* Everything the Streamlit UI needs is imported lazily, so worker processes (which re-import this
  file) start in a fraction of a second.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import heapq
import importlib.util
import inspect
import itertools
import math
import multiprocessing
import os
import runpy
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Worker processes import this file as "__mp_main__" (spawn) or through runpy (pool initializer).
# Register that module under a stable name so task functions pickled by the parent resolve here.
ENGINE_ALIAS = "_clinic_dss_engine"
try:
    APP_FILE = os.path.abspath(__file__)
except NameError:
    APP_FILE = os.path.abspath("app.py")

if __name__ not in ("__main__", ENGINE_ALIAS):
    sys.modules.setdefault(ENGINE_ALIAS, sys.modules[__name__])
# =============================================================================================
# 1. Model constants (paper defaults)
# =============================================================================================
ROLES = ("Receptionist", "Nurse", "Doctor", "Pharmacist")
ROLE_SHORT = ("Rec", "Nur", "Doc", "Pha")
STAGES = ("Check-in", "Nurse triage", "Doctor consult", "Pharmacy", "Check-out")
STAGE_ROLE = (0, 1, 2, 3, 0)          # which role serves each of the five stages
STAFF_LEVELS = (1, 2, 3, 4, 5)
CONFIGS = tuple(itertools.product(STAFF_LEVELS, repeat=4))   # the 625 plans, notebook order
CONFIG_INDEX = {c: i for i, c in enumerate(CONFIGS)}

ARRIVAL_SD = 1.0                       # inter-arrival SD (minutes), fixed in the paper
MIN_SERVICE = 1e-6                     # floor applied to every sampled duration, as in the paper

PAPER = {
    "wages": (110.0, 160.0, 250.0, 130.0),         # $/day: receptionist, nurse, doctor, pharmacist
    "theta0": 5.0,                                 # nominal mean inter-arrival time (min)
    "theta1": 4.0,                                 # surge mean inter-arrival time (min)
    # service times: reception U(min, max); nurse, doctor, pharmacy ~ Normal(mean, sd)
    "svc": (5.0, 7.0, 10.0, 2.0, 15.0, 3.0, 8.0, 1.0),
    "tau": 0.90,                                   # service-level floor
    "sim_time": 480.0,                             # shift length (min)
    "n_reps": 30,                                  # replications per scenario
    "wait_value": 0.5,                             # waiting minute valued at 0.5 x nurse $/min
}


# =============================================================================================
# 2. Discrete-event simulation engine
# =============================================================================================
def arrival_stream(seed: int, arrival_mean: float, sim_time: float):
    """Arrival times of one replication and the generator positioned right after them.

    The notebook draws inter-arrival times one by one, max(Normal(mean, 1), 1e-6), until the
    clock passes `sim_time` (that last draw is consumed too). Here they are drawn in bulk, the
    generator is rewound, and exactly n + 1 draws are replayed, so the stream that follows is
    identical to the notebook's.
    """
    rng = np.random.default_rng(seed)
    bitgen = rng.bit_generator
    start_state = bitgen.state
    k = int(sim_time / max(arrival_mean, 0.05) * 1.3) + 64
    while True:
        times = np.cumsum(np.maximum(rng.normal(arrival_mean, ARRIVAL_SD, size=k), MIN_SERVICE))
        n = int(np.searchsorted(times, sim_time, side="left"))
        bitgen.state = start_state
        if n < k:
            break
        k *= 2
    rng.normal(arrival_mean, ARRIVAL_SD, size=n + 1)
    return times[:n].tolist(), rng


def simulate_day(staffing, arrivals, rng, svc, sim_time, trace=None, grid_step=5.0):
    """Simulate one clinic day (the notebook's `simulate_clinic`, restructured for speed).

    staffing  (receptionists, nurses, doctors, pharmacists)
    arrivals  sorted arrival times from `arrival_stream`; `rng` must be the generator it returned
    svc       (rec_min, rec_max, nurse_mean, nurse_sd, doctor_mean, doctor_sd, pharm_mean, pharm_sd)

    Patients visit the five stages in order; each role's servers work first-come-first-served
    from one queue (receptionists share one queue for check-in and check-out). Events at or after
    `sim_time` are never processed: patients still in the clinic then count as unserved.

    Returns (total_wait, n_served, busy_minutes): the summed queueing time of patients who
    completed all stages (the notebook's waiting-cost base), the number of such patients, and
    per role the integral of busy servers over the shift (utilization = busy / (T x servers)).
    When `trace` is a dict it is filled with diagnostics (per-stage waits, times in system,
    queue lengths every `grid_step` minutes, queue composition at closing time).
    """
    rec_lo, rec_hi, nur_mu, nur_sd, doc_mu, doc_sd, pha_mu, pha_sd = svc
    rec_span = rec_hi - rec_lo
    unif, normal = rng.random, rng.normal
    push, pop = heapq.heappush, heapq.heappop
    inf = math.inf

    cap = tuple(staffing)
    busy = [0, 0, 0, 0]
    queues = (deque(), deque(), deque(), deque())
    n = len(arrivals)
    stage = [0] * n                  # index of the stage each patient is at
    svc_total = [0.0] * n            # accumulated service time per patient
    area = [0.0, 0.0, 0.0, 0.0]      # busy-server minutes per role
    last = [0.0, 0.0, 0.0, 0.0]      # time of the last busy-count change per role
    waits = []                       # queueing time of finished patients, in completion order
    heap = []                        # pending service completions: (time, seq, patient, role)
    seq = 0
    ai = 0
    next_arrival = arrivals[0] if n else inf

    tracing = trace is not None
    if tracing:
        ready = [0.0] * n            # when the patient joined the current stage
        stage_wait = [0.0] * 5
        stage_starts = [0] * 5
        in_system = []
        grid = []
        next_grid = 0.0

    def draw(s):
        # One service duration for stage s, drawn exactly as the notebook does.
        if s == 0 or s == 4:
            d = rec_lo + rec_span * unif()          # == rng.uniform(rec_lo, rec_hi)
        elif s == 2:
            d = normal(doc_mu, doc_sd)
        elif s == 1:
            d = normal(nur_mu, nur_sd)
        else:
            d = normal(pha_mu, pha_sd)
        return d if d > MIN_SERVICE else MIN_SERVICE

    while True:
        if heap and heap[0][0] < next_arrival:
            # ---- a service completes (arrivals win ties, as in the notebook's event order)
            t, _, p, r = pop(heap)
            if t >= sim_time:
                break
            if tracing:
                while next_grid <= t:
                    grid.append((len(queues[0]), len(queues[1]), len(queues[2]), len(queues[3])))
                    next_grid += grid_step
            b = busy[r]
            area[r] += b * (t - last[r])
            last[r] = t
            busy[r] = b - 1
            s = stage[p] + 1
            stage[p] = s
            if s == 5:
                waits.append((t - arrivals[p]) - svc_total[p])
                if tracing:
                    in_system.append(t - arrivals[p])
            else:
                if tracing:
                    ready[p] = t
                nr = STAGE_ROLE[s]
                bn = busy[nr]
                if bn < cap[nr]:
                    d = draw(s)
                    svc_total[p] += d
                    area[nr] += bn * (t - last[nr])
                    last[nr] = t
                    busy[nr] = bn + 1
                    seq += 1
                    push(heap, (t + d, seq, p, nr))
                    if tracing:
                        stage_starts[s] += 1
                else:
                    queues[nr].append(p)
            # the freed server takes the next queued patient(s)
            q = queues[r]
            while q and busy[r] < cap[r]:
                p2 = q.popleft()
                s2 = stage[p2]
                d = draw(s2)
                svc_total[p2] += d
                busy[r] += 1
                seq += 1
                push(heap, (t + d, seq, p2, r))
                if tracing:
                    stage_wait[s2] += t - ready[p2]
                    stage_starts[s2] += 1
        else:
            # ---- a patient arrives
            if ai >= n:
                break
            t = next_arrival
            p = ai
            ai += 1
            next_arrival = arrivals[ai] if ai < n else inf
            if tracing:
                while next_grid <= t:
                    grid.append((len(queues[0]), len(queues[1]), len(queues[2]), len(queues[3])))
                    next_grid += grid_step
                ready[p] = t
            b = busy[0]
            if b < cap[0]:
                d = draw(0)
                svc_total[p] += d
                area[0] += b * (t - last[0])
                last[0] = t
                busy[0] = b + 1
                seq += 1
                push(heap, (t + d, seq, p, 0))
                if tracing:
                    stage_starts[0] += 1
            else:
                queues[0].append(p)

    for r in range(4):
        area[r] += busy[r] * (sim_time - last[r])
    total_wait = np.array(waits).sum() if waits else 0.0   # same summation as the notebook

    if tracing:
        while next_grid <= sim_time + 1e-9:
            grid.append((len(queues[0]), len(queues[1]), len(queues[2]), len(queues[3])))
            next_grid += grid_step
        queued_at_close = [0] * 5
        for q in queues:
            for p in q:
                queued_at_close[stage[p]] += 1
        trace.update(stage_wait=stage_wait, stage_starts=stage_starts, in_system=in_system,
                     grid=grid, queued_at_close=queued_at_close, busy_at_close=list(busy))
    return total_wait, len(waits), area


# ---------------------------------------------------------------------------------------------
# Worker-side batch evaluation
# ---------------------------------------------------------------------------------------------
_STREAM_MEMO: dict = {}
_SCRATCH_RNG = None


def _replication_streams(arrival_mean, sim_time, n_reps):
    """Arrival times + post-arrival generator state for replications 1..n_reps (memoised).

    Arrivals are drawn before any service time, so they depend only on the seed and the arrival
    mean -- not on the staffing plan -- and are shared by all 625 plans (common random numbers).
    """
    key = (float(arrival_mean), float(sim_time), int(n_reps))
    hit = _STREAM_MEMO.get(key)
    if hit is None:
        if len(_STREAM_MEMO) >= 8:
            _STREAM_MEMO.clear()
        hit = []
        for rep in range(n_reps):
            arrivals, rng = arrival_stream(rep + 1, arrival_mean, sim_time)
            hit.append((arrivals, rng.bit_generator.state))
        _STREAM_MEMO[key] = hit
    return hit


def _simulate_chunk(scenario_idx, config_ids, arrival_mean, svc, sim_time, n_reps):
    """Evaluate a batch of staffing plans under one scenario (runs inside a worker process)."""
    global _SCRATCH_RNG
    if _SCRATCH_RNG is None:
        _SCRATCH_RNG = np.random.Generator(np.random.PCG64(0))
    rng = _SCRATCH_RNG
    streams = _replication_streams(arrival_mean, sim_time, n_reps)
    k = len(config_ids)
    wait = np.zeros((k, n_reps))
    served = np.zeros((k, n_reps), dtype=np.int32)
    busy = np.zeros((k, n_reps, 4))
    for j, cid in enumerate(config_ids):
        staffing = CONFIGS[cid]
        for r, (arrivals, state) in enumerate(streams):
            rng.bit_generator.state = state          # == default_rng(r + 1) after its arrivals
            w, s, b = simulate_day(staffing, arrivals, rng, svc, sim_time)
            wait[j, r] = w
            served[j, r] = s
            busy[j, r] = b
    return scenario_idx, config_ids, wait, served, busy


# =============================================================================================
# 3. Parallel enumeration of the 625 plans
# =============================================================================================
def available_cpus() -> int:
    """CPUs this process may use (respects container / affinity limits where the OS exposes them)."""
    if hasattr(os, "process_cpu_count"):
        n = os.process_cpu_count()
    elif hasattr(os, "sched_getaffinity"):
        n = len(os.sched_getaffinity(0))
    else:
        n = os.cpu_count()
    return max(1, n or 1)


def resolve_workers(requested=None) -> int:
    n = available_cpus() if not requested else int(requested)
    if sys.platform == "win32":
        n = min(n, 61)                     # ProcessPoolExecutor limit on Windows
    return max(1, n)


def _engine_module():
    """This file imported once as a regular module, used as the pickling anchor for worker tasks.

    Streamlit executes the script as a brand-new `__main__` module on every rerun and for every
    browser session, so functions living in `__main__` are not stable pickling targets when
    several sessions are active. Tasks are therefore submitted through this separately imported
    copy (children resolve it through the alias registered at the top of the file).
    """
    mod = sys.modules.get(ENGINE_ALIAS)
    if mod is None:
        spec = importlib.util.spec_from_file_location(ENGINE_ALIAS, APP_FILE)
        fresh = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fresh)
        mod = sys.modules.setdefault(ENGINE_ALIAS, fresh)   # first loader wins, everyone shares
    return mod


def generated_per_rep(arrival_mean, sim_time, n_reps):
    return np.array([len(arrival_stream(rep + 1, arrival_mean, sim_time)[0]) for rep in range(n_reps)],
                    dtype=np.int32)


def simulate_scenarios(specs, max_workers=None, progress=None):
    """Simulate all 625 plans for each scenario spec (arrival_mean, svc, sim_time, n_reps).

    Every chunk of every scenario goes into one process pool, so cores stay busy across the
    scenario boundary. `progress(done, total)` is called from the calling thread after each chunk.
    If `progress` raises (e.g. Streamlit interrupting the run), pending work is cancelled at once.
    Returns one result dict per spec.
    """
    specs = [(float(a), tuple(float(x) for x in svc), float(T), int(R)) for a, svc, T, R in specs]
    workers = resolve_workers(max_workers)
    n_cfg = len(CONFIGS)
    total = n_cfg * len(specs)
    chunk = max(1, min(25, math.ceil(total / (workers * 10))))

    results = []
    for arrival_mean, svc, sim_time, n_reps in specs:
        results.append({
            "arrival_mean": arrival_mean, "svc": svc, "sim_time": sim_time, "n_reps": n_reps,
            "wait_sum": np.zeros((n_cfg, n_reps)),
            "served": np.zeros((n_cfg, n_reps), dtype=np.int32),
            "busy": np.zeros((n_cfg, n_reps, 4)),
            "generated": generated_per_rep(arrival_mean, sim_time, n_reps),
        })
    tasks = [(si, tuple(range(lo, min(lo + chunk, n_cfg))), *spec)
             for si, spec in enumerate(specs) for lo in range(0, n_cfg, chunk)]

    def store(out):
        si, ids, wait, served, busy = out
        res = results[si]
        res["wait_sum"][list(ids)] = wait
        res["served"][list(ids)] = served
        res["busy"][list(ids)] = busy
        return len(ids)

    t0 = time.perf_counter()
    done = 0
    if workers == 1:
        for task in tasks:
            done += store(_simulate_chunk(*task))
            if progress:
                progress(done, total)
    else:
        engine = _engine_module()
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=runpy.run_path, initargs=(APP_FILE,))
        finished = False
        try:
            futures = [pool.submit(engine._simulate_chunk, *task) for task in tasks]
            for fut in as_completed(futures):
                done += store(fut.result())
                if progress:
                    progress(done, total)
            finished = True
        finally:
            pool.shutdown(wait=finished, cancel_futures=not finished)
    elapsed = time.perf_counter() - t0
    for res in results:
        res["elapsed"] = elapsed
        res["workers"] = workers
    return results


def engine_fingerprint() -> str:
    """Hash of the engine source: part of every cache key, so editing the model invalidates caches."""
    src = "".join(inspect.getsource(f) for f in (arrival_stream, simulate_day, _simulate_chunk))
    return hashlib.sha1(src.encode()).hexdigest()[:12]


# =============================================================================================
# 4. Analytics: pricing, robust feasibility, Pareto fronts, diagnostics
# =============================================================================================
def pareto_mask(wait, cost, served, eligible):
    """Non-dominated rows among `eligible` (minimise wait and cost, maximise served).

    Same dominance rule as the notebook's `pareto_dominates` (applied to the rounded values).
    """
    idx = np.flatnonzero(eligible)
    mask = np.zeros(len(wait), dtype=bool)
    if idx.size == 0:
        return mask
    W, M, P = wait[idx], cost[idx], served[idx]
    no_worse = (W[:, None] <= W[None, :]) & (M[:, None] <= M[None, :]) & (P[:, None] >= P[None, :])
    better = (W[:, None] < W[None, :]) | (M[:, None] < M[None, :]) | (P[:, None] > P[None, :])
    dominated = (no_worse & better).any(axis=0)
    mask[idx[~dominated]] = True
    return mask


def efficient_2d(cost, wait):
    """Rows not dominated in (cost, wait) alone -- the visible frontier line of a 2-D plot."""
    n = len(cost)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        dom = (cost <= cost[i]) & (wait <= wait[i]) & ((cost < cost[i]) | (wait < wait[i]))
        keep[i] = not dom.any()
    return keep


def front_labels(n):
    """A, B, C, ... exactly as the notebook's LaTeX table labels its rows."""
    return [chr(ord("A") + i) if i < 26 else f"A{i - 25}" for i in range(n)]


def scenario_metrics(sim, wages, wait_value):
    """Daily metrics of all 625 plans under one simulated scenario, priced with `wages`."""
    rate = (wages[1] / sim["sim_time"]) * wait_value       # notebook: (nurse / T) * 0.5
    caps = np.array(CONFIGS, dtype=float)
    return {
        "wait_cost": (sim["wait_sum"] * rate).mean(axis=1),
        "wait_min": sim["wait_sum"].mean(axis=1),
        "served": sim["served"].mean(axis=1),
        "sl": sim["served"].mean(axis=1) / sim["generated"].mean(),
        "util": sim["busy"].mean(axis=1) / (sim["sim_time"] * caps),
        "generated": float(sim["generated"].mean()),
    }


def build_design_table(sim0, sim1, wages, tau, wait_value):
    """One row per staffing plan with nominal, surge and worst-case metrics plus both fronts."""
    import pandas as pd

    m0 = scenario_metrics(sim0, wages, wait_value)
    m1 = scenario_metrics(sim1, wages, wait_value)
    staff = np.array(CONFIGS)
    df = pd.DataFrame(staff, columns=list(ROLE_SHORT))
    df.insert(0, "id", np.arange(len(CONFIGS)))
    df["plan"] = [f"({a}, {b}, {c}, {d})" for a, b, c, d in CONFIGS]
    df["manpower_cost"] = (staff @ np.asarray(wages, dtype=float)).astype(float)
    df["wait_cost0"], df["wait_cost1"] = m0["wait_cost"], m1["wait_cost"]
    df["served0"], df["served1"] = m0["served"], m1["served"]
    df["sl0"], df["sl1"] = m0["sl"], m1["sl"]
    df["waiting_cost_R"] = np.maximum(m0["wait_cost"], m1["wait_cost"])
    df["served_R"] = np.minimum(m0["served"], m1["served"])
    df["sl_min"] = np.minimum(m0["sl"], m1["sl"])
    df["total_cost_R"] = df["manpower_cost"] + df["waiting_cost_R"]
    df["feasible0"] = m0["sl"] >= tau
    df["feasible1"] = m1["sl"] >= tau
    df["feasible_R"] = df["feasible0"] & df["feasible1"]
    for k, role in enumerate(ROLE_SHORT):
        df[f"util0_{role}"] = m0["util"][:, k]
        df[f"util1_{role}"] = m1["util"][:, k]
    # binding scenario = the one with the lower service level; bottleneck = its busiest role
    worse1 = m1["sl"] <= m0["sl"]
    util_bind = np.where(worse1[:, None], m1["util"], m0["util"])
    df["binding"] = np.where(worse1, "theta1", "theta0")
    df["bottleneck"] = [ROLES[k] for k in util_bind.argmax(axis=1)]
    df["bottleneck_util"] = util_bind.max(axis=1)

    # Pareto fronts on values rounded to cents / hundredths, exactly like the notebook tables
    wR, sR = np.round(df["waiting_cost_R"].to_numpy(), 2), np.round(df["served_R"].to_numpy(), 2)
    w0, s0 = np.round(m0["wait_cost"], 2), np.round(m0["served"], 2)
    cost = df["manpower_cost"].to_numpy()
    df["robust_front"] = pareto_mask(wR, cost, sR, df["feasible_R"].to_numpy())
    df["nominal_front"] = pareto_mask(w0, cost, s0, df["feasible0"].to_numpy())

    df["robust_label"] = ""
    rf = df[df.robust_front].sort_values(["manpower_cost", "waiting_cost_R"], kind="mergesort")
    df.loc[rf.index, "robust_label"] = front_labels(len(rf))
    # members of each front that are also efficient in (cost, wait) alone: the line drawn in 2-D plots
    df["robust_2d"] = False
    if len(rf):
        df.loc[rf.index, "robust_2d"] = efficient_2d(rf["manpower_cost"].to_numpy(),
                                                     np.round(rf["waiting_cost_R"].to_numpy(), 2))
    df["nominal_2d"] = False
    nf = df[df.nominal_front].sort_values(["manpower_cost", "wait_cost0"], kind="mergesort")
    if len(nf):
        df.loc[nf.index, "nominal_2d"] = efficient_2d(nf["manpower_cost"].to_numpy(),
                                                      np.round(nf["wait_cost0"].to_numpy(), 2))
    return df


def robustness_summary(df):
    """Cheapest nominal vs cheapest robust plan and the cost of robustness (notebook, Section 10)."""
    nom = df[df.nominal_front].sort_values(["manpower_cost", "wait_cost0"], kind="mergesort")
    rob = df[df.robust_front].sort_values(["manpower_cost", "waiting_cost_R"], kind="mergesort")
    out = {"n_nominal_front": len(nom), "n_robust_front": len(rob),
           "n_nominal_feasible": int(df.feasible0.sum()), "n_robust_feasible": int(df.feasible_R.sum()),
           "nominal_fail_surge": int((nom.feasible1 == False).sum()),   # noqa: E712
           "cheapest_nominal": nom.iloc[0] if len(nom) else None,
           "cheapest_robust": rob.iloc[0] if len(rob) else None}
    if len(nom) and len(rob):
        premium = rob.iloc[0].manpower_cost - nom.iloc[0].manpower_cost
        out["premium"] = premium
        out["premium_pct"] = 100.0 * premium / nom.iloc[0].manpower_cost
    return out


def strategy_presets(df):
    """Named picks on the robust front: cheapest, knee ("balanced") and near-best service."""
    rf = df[df.robust_front].sort_values(["manpower_cost", "waiting_cost_R"], kind="mergesort")
    if rf.empty:
        return {}
    cost, wait = rf["manpower_cost"].to_numpy(), rf["waiting_cost_R"].to_numpy()
    span_c, span_w = np.ptp(cost), np.ptp(wait)
    nc = (cost - cost.min()) / span_c if span_c > 0 else np.zeros_like(cost)
    nw = (wait - wait.min()) / span_w if span_w > 0 else np.zeros_like(wait)
    knee = int(np.argmin(np.hypot(nc, nw)))
    best_wait = wait.min()
    near_best = np.flatnonzero(wait <= best_wait * 1.10 + 0.01)
    high = int(near_best[np.argmin(cost[near_best])])
    return {
        "Cheapest feasible": int(rf.iloc[0].id),
        "Balanced (knee)": int(rf.iloc[knee].id),
        "High service": int(rf.iloc[high].id),
    }


def robust_front_latex(df, n_reps):
    """The robust front as the paper's LaTeX table (same layout as the notebook's Section 8)."""
    rf = df[df.robust_front].sort_values(["manpower_cost", "waiting_cost_R"], kind="mergesort")
    lines = [r"\begin{table}[htbp]", r"\centering", r"\begin{adjustbox}{max width=\textwidth}",
             r"\begin{tabular}{|l|c|c|c|c|c|c|c|c|c|}", r"\hline",
             r"\textbf{Sol.} & \textbf{Rec} & \textbf{Nur} & \textbf{Doc} & \textbf{Pha} & "
             r"\textbf{Manpower (\$)} & \textbf{Wait$^R$ (\$)} & \textbf{Served$^R$} & "
             r"\textbf{SL($\theta_0$)} & \textbf{SL($\theta_1$)} \\", r"\hline"]
    for _, row in rf.iterrows():
        lines.append(f"{row.robust_label} & {row.Rec} & {row.Nur} & {row.Doc} & {row.Pha} & "
                     f"{row.manpower_cost:.0f} & {round(row.waiting_cost_R, 2):.2f} & "
                     f"{round(row.served_R, 2):.2f} & {round(100 * row.sl0, 1):.1f}\\% & "
                     f"{round(100 * row.sl1, 1):.1f}\\% \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{adjustbox}",
              r"\caption{Complete robust Pareto front from exhaustive enumeration "
              r"($\Theta=\{\theta_0,\theta_1\}$, base and high-demand scenarios; worst-case objectives; "
              f"averaged over {n_reps}" r" replications per scenario).}",
              r"\label{tab:robust_pareto_front}", r"\end{table}"]
    return "\n".join(lines)


def offered_load(staffing, arrival_mean, svc):
    """Analytic load per server, rho = lambda x (visits x mean service) / servers."""
    rec_lo, rec_hi, nur_mu, _, doc_mu, _, pha_mu, _ = svc
    work = (2 * (rec_lo + rec_hi) / 2, nur_mu, doc_mu, pha_mu)
    return [work[k] / arrival_mean / staffing[k] for k in range(4)]


def diagnose_plan(staffing, arrival_mean, svc, sim_time, n_reps, grid_step=5.0):
    """Instrumented replications of one plan under one scenario, on the same random numbers as
    the enumeration (so its service level and waiting cost match the tables exactly)."""
    streams = _replication_streams(arrival_mean, sim_time, n_reps)
    rng = np.random.Generator(np.random.PCG64(0))
    stage_wait, stage_starts = np.zeros(5), np.zeros(5)
    queued_close = np.zeros(5)
    busy_min = np.zeros(4)
    in_system, grids, served, total_wait = [], [], 0, 0.0
    generated = 0
    for arrivals, state in streams:
        rng.bit_generator.state = state
        tr = {}
        w, s, b = simulate_day(staffing, arrivals, rng, svc, sim_time, trace=tr, grid_step=grid_step)
        stage_wait += tr["stage_wait"]
        stage_starts += tr["stage_starts"]
        queued_close += tr["queued_at_close"]
        busy_min += b
        in_system.extend(tr["in_system"])
        grids.append(np.array(tr["grid"], dtype=float))
        served += s
        total_wait += float(w)
        generated += len(arrivals)
    g = min(len(x) for x in grids)
    return {
        "staffing": tuple(staffing), "arrival_mean": arrival_mean,
        "util": busy_min / (n_reps * sim_time * np.array(staffing, dtype=float)),
        "rho": offered_load(staffing, arrival_mean, svc),
        "stage_wait_mean": np.divide(stage_wait, stage_starts, out=np.zeros(5), where=stage_starts > 0),
        "queued_at_close": queued_close / n_reps,
        "in_system": np.array(in_system),
        "queue_grid": np.mean([x[:g] for x in grids], axis=0),
        "grid_times": np.arange(g) * grid_step,
        "sl": served / generated if generated else float("nan"),
        "served_per_day": served / n_reps,
        "generated_per_day": generated / n_reps,
        "wait_min_per_day": total_wait / n_reps,
    }


# =============================================================================================
# 5. Headless command line (python app.py --benchmark / --export DIR)
# =============================================================================================
def _default_specs(n_reps=None, sim_time=None):
    T = PAPER["sim_time"] if sim_time is None else sim_time
    R = PAPER["n_reps"] if n_reps is None else n_reps
    return [(PAPER["theta0"], PAPER["svc"], T, R), (PAPER["theta1"], PAPER["svc"], T, R)]


def _cli(argv=None):
    ap = argparse.ArgumentParser(description="Walk-in clinic staffing DSS (Streamlit app + headless tools).")
    ap.add_argument("--benchmark", action="store_true", help="run the paper scenarios headless and report timings")
    ap.add_argument("--export", metavar="DIR", help="run headless and write the notebook's CSV tables to DIR")
    ap.add_argument("--workers", type=int, default=None, help="worker processes (default: all CPUs)")
    ap.add_argument("--reps", type=int, default=None, help="replications per scenario (default: 30)")
    args = ap.parse_args(argv)

    if not (args.benchmark or args.export):
        # Launch the dashboard with the slate theme locked in (same as `streamlit run app.py`
        # plus theme flags; the app also adapts to Streamlit's default light/dark themes).
        theme = ["--theme.base", "light", "--theme.primaryColor", "#2A78D6",
                 "--theme.backgroundColor", "#F8FAFC", "--theme.secondaryBackgroundColor", "#EEF2F7",
                 "--theme.textColor", "#0F172A"]
        import subprocess
        return subprocess.call([sys.executable, "-m", "streamlit", "run", APP_FILE, *theme])

    def bar(done, total):
        print(f"\r  simulated {done:4d}/{total} plan-scenarios", end="", flush=True)

    specs = _default_specs(n_reps=args.reps)
    workers = resolve_workers(args.workers)
    n_sims = len(CONFIGS) * sum(s[3] for s in specs)
    print(f"Enumerating {len(CONFIGS)} plans x {len(specs)} scenarios x {specs[0][3]} replications "
          f"({n_sims:,} clinic-days) on {workers} worker(s) ...")
    sim0, sim1 = simulate_scenarios(specs, max_workers=workers, progress=bar)
    print(f"\n  done in {sim0['elapsed']:.1f}s ({sim0['elapsed'] / n_sims * 1e3 * workers:.2f} ms per "
          f"clinic-day per core)")

    df = build_design_table(sim0, sim1, PAPER["wages"], PAPER["tau"], PAPER["wait_value"])
    s = robustness_summary(df)
    print(f"\nNominal: {s['n_nominal_feasible']}/625 feasible, front = {s['n_nominal_front']} plans")
    print(f"Robust : {s['n_robust_feasible']}/625 feasible, front = {s['n_robust_front']} plans")
    cn, cr = s["cheapest_nominal"], s["cheapest_robust"]
    if cn is not None and cr is not None:
        print(f"Cheapest nominal plan {cn.plan} ${cn.manpower_cost:,.0f} | cheapest robust plan "
              f"{cr.plan} ${cr.manpower_cost:,.0f} | cost of robustness ${s['premium']:,.0f} "
              f"({s['premium_pct']:.1f}%)")
        print(f"{s['nominal_fail_surge']} of {s['n_nominal_front']} nominal-front plans miss the "
              f"{PAPER['tau']:.0%} floor under theta1.")
    print(f"Highest service level of any plan: theta0 {df.sl0.max():.2%}, theta1 {df.sl1.max():.2%}")

    if args.export:
        os.makedirs(args.export, exist_ok=True)
        cols = ["Rec", "Nur", "Doc", "Pha"]
        robust_all = df[cols].assign(
            waiting_cost=np.round(df.waiting_cost_R, 2), manpower_cost=np.round(df.manpower_cost, 2),
            patients_served=np.round(df.served_R, 2), sl0=np.round(100 * df.sl0, 1),
            sl1=np.round(100 * df.sl1, 1), feasible=df.feasible_R)
        robust_all.to_csv(os.path.join(args.export, "robust_all_625.csv"), index=False)
        rf = df[df.robust_front].sort_values(["manpower_cost", "waiting_cost_R"], kind="mergesort")
        robust_all.loc[rf.index].to_csv(os.path.join(args.export, "robust_pareto_front.csv"), index=False)
        nominal_all = df[cols].assign(
            waiting_cost=np.round(df.wait_cost0, 2), manpower_cost=np.round(df.manpower_cost, 2),
            patients_served=np.round(df.served0, 2), feasible=df.feasible0)
        nominal_all.to_csv(os.path.join(args.export, "nominal_all_625.csv"), index=False)
        nf = df[df.nominal_front].sort_values(["manpower_cost", "wait_cost0"], kind="mergesort")
        nominal_all.loc[nf.index].to_csv(os.path.join(args.export, "nominal_pareto_front.csv"), index=False)
        print(f"Wrote 4 CSV tables to {os.path.abspath(args.export)}")
    return 0


def _inside_streamlit() -> bool:
    if "streamlit" not in sys.modules:
        return False
    try:
        from streamlit import runtime
        return runtime.exists()
    except Exception:
        return False


# =============================================================================================
# 6. Streamlit dashboard (runs only under `streamlit run`; nothing above imports Streamlit)
# =============================================================================================
st = go = pd = make_subplots = None        # UI stack, imported by _load_ui_stack()

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
TH0, TH1 = "θ₀", "θ₁"
STRATEGIES = ("Cheapest feasible", "Balanced (knee)", "High service")
OTHER_PLAN = "Other plan"
STRATEGY_CAPTIONS = {
    "Cheapest feasible": "Lowest staffing cost that clears the floor on both days.",
    "Balanced (knee)": "Knee of the front: closest to the ideal cost-and-wait corner.",
    "High service": "Worst-case waiting within 10% of the best, at the lowest cost.",
    OTHER_PLAN: "Pick a front solution, click any point, or edit the plan under Diagnostics.",
}
PLOTLY_CONFIG = {"displaylogo": False,
                 "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"]}


def _load_ui_stack():
    global st, go, pd, make_subplots
    import pandas
    import plotly.graph_objects
    import streamlit
    from plotly.subplots import make_subplots as _make_subplots
    st, go, pd, make_subplots = streamlit, plotly.graph_objects, pandas, _make_subplots


def _ui_defaults():
    rec_lo, rec_hi, nur_mu, nur_sd, doc_mu, doc_sd, pha_mu, pha_sd = PAPER["svc"]
    w = PAPER["wages"]
    return {"w_rec": int(w[0]), "w_nur": int(w[1]), "w_doc": int(w[2]), "w_pha": int(w[3]),
            "theta0": PAPER["theta0"], "theta1": PAPER["theta1"], "rec_lo": rec_lo, "rec_hi": rec_hi,
            "nur_mu": nur_mu, "nur_sd": nur_sd, "doc_mu": doc_mu, "doc_sd": doc_sd,
            "pha_mu": pha_mu, "pha_sd": pha_sd, "tau_pct": int(round(100 * PAPER["tau"])),
            "reps": PAPER["n_reps"], "shift": int(PAPER["sim_time"]),
            "wait_pct": int(round(100 * PAPER["wait_value"])), "workers": resolve_workers()}


# ---------------------------------------------------------------------------------------------
# Theme, styling and small HTML components
# ---------------------------------------------------------------------------------------------
def _detect_dark() -> bool:
    try:                                        # st.context.theme exists in recent Streamlit versions
        theme = st.context.theme
        kind = theme.get("type") if isinstance(theme, dict) else getattr(theme, "type", None)
        return str(kind).lower() == "dark"
    except Exception:
        return False


def _tokens(dark: bool) -> dict:
    """Chart colours: validated categorical steps (robust = blue, surge = orange), neutral greys
    for context, and the reserved status red for 'fails the floor' (always paired with an icon)."""
    t = {"fail": "#d03b3b", "good": "#0ca30c", "warn": "#fab219", "dark": dark}
    if dark:
        t.update(robust="#3987e5", surge="#d95926", nominal="#94a3b8", context="#475569",
                 faint="#334155", ink2="#cbd5e1", ring="#0e1117")
    else:
        t.update(robust="#2a78d6", surge="#eb6834", nominal="#64748b", context="#94a3b8",
                 faint="#cbd5e1", ink2="#475569", ring="#ffffff")
    return t


def _css(T) -> str:
    return f"""<style>
:root {{ --dss-accent:{T['robust']}; --dss-surge:{T['surge']}; --dss-bad:{T['fail']}; --dss-good:{T['good']};
  --dss-warn:{T['warn']}; --dss-card:rgba(148,163,184,0.09); --dss-card2:rgba(148,163,184,0.16);
  --dss-line:rgba(148,163,184,0.30); --dss-font:{FONT}; }}
html, body, .stApp {{ font-family: var(--dss-font); }}
.stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ font-family: var(--dss-font) !important; }}
[data-testid="stMainBlockContainer"], .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1480px; }}
[data-testid="stSidebar"] [data-testid="stForm"] {{ padding: 0; border: 0; }}
[data-testid="stFormSubmitButton"] button {{ width: 100%; min-height: 2.9rem; font-weight: 650; }}
button[data-baseweb="tab"] p {{ font-weight: 600; font-size: .95rem; }}
.dss-brand {{ display:flex; align-items:center; gap:.6rem; margin:.1rem 0 .4rem; }}
.dss-mark {{ width:34px; height:34px; border-radius:10px; background:var(--dss-accent); color:#fff;
  display:flex; align-items:center; justify-content:center; font-weight:750; font-size:.95rem; letter-spacing:-.02em; }}
.dss-brand-name {{ font-weight:700; font-size:1rem; line-height:1.1; }}
.dss-brand-sub {{ font-size:.75rem; opacity:.65; }}
.dss-side-h {{ font-size:.7rem; text-transform:uppercase; letter-spacing:.09em; font-weight:700; opacity:.62; margin:1.1rem 0 .15rem; }}
.dss-hero {{ margin:0 0 1rem; }}
.dss-eyebrow {{ text-transform:uppercase; letter-spacing:.09em; font-size:.72rem; font-weight:700; color:var(--dss-accent); }}
.dss-title {{ font-size:1.95rem; font-weight:720; letter-spacing:-.02em; line-height:1.15; margin:.2rem 0 .4rem; }}
.dss-sub {{ opacity:.74; font-size:.97rem; max-width:78ch; margin:0; line-height:1.5; }}
.dss-chips {{ display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.85rem; }}
.dss-chip {{ border:1px solid var(--dss-line); background:var(--dss-card); border-radius:999px; padding:.18rem .7rem; font-size:.8rem; white-space:nowrap; }}
.dss-chip b {{ font-weight:650; }}
.dss-chip.run {{ border-color:transparent; background:transparent; opacity:.7; }}
.dss-kpis {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:.75rem; margin:.3rem 0 1.1rem; }}
.dss-kpi {{ border:1px solid var(--dss-line); background:var(--dss-card); border-radius:14px; padding:.85rem 1rem .8rem; }}
.dss-kpi.accent {{ box-shadow: inset 3px 0 0 var(--dss-accent); }}
.dss-kpi.bad {{ box-shadow: inset 3px 0 0 var(--dss-bad); }}
.dss-kpi-label {{ font-size:.8rem; opacity:.72; font-weight:560; }}
.dss-kpi-value {{ font-size:1.6rem; font-weight:700; letter-spacing:-.02em; line-height:1.2; margin-top:.2rem; }}
.dss-kpi-unit {{ font-size:.92rem; font-weight:500; opacity:.62; margin-left:.15rem; }}
.dss-kpi-foot {{ font-size:.78rem; opacity:.7; margin-top:.25rem; line-height:1.35; }}
.dss-h {{ font-size:1.05rem; font-weight:700; letter-spacing:-.01em; margin:.5rem 0 .1rem; }}
.dss-note {{ font-size:.83rem; opacity:.7; margin:0 0 .45rem; line-height:1.45; }}
.dss-callout {{ border:1px solid var(--dss-line); box-shadow: inset 3px 0 0 var(--dss-accent); background:var(--dss-card);
  border-radius:10px; padding:.75rem 1rem; font-size:.92rem; line-height:1.55; margin:.2rem 0 1rem; }}
.dss-callout.warn {{ box-shadow: inset 3px 0 0 var(--dss-warn); }}
.dss-callout.bad {{ box-shadow: inset 3px 0 0 var(--dss-bad); }}
.dss-pill {{ display:inline-flex; align-items:center; gap:.4rem; border-radius:999px; padding:.18rem .7rem; font-size:.8rem; font-weight:600; border:1px solid var(--dss-line); }}
.dss-pill.good {{ background:rgba(12,163,12,.10); border-color:rgba(12,163,12,.35); }}
.dss-pill.bad {{ background:rgba(208,59,59,.10); border-color:rgba(208,59,59,.38); }}
.dss-pill.neutral {{ background:var(--dss-card); }}
.dss-pill-icon {{ font-weight:800; }}
.dss-pill.good .dss-pill-icon {{ color:var(--dss-good); }}
.dss-pill.bad .dss-pill-icon {{ color:var(--dss-bad); }}
.dss-staff {{ display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:.5rem; margin:.7rem 0 .8rem; }}
.dss-role {{ border:1px solid var(--dss-line); background:var(--dss-card); border-radius:12px; padding:.55rem .65rem .6rem; }}
.dss-role-name {{ font-size:.72rem; opacity:.7; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.dss-role-count {{ font-size:1.45rem; font-weight:720; line-height:1.15; }}
.dss-pips {{ display:flex; gap:3px; margin-top:.3rem; }}
.dss-pips i {{ flex:1; height:5px; border-radius:3px; background:var(--dss-card2); }}
.dss-pips i.on {{ background:var(--dss-accent); }}
.dss-dl {{ display:grid; grid-template-columns:1fr auto; gap:.32rem 1rem; font-size:.88rem; margin:.2rem 0 .6rem; }}
.dss-dl dt {{ opacity:.72; }}
.dss-dl dd {{ margin:0; font-weight:620; text-align:right; font-variant-numeric:tabular-nums; }}
.dss-flow {{ display:flex; flex-wrap:wrap; align-items:center; gap:.45rem; margin:.7rem 0 1.1rem; }}
.dss-stage {{ border:1px solid var(--dss-line); background:var(--dss-card); border-radius:10px; padding:.45rem .75rem; font-size:.86rem; font-weight:600; }}
.dss-stage small {{ display:block; font-weight:500; opacity:.65; font-size:.74rem; }}
.dss-arrow {{ opacity:.4; }}
.dss-table {{ width:100%; border-collapse:collapse; font-size:.86rem; margin:.2rem 0 .8rem; }}
.dss-table th {{ text-align:left; font-weight:600; opacity:.7; font-size:.76rem; padding:.35rem .5rem; border-bottom:1px solid var(--dss-line); }}
.dss-table td {{ padding:.42rem .5rem; border-bottom:1px solid var(--dss-line); font-variant-numeric:tabular-nums; }}
.dss-table td.num, .dss-table th.num {{ text-align:right; }}
.dss-muted {{ opacity:.65; }}
</style>"""


def _usd(x, dec=0) -> str:
    """Dollar amount for HTML snippets (&#36; keeps Streamlit's markdown from reading $...$ as maths)."""
    s = f"{abs(x):,.{dec}f}"
    return ("&minus;" if x < 0 else "") + "&#36;" + s


def _pct(x, dec=2) -> str:
    return f"{100 * x:.{dec}f}%"


def _workers(n) -> str:
    return f"{n} worker process" + ("" if n == 1 else "es")


def _html(s: str):
    st.markdown(s, unsafe_allow_html=True)


def _kpi(label, value, unit="", foot="", tone=""):
    return (f'<div class="dss-kpi {tone}"><div class="dss-kpi-label">{label}</div>'
            f'<div class="dss-kpi-value">{value}<span class="dss-kpi-unit">{unit}</span></div>'
            f'<div class="dss-kpi-foot">{foot}</div></div>')


def _kpi_row(items):
    _html('<div class="dss-kpis">' + "".join(items) + "</div>")


def _pill(ok, text):
    if ok is None:
        return f'<span class="dss-pill neutral">{text}</span>'
    return (f'<span class="dss-pill {"good" if ok else "bad"}"><span class="dss-pill-icon">'
            f'{"✓" if ok else "✕"}</span>{text}</span>')


def _section(title, note=None):
    _html(f'<div class="dss-h">{title}</div>' + (f'<div class="dss-note">{note}</div>' if note else ""))


def _callout(body, tone=""):
    _html(f'<div class="dss-callout {tone}">{body}</div>')


def _staff_tiles(staffing):
    tiles = []
    for role, n in zip(("Reception", "Nurses", "Doctors", "Pharmacy"), staffing):
        pips = "".join(f'<i class="{"on" if k < n else ""}"></i>' for k in range(max(STAFF_LEVELS)))
        tiles.append(f'<div class="dss-role"><div class="dss-role-name">{role}</div>'
                     f'<div class="dss-role-count">{n}</div><div class="dss-pips">{pips}</div></div>')
    return '<div class="dss-staff">' + "".join(tiles) + "</div>"


def _dl(rows):
    return '<dl class="dss-dl">' + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows) + "</dl>"


def _rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def _stretch(fn) -> dict:
    """Full-width keyword for this Streamlit version: `width="stretch"` on newer releases,
    `use_container_width=True` on older ones (avoids deprecation notices on either)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    width = params.get("width")
    if width is not None and isinstance(width.default, str):
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


def _chart(fig, key=None, on_select=None):
    kwargs = {"theme": "streamlit", "config": PLOTLY_CONFIG, **_stretch(st.plotly_chart)}
    if key is not None:
        kwargs["key"] = key
    if on_select is not None:
        kwargs.update(on_select=on_select, selection_mode="points")
    try:
        return st.plotly_chart(fig, **kwargs)
    except TypeError:                        # Streamlit without chart selections
        kwargs.pop("on_select", None)
        kwargs.pop("selection_mode", None)
        return st.plotly_chart(fig, **kwargs)


def _base_layout(fig, height, **extra):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=48, b=8), font=dict(family=FONT, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        hoverlabel=dict(font=dict(family=FONT, size=12)), **extra)
    return fig


# ---------------------------------------------------------------------------------------------
# Parameters, caching and the simulation run
# ---------------------------------------------------------------------------------------------
def _params(v) -> dict:
    return {
        "wages": (float(v["w_rec"]), float(v["w_nur"]), float(v["w_doc"]), float(v["w_pha"])),
        "theta0": round(float(v["theta0"]), 4), "theta1": round(float(v["theta1"]), 4),
        "svc": tuple(round(float(v[k]), 4) for k in
                     ("rec_lo", "rec_hi", "nur_mu", "nur_sd", "doc_mu", "doc_sd", "pha_mu", "pha_sd")),
        "tau": int(v["tau_pct"]) / 100.0, "sim_time": float(v["shift"]), "n_reps": int(v["reps"]),
        "wait_value": int(v["wait_pct"]) / 100.0, "workers": int(v["workers"]),
    }


_FINGERPRINT = None


def _scenario_key(P, theta):
    global _FINGERPRINT
    if _FINGERPRINT is None:
        _FINGERPRINT = engine_fingerprint()
    return ("clinic-scenario-v1", float(theta), P["svc"], P["sim_time"], P["n_reps"], _FINGERPRINT)


def _keys(P):
    return _scenario_key(P, P["theta0"]), _scenario_key(P, P["theta1"])


def _cached_api():
    @st.cache_data(show_spinner=False, persist="disk", max_entries=48)
    def scenario_store(key, _payload=None):
        """Simulated scenarios (all 625 plans x R replications), keyed only by what changes the
        simulation. Called with `_payload` it stores it; without, it is a pure lookup and a miss
        raises LookupError (Streamlit never caches exceptions). The simulation itself runs outside
        this function so it can drive a live progress bar."""
        if _payload is None:
            raise LookupError(key)
        return _payload

    @st.cache_data(show_spinner=False, max_entries=64)
    def design_table(key0, key1, wages, tau, wait_value):
        return build_design_table(scenario_store(key0), scenario_store(key1), wages, tau, wait_value)

    @st.cache_data(show_spinner=False, max_entries=512)
    def diagnostics(staffing, key):
        _, theta, svc, sim_time, n_reps, _ = key
        return diagnose_plan(staffing, theta, svc, sim_time, n_reps)

    return {"store": scenario_store, "table": design_table, "diag": diagnostics}


def _is_cached(api, key) -> bool:
    try:
        api["store"](key)
        return True
    except LookupError:
        return False


def _validate(P):
    errors, warnings = [], []
    rec_lo, rec_hi = P["svc"][0], P["svc"][1]
    if rec_hi < rec_lo:
        errors.append("Reception service time: the maximum must be at least the minimum.")
    for name, mu in (("Nurse", P["svc"][2]), ("Doctor", P["svc"][4]), ("Pharmacy", P["svc"][6])):
        if mu <= 0:
            errors.append(f"{name} mean service time must be positive.")
    if P["theta1"] > P["theta0"]:
        warnings.append(f"The surge scenario {TH1} = {P['theta1']:.1f} min brings fewer arrivals than the "
                        f"nominal {TH0} = {P['theta0']:.1f} min. The robust evaluation still takes the worse "
                        f"of the two days, but the labels 'nominal' and 'surge' are swapped.")
    return errors, warnings


def _run(P, api, slot):
    """Simulate whichever scenarios are not cached yet, with a live progress bar."""
    todo = [k for k in dict.fromkeys(_keys(P)) if not _is_cached(api, k)]
    t0 = time.perf_counter()
    workers = resolve_workers(P["workers"])
    days = len(todo) * len(CONFIGS) * P["n_reps"]
    if todo:
        with slot.container():
            _html('<div class="dss-callout"><b>Running clinic optimization &amp; stress test.</b> '
                  f'Simulating {len(CONFIGS)} staffing plans × {len(todo)} scenario(s) × {P["n_reps"]} '
                  f'replications = {days:,} clinic days on {_workers(workers)}.</div>')
            bar = st.progress(0.0, text="Starting worker processes…")

        def progress(done, total):
            elapsed = time.perf_counter() - t0
            left = elapsed / done * (total - done) if done else 0.0
            bar.progress(min(done / total, 1.0),
                         text=f"{done:,} of {total:,} plan-scenarios simulated · {elapsed:.1f} s elapsed"
                              f" · about {left:.0f} s left")

        results = simulate_scenarios([(k[1], k[2], k[3], k[4]) for k in todo],
                                     max_workers=workers, progress=progress)
        for key, res in zip(todo, results):
            api["store"](key, _payload=res)
        slot.empty()
    seconds = time.perf_counter() - t0
    st.session_state["last_run"] = {"days": days, "seconds": seconds, "workers": workers,
                                     "scenarios": len(todo)}
    if todo:
        st.toast(f"Simulated {days:,} clinic days in {seconds:.1f} s on {_workers(workers)}.")
    else:
        st.toast("Re-priced from cached simulations: no new runs were needed.")


# ---------------------------------------------------------------------------------------------
# Selection state (shared by the strategy explorer, the chart and the diagnostics tab)
# ---------------------------------------------------------------------------------------------
def _init_state():
    ss = st.session_state
    for key, value in (("form_ver", 0), ("strategy", STRATEGIES[1]), ("custom_id", None),
                       ("chart_nonce", 0), ("show_all", False), ("logy", False), ("logy2", True),
                       ("last_sel_id", None), ("_presets", {})):
        if key not in ss:
            ss[key] = value


def _preset_name(pid):
    for name, preset_id in st.session_state.get("_presets", {}).items():
        if preset_id == pid:
            return name
    return None


def _choose(pid, from_chart=False):
    ss = st.session_state
    ss["custom_id"] = int(pid)
    ss["strategy"] = _preset_name(pid) or OTHER_PLAN
    if not from_chart:
        ss["chart_nonce"] += 1                  # clears the chart's own click selection


def _on_strategy():
    ss = st.session_state
    if ss.get("strategy") == OTHER_PLAN and ss.get("last_sel_id") is not None:
        ss["custom_id"] = ss["last_sel_id"]
    ss["chart_nonce"] += 1


def _on_front_pick():
    pick = st.session_state.get("front_pick")
    if pick is not None:
        _choose(pick)


def _on_dx_change():
    ss = st.session_state
    cfg = tuple(int(ss[f"dx_{r}"]) for r in ROLE_SHORT)
    _choose(CONFIG_INDEX[cfg])


def _on_show_all():
    st.session_state["logy"] = bool(st.session_state.get("show_all"))


def _picked_plan(state, key):
    """Plan id from a Plotly click event (customdata[0]; falls back to the trace/point index)."""
    try:
        sel = state["selection"] if isinstance(state, dict) else state.selection
        points = sel["points"] if isinstance(sel, dict) else sel.points
    except Exception:
        return None
    lookup = st.session_state.get("_chart_ids", {}).get(key, {})
    for pt in points or []:
        cd = pt.get("customdata") if isinstance(pt, dict) else None
        if cd is not None:
            try:
                return int(cd[0] if isinstance(cd, (list, tuple)) else cd)
            except (TypeError, ValueError):
                pass
        try:
            return int(lookup[int(pt["curve_number"])][int(pt["point_index"])])
        except Exception:
            continue
    return None


def _on_chart_select(key):
    pid = _picked_plan(st.session_state.get(key), key)
    if pid is not None:
        _choose(pid, from_chart=True)


def _selected_id(df, presets) -> int:
    ss = st.session_state
    strategy = ss.get("strategy")
    if strategy in presets:
        sel = presets[strategy]
    else:
        sel = ss.get("custom_id")
        if sel is None:
            sel = presets.get(STRATEGIES[1], int(df["sl_min"].idxmax()))
    ss["last_sel_id"] = int(sel)
    return int(sel)


# ---------------------------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------------------------
def _sidebar():
    ss = st.session_state
    d = _ui_defaults()
    v = ss["form_ver"]                         # bumping it re-creates the inputs at their defaults
    k = lambda name: f"in_{name}_{v}"          # noqa: E731
    cpus = resolve_workers()
    with st.sidebar:
        _html('<div class="dss-brand"><div class="dss-mark">Rx</div><div><div class="dss-brand-name">'
              'Clinic Staffing DSS</div><div class="dss-brand-sub">Robust staffing under demand uncertainty'
              '</div></div></div>')
        with st.form("clinic_config", border=False):
            vals = {}
            _html('<div class="dss-side-h">Daily staff wages</div>')
            vals["w_rec"] = st.slider("Receptionist", 50, 400, d["w_rec"], 5, format="$%d", key=k("w_rec"))
            vals["w_nur"] = st.slider("Nurse", 50, 500, d["w_nur"], 5, format="$%d", key=k("w_nur"),
                                      help="The nurse wage also prices patient waiting time (see Advanced).")
            vals["w_doc"] = st.slider("Doctor", 100, 800, d["w_doc"], 5, format="$%d", key=k("w_doc"))
            vals["w_pha"] = st.slider("Pharmacist", 50, 500, d["w_pha"], 5, format="$%d", key=k("w_pha"))

            _html('<div class="dss-side-h">Arrival scenarios</div>')
            vals["theta0"] = st.slider(f"Nominal day {TH0}: mean minutes between arrivals", 2.5, 10.0,
                                       float(d["theta0"]), 0.1, format="%.1f min", key=k("theta0"),
                                       help="About shift length ÷ θ patients a day: 96 at 5.0 min over 8 h.")
            vals["theta1"] = st.slider(f"Surge day {TH1}: mean minutes between arrivals", 2.5, 10.0,
                                       float(d["theta1"]), 0.1, format="%.1f min", key=k("theta1"),
                                       help="Shorter gaps mean more patients (120 a day at 4.0 min over 8 h). "
                                            "Inter-arrival SD is 1 min, as in the paper.")

            _html('<div class="dss-side-h">Service times (minutes)</div>')
            c1, c2 = st.columns(2)
            vals["rec_lo"] = c1.number_input("Reception min", 0.5, 30.0, float(d["rec_lo"]), 0.5,
                                             format="%.1f", key=k("rec_lo"),
                                             help="Check-in and check-out times are Uniform(min, max).")
            vals["rec_hi"] = c2.number_input("Reception max", 0.5, 30.0, float(d["rec_hi"]), 0.5,
                                             format="%.1f", key=k("rec_hi"))
            for role, name, top_mu, top_sd in (("nur", "Nurse", 60.0, 20.0), ("doc", "Doctor", 90.0, 30.0),
                                               ("pha", "Pharmacy", 60.0, 20.0)):
                vals[f"{role}_mu"] = c1.number_input(f"{name} mean", 0.5, top_mu, float(d[f"{role}_mu"]), 0.5,
                                                     format="%.1f", key=k(f"{role}_mu"))
                vals[f"{role}_sd"] = c2.number_input(f"{name} SD", 0.0, top_sd, float(d[f"{role}_sd"]), 0.5,
                                                     format="%.1f", key=k(f"{role}_sd"),
                                                     help="Normal service times, floored at 1e-6 min as in the paper."
                                                     if role == "nur" else None)

            _html('<div class="dss-side-h">Managerial policy</div>')
            vals["tau_pct"] = st.slider("Minimum service level τ", 50, 99, d["tau_pct"], 1, format="%d%%",
                                        key=k("tau"), help="Share of the day's arrivals that must complete all "
                                        "five stages before closing, on both days.")
            with st.expander("Advanced simulation settings"):
                vals["reps"] = st.slider("Replications per scenario", 5, 100, d["reps"], 5, key=k("reps"),
                                         help="Paper: 30. Runtime grows linearly.")
                vals["shift"] = st.slider("Shift length", 240, 720, d["shift"], 30, format="%d min", key=k("shift"))
                vals["wait_pct"] = st.slider("Value of a waiting minute", 0, 300, d["wait_pct"], 5, format="%d%%",
                                             key=k("wait"), help="As a share of a nurse's per-minute wage "
                                             "(paper: 50%). Rescales waiting costs; does not re-simulate.")
                vals["workers"] = st.number_input("Worker processes", 1, cpus, min(d["workers"], cpus), 1,
                                                  key=k("workers"), help=f"{cpus} CPU(s) available.")
            submitted = st.form_submit_button("Run Clinic Optimization & Stress Test", type="primary",
                                              **_stretch(st.form_submit_button))
        st.button("Reset inputs to paper defaults", on_click=_reset_inputs, **_stretch(st.button))
        _html('<div class="dss-note" style="margin-top:.6rem">Wages, τ and the waiting-cost value re-price '
              'cached simulations instantly. Arrival, service, shift and replication changes trigger a '
              'new parallel run.</div>')
    return submitted, vals


def _reset_inputs():
    st.session_state["form_ver"] += 1


# ---------------------------------------------------------------------------------------------
# Header and landing page
# ---------------------------------------------------------------------------------------------
def _header(P=None):
    chips = ""
    if P is not None:
        items = [f"<b>{TH0}</b> {P['theta0']:.1f} min", f"<b>{TH1}</b> {P['theta1']:.1f} min",
                 f"<b>τ</b> {100 * P['tau']:.0f}%", f"<b>{P['n_reps']}</b> replications",
                 f"<b>{P['sim_time'] / 60:g} h</b> shift",
                 "<b>Wages</b> " + " · ".join(f"{r[0]} {_usd(x)}" for r, x in zip(ROLES, P["wages"]))]
        chips = "".join(f'<span class="dss-chip">{c}</span>' for c in items)
        run = st.session_state.get("last_run")
        if run and run.get("scenarios"):
            chips += (f'<span class="dss-chip run">{run["days"]:,} clinic days simulated in {run["seconds"]:.1f} s '
                      f'on {_workers(run["workers"])}</span>')
        chips = f'<div class="dss-chips">{chips}</div>'
    _html('<div class="dss-hero"><div class="dss-eyebrow">Robust staffing · walk-in clinic</div>'
          '<div class="dss-title">Staffing plans that hold up on a surge day</div>'
          f'<p class="dss-sub">All {len(CONFIGS)} plans in {{1,…,5}}⁴ are simulated on a nominal day ({TH0}) '
          f'and a surge day ({TH1}). A plan is <b>robustly feasible</b> when it serves at least τ of arrivals '
          'on both days; its waiting cost and patients served are judged on its worse day.</p>' + chips + "</div>")


def _landing(P):
    stages = [("Check-in", "receptionist"), ("Triage", "nurse"), ("Consultation", "doctor"),
              ("Dispensing", "pharmacist"), ("Check-out", "receptionist")]
    flow = '<span class="dss-arrow">→</span>'.join(
        f'<div class="dss-stage">{a}<small>{b}</small></div>' for a, b in stages)
    days = 2 * len(CONFIGS) * P["n_reps"]
    _html(f'<div class="dss-flow">{flow}</div>')
    _kpi_row([
        _kpi("Staffing plans", f"{len(CONFIGS)}", "", "1–5 of each role, enumerated exhaustively"),
        _kpi("Scenarios", "2", "", f"nominal {TH0} and surge {TH1}, worst case taken"),
        _kpi("Clinic days per run", f"{days:,}", "", f"{P['n_reps']} common-random-number replications"),
        _kpi("Worker processes", f"{resolve_workers(P['workers'])}", "", "the enumeration runs in parallel"),
    ])
    _callout("Set wages, demand and service assumptions in the sidebar, then press "
             "<b>Run Clinic Optimization &amp; Stress Test</b>. Results are cached (also on disk), so "
             "switching tabs, choosing plans or changing wages and τ never re-runs the simulation.")


# ---------------------------------------------------------------------------------------------
# Tab 1 — Pareto frontier & strategy explorer
# ---------------------------------------------------------------------------------------------
def _hover_rows(sub):
    """customdata rows: [id, plan, tag, sl0, sl1, served_R, busiest role, its utilization, cost, wait_R]."""
    both = ~sub.feasible0 & ~sub.feasible1
    conditions = [sub.robust_front.to_numpy(), sub.feasible_R.to_numpy(), both.to_numpy(),
                  ~sub.feasible1.to_numpy()]
    tag = np.select(conditions,
                    [(" · robust front " + sub.robust_label).to_numpy(dtype=object), " · dominated",
                     " · misses τ on both days", f" · misses τ on {TH1}"],
                    default=f" · misses τ on {TH0}")
    cols = (sub.id.astype(int).tolist(), sub.plan.tolist(), [str(t) for t in tag], sub.sl0.tolist(),
            sub.sl1.tolist(), sub.served_R.tolist(), sub.bottleneck.tolist(), sub.bottleneck_util.tolist(),
            sub.manpower_cost.tolist(), sub.waiting_cost_R.tolist())
    return [list(r) for r in zip(*cols)]


HOVER_PLAN = ("<b>%{customdata[1]}</b>%{customdata[2]}<br>"
              "Manpower %{customdata[8]:$,.0f}/day · worst-case wait %{customdata[9]:$,.2f}/day<br>"
              f"Service level {TH0} %{{customdata[3]:.2%}} · {TH1} %{{customdata[4]:.2%}}<br>"
              "Patients served on the worse day %{customdata[5]:.1f}<br>"
              "Busiest role that day: %{customdata[6]} (%{customdata[7]:.0%})<extra></extra>")


def _fig_frontier(df, sel, T, show_all, log_y, key):
    fig = go.Figure()
    ids_by_curve = {}

    def add(trace, ids):
        ids_by_curve[len(fig.data)] = [int(i) for i in ids]
        fig.add_trace(trace)

    keep = dict(selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1)))
    if show_all:
        sub = df[~df.feasible_R]
        add(go.Scatter(x=sub.manpower_cost, y=sub.waiting_cost_R, mode="markers",
                       name=f"Misses τ on some day ({len(sub)})",
                       marker=dict(symbol="circle-open", size=7, color=T["faint"], line=dict(width=1.2)),
                       customdata=_hover_rows(sub), hovertemplate=HOVER_PLAN, **keep), sub.id)
    sub = df[df.feasible_R & ~df.robust_front]
    add(go.Scatter(x=sub.manpower_cost, y=sub.waiting_cost_R, mode="markers",
                   name=f"Robustly feasible, dominated ({len(sub)})",
                   marker=dict(size=9, color=T["context"], line=dict(width=2, color=T["ring"])),
                   customdata=_hover_rows(sub), hovertemplate=HOVER_PLAN, **keep), sub.id)
    rf = df[df.robust_front].sort_values(["manpower_cost", "waiting_cost_R"], kind="mergesort")
    eff = rf[rf.robust_2d]
    fig.add_trace(go.Scatter(x=eff.manpower_cost, y=eff.waiting_cost_R, mode="lines", hoverinfo="skip",
                             line=dict(color=T["robust"], width=2), showlegend=False))
    ids_by_curve[len(fig.data) - 1] = []
    for sub, name, symbol in ((eff, f"Robust Pareto front ({len(rf)})", "circle"),
                              (rf[~rf.robust_2d], "Pareto thanks to patients served", "circle-open")):
        if sub.empty:
            continue
        add(go.Scatter(x=sub.manpower_cost, y=sub.waiting_cost_R, mode="markers+text", name=name,
                       text=sub.robust_label, textposition="top center", textfont=dict(size=11),
                       marker=dict(symbol=symbol, size=12, color=T["robust"],
                                   line=dict(width=2.5 if symbol == "circle-open" else 2,
                                             color=T["robust"] if symbol == "circle-open" else T["ring"])),
                       customdata=_hover_rows(sub), hovertemplate=HOVER_PLAN, **keep), sub.id)
    r = df.loc[sel]
    if show_all or r.feasible_R:
        fig.add_trace(go.Scatter(x=[r.manpower_cost], y=[r.waiting_cost_R], mode="markers",
                                 name="Selected plan", hoverinfo="skip",
                                 marker=dict(symbol="circle-open", size=24, color=T["ink2"], line=dict(width=2))))
        ids_by_curve[len(fig.data) - 1] = []
    st.session_state["_chart_ids"] = {key: ids_by_curve}
    _base_layout(fig, 540, hovermode="closest", clickmode="event+select", dragmode="zoom",
                 uirevision=f"frontier-{show_all}-{log_y}")
    fig.update_xaxes(title_text="Manpower cost ($ per day)", tickprefix="$", tickformat=",.0f", zeroline=False)
    fig.update_yaxes(title_text="Worst-case waiting cost ($ per day)", tickprefix="$", tickformat=",.2~f",
                     type="log" if log_y else "linear", rangemode="normal" if log_y else "tozero",
                     zeroline=False)
    return fig


def _feasibility_pill(row):
    if row.feasible_R:
        return _pill(True, "Robustly feasible")
    if not row.feasible0 and not row.feasible1:
        return _pill(False, "Misses τ on both days")
    return _pill(False, f"Misses τ on the nominal day ({TH0})" if not row.feasible0
                 else f"Misses τ on the surge day ({TH1})")


def _plan_card(row, P):
    ok = bool(row.feasible_R)
    status = _feasibility_pill(row)
    where = (_pill(None, f"Robust front · solution {row.robust_label}") if row.robust_front
             else _pill(None, "Dominated") if ok else "")
    rows = [("Manpower cost", _usd(row.manpower_cost) + "/day"),
            ("Worst-case waiting cost", _usd(row.waiting_cost_R, 2) + "/day"),
            ("Total daily cost (worse day)", _usd(row.total_cost_R) + "/day"),
            (f"Service level, nominal {TH0}", _pct(row.sl0)),
            (f"Service level, surge {TH1}", _pct(row.sl1)),
            ("Patients served, worse day", f"{row.served_R:.1f}"),
            ("Busiest role, worse day", f"{row.bottleneck} · {row.bottleneck_util:.0%}")]
    _html(f'<div style="display:flex;gap:.4rem;flex-wrap:wrap;margin:.2rem 0 .1rem">{status}{where}</div>'
          + _staff_tiles((int(row.Rec), int(row.Nur), int(row.Doc), int(row.Pha))) + _dl(rows))


def _front_table(rf):
    return pd.DataFrame({
        "Sol.": rf.robust_label, "Rec": rf.Rec, "Nur": rf.Nur, "Doc": rf.Doc, "Pha": rf.Pha,
        "Manpower ($/day)": rf.manpower_cost, "Worst-case wait ($/day)": rf.waiting_cost_R.round(2),
        "Served (worse day)": rf.served_R.round(2), f"SL {TH0} (%)": (100 * rf.sl0).round(2),
        f"SL {TH1} (%)": (100 * rf.sl1).round(2), "Busiest role (worse day)": rf.bottleneck,
    })


def _table_config(table):
    nc = st.column_config.NumberColumn
    config = {"Manpower ($/day)": nc(format="$%.0f"), "Worst-case wait ($/day)": nc(format="$%.2f"),
            "Served (worse day)": nc(format="%.2f"), f"SL {TH0} (%)": nc(format="%.2f"),
            f"SL {TH1} (%)": nc(format="%.2f"), "Wait θ₀ ($/day)": nc(format="$%.2f"),
            "Wait θ₁ ($/day)": nc(format="$%.2f")}
    return {name: cfg for name, cfg in config.items() if name in table.columns}


def _min_visit_minutes(P):
    rec_lo, rec_hi, nur_mu, _, doc_mu, _, pha_mu, _ = P["svc"]
    return (rec_lo + rec_hi) + nur_mu + doc_mu + pha_mu


def _tab_frontier(df, P, T, presets):
    ss = st.session_state
    rf = df[df.robust_front].sort_values(["manpower_cost", "waiting_cost_R"], kind="mergesort")
    n_feas = int(df.feasible_R.sum())
    ceiling = float(df.sl_min.max())
    visit = _min_visit_minutes(P)
    cheapest = rf.iloc[0] if len(rf) else None
    _kpi_row([
        _kpi("Robustly feasible plans", f"{n_feas}", f"/ {len(df)}",
             f"serve ≥ {100 * P['tau']:.0f}% of arrivals on both days", "accent" if n_feas else "bad"),
        _kpi("Robust Pareto front", f"{len(rf)}", "plans",
             "non-dominated in cost, worst-case wait and patients served"),
        _kpi("Cheapest robust plan", _usd(cheapest.manpower_cost) if cheapest is not None else "—",
             "/day" if cheapest is not None else "", cheapest.plan if cheapest is not None else "no plan qualifies"),
        _kpi("Service-level ceiling", _pct(ceiling), "",
             f"best worse-day level of any plan; arrivals in the last ~{visit:.0f} min cannot finish"),
    ])
    if n_feas == 0:
        _callout(f"<b>No plan reaches τ = {100 * P['tau']:.0f}% on both days.</b> The best worse-day service "
                 f"level of any plan is {_pct(ceiling)}. A patient needs about {visit:.0f} minutes of pure "
                 f"service to pass all five stages, so arrivals in the last ~{visit:.0f} minutes of the "
                 f"{P['sim_time']:.0f}-minute shift cannot finish however many staff are on duty. Lower τ or "
                 "lengthen the shift (Advanced) to see a frontier.", "bad")

    left, right = st.columns([1.75, 1], gap="large")
    with right:
        _section("Strategy explorer", "Named strategies are points on the robust front.")
        options = [s for s in STRATEGIES if s in presets] + [OTHER_PLAN]
        if ss.get("strategy") not in options:
            ss["strategy"] = options[0]
        st.radio("Strategy", options, key="strategy", on_change=_on_strategy,
                 captions=[STRATEGY_CAPTIONS[o] for o in options], label_visibility="collapsed")
        sel = _selected_id(df, presets)
        front_ids = [int(i) for i in rf.id]
        labels = {int(r.id): f"{r.robust_label} · {r.plan} · ${r.manpower_cost:,.0f}/day" for _, r in rf.iterrows()}
        ss["front_pick"] = sel if sel in labels else None
        st.selectbox("Robust front solution", front_ids, index=None, key="front_pick",
                     format_func=lambda i: labels.get(i, str(i)), on_change=_on_front_pick,
                     placeholder="Selected plan is not on the robust front")
        _plan_card(df.loc[sel], P)

    with left:
        t1, t2, _ = st.columns([1.2, 1, 2])
        t1.toggle("Show infeasible plans", key="show_all", on_change=_on_show_all)
        t2.toggle("Log scale", key="logy")
        key = f"frontier_{ss['chart_nonce']}"
        fig = _fig_frontier(df, sel, T, ss["show_all"], ss["logy"], key)
        _chart(fig, key=key, on_select=functools.partial(_on_chart_select, key))
        st.caption("Click any point to inspect that plan. Filled blue points trade cost against worst-case "
                   "waiting; hollow blue points are on the Pareto front because they serve more patients on "
                   "their worse day (the paper's three-objective definition). Hover for details.")

    if len(rf):
        _section("Robust Pareto front", "Worst case across both scenarios; same layout as the paper's table.")
        front = _front_table(rf)
        st.dataframe(front, hide_index=True, column_config=_table_config(front), **_stretch(st.dataframe))
        c1, c2, _ = st.columns([1, 1, 3])
        c1.download_button("Download CSV", front.to_csv(index=False).encode(),
                           file_name="robust_pareto_front.csv", mime="text/csv", **_stretch(st.download_button))
        c2.download_button("Download LaTeX table", robust_front_latex(df, P["n_reps"]).encode(),
                           file_name="robust_pareto_front.tex", mime="text/plain", **_stretch(st.download_button))
    with st.expander(f"All {len(df)} plans (table view)"):
        full = pd.DataFrame({
            "Plan": df.plan, "Manpower ($/day)": df.manpower_cost,
            "Wait θ₀ ($/day)": df.wait_cost0.round(2), "Wait θ₁ ($/day)": df.wait_cost1.round(2),
            f"SL {TH0} (%)": (100 * df.sl0).round(2), f"SL {TH1} (%)": (100 * df.sl1).round(2),
            "Robustly feasible": df.feasible_R, "Robust front": df.robust_label.replace("", "—"),
            "Busiest role (worse day)": df.bottleneck})
        st.dataframe(full, hide_index=True, column_config=_table_config(full), height=380,
                     **_stretch(st.dataframe))
        st.download_button("Download all plans (CSV)", full.to_csv(index=False).encode(),
                           file_name="all_625_plans.csv", mime="text/csv")


# ---------------------------------------------------------------------------------------------
# Tab 2 — Nominal vs robust
# ---------------------------------------------------------------------------------------------
def _fig_fronts(df, S, T, log_y):
    fig = go.Figure()
    nom = df[df.nominal_front].sort_values(["manpower_cost", "wait_cost0"], kind="mergesort")
    rob = df[df.robust_front].sort_values(["manpower_cost", "waiting_cost_R"], kind="mergesort")
    nom_hover = ("<b>%{customdata[0]}</b> · nominal front<br>Manpower %{x:$,.0f}/day · "
                 f"{TH0} wait %{{y:$,.2f}}/day<br>Surge day {TH1}: service level %{{customdata[1]:.2%}} "
                 "%{customdata[2]} · wait %{customdata[3]:$,.2f}/day<extra></extra>")
    nom_cd = [[r.plan, float(r.sl1), "✓" if r.feasible1 else "✕ below τ", float(r.wait_cost1)]
              for _, r in nom.iterrows()]
    eff = nom[nom.nominal_2d]
    fig.add_trace(go.Scatter(x=eff.manpower_cost, y=eff.wait_cost0, mode="lines", hoverinfo="skip",
                             line=dict(color=T["context"], width=2), showlegend=False))
    fig.add_trace(go.Scatter(x=nom.manpower_cost, y=nom.wait_cost0, mode="markers",
                             name=f"Nominal front, {TH0} only ({len(nom)})", customdata=nom_cd,
                             hovertemplate=nom_hover,
                             marker=dict(symbol="circle-open", size=11, color=T["context"], line=dict(width=2.5))))
    fails = nom[~nom.feasible1]
    if len(fails):
        fig.add_trace(go.Scatter(x=fails.manpower_cost, y=fails.wait_cost0, mode="markers",
                                 name=f"✕ Misses τ on the surge day ({len(fails)})", hoverinfo="skip",
                                 marker=dict(symbol="x", size=12, color=T["fail"],
                                             line=dict(width=1.5, color=T["ring"]))))
    eff = rob[rob.robust_2d]
    fig.add_trace(go.Scatter(x=eff.manpower_cost, y=eff.waiting_cost_R, mode="lines", hoverinfo="skip",
                             line=dict(color=T["robust"], width=2), showlegend=False))
    fig.add_trace(go.Scatter(
        x=rob.manpower_cost, y=rob.waiting_cost_R, mode="markers", name=f"Robust front, {TH0} + {TH1} ({len(rob)})",
        customdata=[[r.plan, r.robust_label, float(r.sl0), float(r.sl1)] for _, r in rob.iterrows()],
        hovertemplate=("<b>%{customdata[0]}</b> · robust front %{customdata[1]}<br>Manpower %{x:$,.0f}/day · "
                       f"worst-case wait %{{y:$,.2f}}/day<br>Service level {TH0} %{{customdata[2]:.2%}} · "
                       f"{TH1} %{{customdata[3]:.2%}}<extra></extra>"),
        marker=dict(size=11, color=T["robust"], line=dict(width=2, color=T["ring"]))))
    if S.get("premium") is not None:
        x0, x1 = S["cheapest_nominal"].manpower_cost, S["cheapest_robust"].manpower_cost
        for x in (x0, x1):
            fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper",
                          line=dict(color=T["context"], width=1))
        fig.add_shape(type="line", x0=x0, x1=x1, y0=0.965, y1=0.965, yref="paper",
                      line=dict(color=T["ink2"], width=1.5))
        fig.add_annotation(x=(x0 + x1) / 2, y=0.965, yref="paper", yanchor="bottom", showarrow=False,
                           text=f"cost of robustness +${S['premium']:,.0f}/day (+{S['premium_pct']:.1f}%)",
                           font=dict(size=12))
    _base_layout(fig, 500, hovermode="closest", uirevision=f"fronts-{log_y}")
    fig.update_xaxes(title_text="Manpower cost ($ per day)", tickprefix="$", tickformat=",.0f", zeroline=False)
    fig.update_yaxes(title_text="Waiting cost ($ per day): nominal for the nominal front, worst case for the robust",
                     tickprefix="$", tickformat=",.2~f", type="log" if log_y else "linear",
                     rangemode="normal" if log_y else "tozero", zeroline=False)
    return fig


def _tab_robustness(df, P, T):
    S = robustness_summary(df)
    cn, cr = S["cheapest_nominal"], S["cheapest_robust"]
    tau = P["tau"]
    _kpi_row([
        _kpi("Cheapest nominal plan", _usd(cn.manpower_cost) if cn is not None else "—",
             "/day" if cn is not None else "",
             f"{cn.plan} · clears τ on {TH0} only" if cn is not None else "no plan clears τ on the nominal day"),
        _kpi("Cheapest robust plan", _usd(cr.manpower_cost) if cr is not None else "—",
             "/day" if cr is not None else "",
             f"{cr.plan} · clears τ on {TH0} and {TH1}" if cr is not None else "no plan clears τ on both days",
             "accent"),
        _kpi("Cost of robustness", (("+" if S["premium"] >= 0 else "&minus;") + _usd(abs(S["premium"])))
             if "premium" in S else "—", "/day" if "premium" in S else "",
             f"{S['premium_pct']:+.1f}% over the cheapest nominal plan" if "premium" in S else "needs both fronts"),
        _kpi("Nominal plans failing the surge", f"{S['nominal_fail_surge']}",
             f"of {S['n_nominal_front']}", f"nominal-front plans serving under {100 * tau:.0f}% on the surge day",
             "bad" if S["nominal_fail_surge"] else ""),
    ])
    if cn is not None and cr is not None:
        text = (f"Staffing for the nominal day alone, the cheapest plan is <b>{cn.plan}</b> at "
                f"{_usd(cn.manpower_cost)}/day. ")
        if not cn.feasible1:
            text += (f"On a surge day it completes only <b>{_pct(cn.sl1)}</b> of arrivals (floor {100 * tau:.0f}%) "
                     f"and its waiting cost rises from {_usd(cn.wait_cost0, 2)} to {_usd(cn.wait_cost1, 2)}/day. ")
        else:
            text += "It also clears the floor on the surge day. "
        text += (f"The cheapest plan that clears the floor on both days is <b>{cr.plan}</b> at "
                 f"{_usd(cr.manpower_cost)}/day, so robustness costs <b>{_usd(S['premium'])}/day "
                 f"({S['premium_pct']:+.1f}%)</b>.")
        _callout(text)
    elif cn is None:
        _callout(f"No plan clears τ = {100 * tau:.0f}% even on the nominal day, so there is no nominal front "
                 "to compare. Lower τ or lengthen the shift.", "bad")
    else:
        _callout(f"Nominal plans exist, but none clears τ = {100 * tau:.0f}% on the surge day: robust staffing "
                 "is infeasible at this floor.", "bad")

    c1, _ = st.columns([1, 5])
    c1.toggle("Log scale", key="logy2")
    _chart(_fig_fronts(df, S, T, st.session_state["logy2"]), key="chart_fronts")
    st.caption("Hollow grey: the nominal front (cheapest plans that clear τ on the nominal day). Red ✕: nominal "
               "plans that miss τ when the surge arrives. Blue: the robust front, judged on each plan's worse "
               "day. Vertical lines mark the two cheapest plans.")

    nom = df[df.nominal_front].sort_values(["manpower_cost", "wait_cost0"], kind="mergesort")
    if len(nom):
        _section("Nominal front under the surge stress test")
        table = pd.DataFrame({
            "Plan": nom.plan, "Manpower ($/day)": nom.manpower_cost, "Wait θ₀ ($/day)": nom.wait_cost0.round(2),
            f"SL {TH0} (%)": (100 * nom.sl0).round(2), f"SL {TH1} (%)": (100 * nom.sl1).round(2),
            "Wait θ₁ ($/day)": nom.wait_cost1.round(2),
            "Surge day": np.where(nom.feasible1, "✓ clears τ", "✕ misses τ"),
            "Robust front": nom.robust_label.replace("", "—")})
        st.dataframe(table, hide_index=True, column_config=_table_config(table), **_stretch(st.dataframe))


# ---------------------------------------------------------------------------------------------
# Tab 3 — Operational diagnostics
# ---------------------------------------------------------------------------------------------
def _fig_utilization(d0, d1, staffing, T):
    fig = go.Figure()
    roles = [f"{r} × {n}" for r, n in zip(ROLES, staffing)]
    for d, name, color in ((d0, f"Nominal day {TH0}", T["nominal"]), (d1, f"Surge day {TH1}", T["surge"])):
        util = 100 * np.asarray(d["util"])
        rho = 100 * np.asarray(d["rho"])
        fig.add_trace(go.Bar(y=roles, x=util, orientation="h", name=name, marker=dict(color=color),
                             text=[f"{u:.0f}%" for u in util], textposition="outside", cliponaxis=False,
                             customdata=np.c_[rho],
                             hovertemplate="%{y}<br>%{x:.1f}% of staffed time busy<br>"
                                           "offered load %{customdata[0]:.0f}% of capacity<extra>" + name + "</extra>"))
    fig.add_shape(type="line", x0=85, x1=85, y0=0, y1=1, yref="paper", line=dict(color=T["context"], width=1))
    _base_layout(fig, 330, barmode="group", bargap=0.34, bargroupgap=0.1)
    fig.update_xaxes(range=[0, 115], ticksuffix="%", title_text="Utilization (busy share of staffed time)")
    fig.update_yaxes(autorange="reversed", title_text=None)
    return fig


def _fig_stage_waits(d0, d1, T):
    fig = go.Figure()
    for d, name, color in ((d0, f"Nominal day {TH0}", T["nominal"]), (d1, f"Surge day {TH1}", T["surge"])):
        w = np.asarray(d["stage_wait_mean"])
        fig.add_trace(go.Bar(y=list(STAGES), x=w, orientation="h", name=name, marker=dict(color=color),
                             text=[f"{x:.1f} min" for x in w], textposition="outside", cliponaxis=False,
                             hovertemplate="%{y}: %{x:.1f} min average wait<extra>" + name + "</extra>"))
    _base_layout(fig, 330, barmode="group", bargap=0.3, bargroupgap=0.1)
    top = max(float(np.max(d0["stage_wait_mean"])), float(np.max(d1["stage_wait_mean"])), 1.0)
    fig.update_xaxes(range=[0, top * 1.25], title_text="Average wait before service starts (minutes)")
    fig.update_yaxes(autorange="reversed", title_text=None)
    return fig


def _fig_time_in_clinic(d0, d1, T):
    fig = go.Figure()
    both = np.r_[d0["in_system"], d1["in_system"]]
    if both.size == 0:
        return fig
    edges = np.linspace(0, max(float(np.percentile(both, 99.5)), 1.0), 41)
    for d, name, color in ((d0, f"Nominal day {TH0}", T["nominal"]), (d1, f"Surge day {TH1}", T["surge"])):
        arr = np.asarray(d["in_system"])
        if arr.size == 0:
            continue
        counts, _ = np.histogram(np.clip(arr, edges[0], edges[-1]), bins=edges)
        share = 100 * counts / arr.size
        fig.add_trace(go.Scatter(
            x=edges, y=np.r_[share, share[-1]], mode="lines", line=dict(color=color, width=2, shape="hv"),
            fill="tozeroy", fillcolor=_rgba(color, 0.10),
            name=f"{name} · median {np.median(arr):.0f} min · P90 {np.percentile(arr, 90):.0f} min",
            hovertemplate="%{x:.0f} min and up: %{y:.1f}% of patients<extra>" + name + "</extra>"))
    _base_layout(fig, 340, hovermode="x unified")
    fig.update_xaxes(title_text="Time in clinic, arrival to check-out (minutes; patients who finished)",
                     rangemode="tozero")
    fig.update_yaxes(title_text="Share of patients (%)", rangemode="tozero")
    return fig


def _fig_queues(d0, d1, T):
    fig = make_subplots(rows=1, cols=4, shared_yaxes=True, horizontal_spacing=0.035,
                        subplot_titles=[f"{r}s" for r in ROLES])
    for k in range(4):
        for d, name, color in ((d0, f"Nominal day {TH0}", T["nominal"]), (d1, f"Surge day {TH1}", T["surge"])):
            fig.add_trace(go.Scatter(x=np.asarray(d["grid_times"]) / 60, y=np.asarray(d["queue_grid"])[:, k],
                                     mode="lines", line=dict(color=color, width=2), name=name, legendgroup=name,
                                     showlegend=(k == 0),
                                     hovertemplate="hour %{x:.1f}: %{y:.1f} waiting<extra>" + name + "</extra>"),
                          row=1, col=k + 1)
    _base_layout(fig, 320, hovermode="x unified")
    fig.update_layout(margin=dict(l=8, r=8, t=36, b=8),
                      legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0))
    fig.update_xaxes(ticksuffix="h", dtick=2, zeroline=False)
    fig.update_yaxes(rangemode="tozero", zeroline=False)
    fig.update_yaxes(title_text="Patients waiting", row=1, col=1)
    return fig


def _levers_html(df, sel, P):
    base = df.loc[sel]
    rows = []
    for k, role in enumerate(ROLES):
        for delta in (+1, -1):
            n = int(base[ROLE_SHORT[k]]) + delta
            if n < min(STAFF_LEVELS) or n > max(STAFF_LEVELS):
                continue
            cfg = list(CONFIGS[sel])
            cfg[k] = n
            alt = df.loc[CONFIG_INDEX[tuple(cfg)]]
            d_cost = alt.manpower_cost - base.manpower_cost
            d_wait = alt.waiting_cost_R - base.waiting_cost_R
            status = _pill(bool(alt.feasible_R), "robustly feasible" if alt.feasible_R else "not robust")
            rows.append(
                f"<tr><td>{'Add' if delta > 0 else 'Remove'} one {role.lower()} "
                f"<span class='dss-muted'>→ {alt.plan}</span></td>"
                f"<td class='num'>{'+' if d_cost >= 0 else '&minus;'}{_usd(abs(d_cost))}</td>"
                f"<td class='num'>{_pct(base.sl_min)} → <b>{_pct(alt.sl_min)}</b></td>"
                f"<td class='num'>{'+' if d_wait >= 0 else '&minus;'}{_usd(abs(d_wait), 2)}</td>"
                f"<td>{status}</td></tr>")
    head = ("<tr><th>Change</th><th class='num'>Staff cost/day</th><th class='num'>Worse-day service level</th>"
            "<th class='num'>Worst-case wait/day</th><th>Result</th></tr>")
    return f"<table class='dss-table'>{head}{''.join(rows)}</table>"


def _diagnosis_text(row, d0, d1, df, sel, P):
    k = int(np.argmax(d1["util"]))
    role, util, rho = ROLES[k], float(d1["util"][k]), float(d1["rho"][k])
    s = int(np.argmax(d1["stage_wait_mean"]))
    parts = [f"On the surge day the <b>{role.lower()}s</b> are the busiest role: {util:.0%} of staffed time "
             f"busy, against an offered load of {rho:.0%} of their capacity."]
    if rho >= 1.0:
        parts.append("Demand there exceeds capacity, so that queue keeps growing until closing.")
    elif util >= 0.85:
        parts.append("That is close to saturation, so queues build quickly after any burst of arrivals.")
    parts.append(f"Patients wait longest at <b>{STAGES[s].lower()}</b> "
                 f"({float(d1['stage_wait_mean'][s]):.1f} min on average).")
    unserved = d1["generated_per_day"] - d1["served_per_day"]
    if row.sl1 < P["tau"]:
        parts.append(f"The plan completes {_pct(row.sl1)} of surge-day arrivals, below the "
                     f"{100 * P['tau']:.0f}% floor; about {unserved:.1f} patients a day are still in the clinic "
                     "at closing.")
    else:
        parts.append(f"It completes {_pct(row.sl1)} of surge-day arrivals; the ~{unserved:.1f} patients a day "
                     "still in the clinic at closing are mostly late arrivals who could not finish in time.")
    if int(row[ROLE_SHORT[k]]) < max(STAFF_LEVELS):
        cfg = list(CONFIGS[sel])
        cfg[k] += 1
        alt = df.loc[CONFIG_INDEX[tuple(cfg)]]
        parts.append(f"Adding one {role.lower()} (+{_usd(alt.manpower_cost - row.manpower_cost)}/day) moves the "
                     f"worse-day service level to {_pct(alt.sl_min)} and the worst-case waiting cost to "
                     f"{_usd(alt.waiting_cost_R, 2)}/day.")
    return " ".join(parts)


def _tab_diagnostics(df, P, T, api, sel):
    ss = st.session_state
    row = df.loc[sel]
    staffing = CONFIGS[sel]
    for r, n in zip(ROLE_SHORT, staffing):      # keep the plan editor in sync with the selection
        ss[f"dx_{r}"] = int(n)
    top_l, top_r = st.columns([1.3, 2], gap="large")
    with top_l:
        _section(f"Inspecting plan {row.plan}",
                 "Change any role to inspect another plan; the selection is shared with the explorer.")
        _html(_feasibility_pill(row))
    with top_r:
        cols = st.columns(4)
        for c, r, name in zip(cols, ROLE_SHORT, ROLES):
            c.selectbox(f"{name}s", list(STAFF_LEVELS), key=f"dx_{r}", on_change=_on_dx_change)

    k0, k1 = _keys(P)
    with st.spinner("Tracing patient flows…"):
        d0 = api["diag"](tuple(staffing), k0)
        d1 = api["diag"](tuple(staffing), k1)
    b = int(np.argmax(d1["util"]))
    tis1 = np.asarray(d1["in_system"])
    _kpi_row([
        _kpi(f"Service level, nominal {TH0}", _pct(row.sl0), "", f"floor {100 * P['tau']:.0f}%",
             "accent" if row.feasible0 else "bad"),
        _kpi(f"Service level, surge {TH1}", _pct(row.sl1), "", f"floor {100 * P['tau']:.0f}%",
             "accent" if row.feasible1 else "bad"),
        _kpi("Surge-day time in clinic", f"{np.median(tis1):.0f}" if tis1.size else "—", "min",
             f"median; 90th percentile {np.percentile(tis1, 90):.0f} min" if tis1.size else ""),
        _kpi("Surge-day bottleneck", ROLES[b], "",
             f"{float(d1['util'][b]):.0%} busy · offered load {float(d1['rho'][b]):.0%}"),
    ])
    _callout(_diagnosis_text(row, d0, d1, df, sel, P))

    c1, c2 = st.columns(2, gap="large")
    with c1:
        _section("Resource utilization", "Busy share of staffed time; the hairline marks 85%.")
        _chart(_fig_utilization(d0, d1, staffing, T), key="chart_util")
    with c2:
        _section("Where patients wait", "Average queueing time before each stage starts.")
        _chart(_fig_stage_waits(d0, d1, T), key="chart_waits")
    c3, c4 = st.columns(2, gap="large")
    with c3:
        _section("Time in clinic", "Distribution for patients who completed all five stages.")
        _chart(_fig_time_in_clinic(d0, d1, T), key="chart_tis")
    with c4:
        _section("Queues through the day", f"Patients waiting for each role, averaged over "
                 f"{P['n_reps']} replications.")
        _chart(_fig_queues(d0, d1, T), key="chart_queues")

    _section("Staffing levers", "One more or one fewer of each role, from the enumeration (no extra simulation).")
    _html(_levers_html(df, sel, P))
    with st.expander("Diagnostics data (table view)"):
        cap = pd.DataFrame({
            "Role": ROLES, "Staff": staffing,
            f"Utilization {TH0} (%)": (100 * np.asarray(d0["util"])).round(1),
            f"Utilization {TH1} (%)": (100 * np.asarray(d1["util"])).round(1),
            f"Offered load {TH0} (%)": (100 * np.asarray(d0["rho"])).round(1),
            f"Offered load {TH1} (%)": (100 * np.asarray(d1["rho"])).round(1)})
        st.dataframe(cap, hide_index=True, **_stretch(st.dataframe))
        waits = pd.DataFrame({
            "Stage": STAGES, f"Mean wait {TH0} (min)": np.round(d0["stage_wait_mean"], 2),
            f"Mean wait {TH1} (min)": np.round(d1["stage_wait_mean"], 2),
            f"Queued at closing {TH0}": np.round(d0["queued_at_close"], 2),
            f"Queued at closing {TH1}": np.round(d1["queued_at_close"], 2)})
        st.dataframe(waits, hide_index=True, **_stretch(st.dataframe))


# ---------------------------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------------------------
def main():
    _load_ui_stack()
    st.set_page_config(page_title="Clinic Staffing DSS", page_icon="🩺", layout="wide",
                       initial_sidebar_state="expanded")
    T = _tokens(_detect_dark())
    _html(_css(T))
    _init_state()
    api = _cached_api()
    ss = st.session_state

    submitted, inputs = _sidebar()
    P_form = _params(inputs)
    slot = st.empty()
    if submitted:
        errors, warnings = _validate(P_form)
        for msg in errors:
            st.error(msg)
        if not errors:
            _run(P_form, api, slot)
            ss["active"] = P_form
            ss["warnings"] = warnings
    if "active" not in ss and all(_is_cached(api, k) for k in _keys(P_form)):
        ss["active"] = P_form                   # results from an earlier session are on disk
        ss["warnings"] = []

    if "active" not in ss:
        _header()
        _landing(P_form)
        return

    P = ss["active"]
    try:
        df = api["table"](*_keys(P), P["wages"], P["tau"], P["wait_value"])
    except LookupError:
        del ss["active"]
        _header()
        st.warning("The cached simulations for these settings were evicted. Press Run to simulate them again.")
        _landing(P_form)
        return

    _header(P)
    for msg in ss.get("warnings", []):
        st.warning(msg)
    presets = strategy_presets(df)
    ss["_presets"] = presets
    tab1, tab2, tab3 = st.tabs(["Pareto frontier & strategy explorer", "Nominal vs robust",
                                "Operational diagnostics"])
    with tab1:
        _tab_frontier(df, P, T, presets)
    sel = int(ss["last_sel_id"])
    with tab2:
        _tab_robustness(df, P, T)
    with tab3:
        _tab_diagnostics(df, P, T, api, sel)


if __name__ == "__main__":
    # Under `streamlit run app.py` the script executes as __main__ inside the Streamlit runtime.
    # Worker processes import this file as "__mp_main__" (spawn) and skip this block entirely.
    if _inside_streamlit():
        main()
    else:
        sys.exit(_cli())
