"""
task_generator.py
-----------------
Random periodic task-set generation for batch experiments.

Utilizations: UUniFast-Discard.
  UUniFast (Bini & Buttazzo, 2005) draws n utilizations that sum exactly to
  a target U and are uniformly distributed over that set. On multiprocessors
  U may exceed 1, so a single task could get u > 1; such sets are discarded
  and redrawn ("UUniFast-Discard", Davis & Burns, 2009/2011).

Periods: drawn uniformly from a fixed set (default 10, 20, 25, 40, 50, 100)
  so the hyperperiod stays small (LCM = 200) and every set can be simulated
  for one full hyperperiod.

Execution times: C = round(u * T), clamped to [1, T]. Integer rounding means
  the ACTUAL utilization differs slightly from the target; the actual value
  is always recorded and used in the results.

Deadlines: implicit (D = T) by default; "constrained" draws D uniformly from
  the integers in [C, T].
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from task_model import CRITICALITY_LEVELS, Task

DEFAULT_PERIODS = (10, 20, 25, 40, 50, 100)


def uunifast(n: int, total_u: float, rng: np.random.Generator) -> List[float]:
    utils = []
    sum_u = total_u
    for i in range(1, n):
        next_sum = sum_u * rng.random() ** (1.0 / (n - i))
        utils.append(sum_u - next_sum)
        sum_u = next_sum
    utils.append(sum_u)
    return utils


def uunifast_discard(n: int, total_u: float, rng: np.random.Generator, max_tries: int = 10000) -> List[float]:
    if n < 1:
        raise ValueError("n must be >= 1")
    if total_u <= 0 or total_u > n:
        raise ValueError(f"target utilization must be in (0, n] = (0, {n}]")
    for _ in range(max_tries):
        utils = uunifast(n, total_u, rng)
        if all(0 < u <= 1 for u in utils):
            return utils
    raise RuntimeError(
        f"Could not generate {n} task utilizations summing to {total_u} with every u <= 1. "
        "Use more tasks or a lower utilization."
    )


def generate_taskset(
    n: int,
    total_u: float,
    rng: np.random.Generator,
    periods: Sequence[int] = DEFAULT_PERIODS,
    deadline_type: str = "implicit",
    sporadic_frac: float = 0.0,
    n_aperiodic: int = 0,
    aperiodic_max_c: int = 5,
) -> List[Task]:
    """
    n real-time tasks with total utilization total_u (UUniFast-Discard), random
    criticality, a share `sporadic_frac` of them sporadic (type S), plus
    `n_aperiodic` aperiodic jobs (type A) with C in [1, aperiodic_max_c] and a
    random arrival inside the hyperperiod. Priorities are automatic (shorter
    period first).
    """
    utils = uunifast_discard(n, total_u, rng)
    raw = []
    for i, u in enumerate(utils):
        T = int(rng.choice(periods))
        C = int(min(T, max(1, round(u * T))))
        D = int(rng.integers(C, T + 1)) if deadline_type == "constrained" else T
        crit = CRITICALITY_LEVELS[int(rng.integers(0, len(CRITICALITY_LEVELS)))]
        raw.append((C, T, D, crit))
    kinds = ["P"] * n
    if sporadic_frac > 0:
        k = int(round(sporadic_frac * n))
        for idx in rng.choice(n, size=k, replace=False):
            kinds[int(idx)] = "S"
    order = sorted(range(n), key=lambda i: (raw[i][1], i))
    prio = {i: p for p, i in enumerate(order, start=1)}
    tasks = [Task(task_id=i, name=f"T{i + 1}", wcet=C, period=T, deadline=D, priority=prio[i],
                  criticality=crit, kind=kinds[i]) for i, (C, T, D, crit) in enumerate(raw)]
    if n_aperiodic > 0:
        from task_model import hyperperiod
        H = hyperperiod(tasks)
        for j in range(n_aperiodic):
            tasks.append(Task(task_id=n + j, name=f"A{j + 1}", wcet=int(rng.integers(1, aperiodic_max_c + 1)),
                              period=0, deadline=0, priority=0, criticality="Low", kind="A",
                              arrival=int(rng.integers(0, H))))
    return tasks
