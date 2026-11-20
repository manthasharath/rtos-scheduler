"""
partitioned_edf.py
------------------
Partitioned EDF (P-EDF).

Allocation - First Fit Decreasing (FFD):
  1. Order the real-time tasks:
       plain EDF            : by density C/min(D,T), largest first
       criticality-first EDF: DCDU (Decreasing Criticality, Decreasing
                              Utilization): High tasks first, then Medium,
                              then Low, largest density first inside each
                              level (ties: table order).
  2. Put each task on the FIRST core whose total density stays <= 1.
     (On one core EDF meets all deadlines when utilization <= 1 for D = T,
     Liu & Layland 1973; the density test is sufficient for D < T.)
  3. If no core fits, it is an ALLOCATION FAILURE: the task goes to the
     least-loaded core (so the simulation shows the effect) and is reported.

Run time: each core runs its own EDF. Migration: never.
Aperiodic jobs: background service on any idle core.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from global_edf import high_group_check, summarize_high
from scheduler import SimulationResult, edf_schedule, merge_group_runs, serve_aperiodic
from task_model import Task, ap_tasks, rt_tasks

EPS = 1e-9


def allocation_order(tasks: List[Task], dcdu: bool) -> List[Task]:
    if dcdu:
        return sorted(tasks, key=lambda t: (-t.crit_rank, -t.density, t.task_id))
    return sorted(tasks, key=lambda t: (-t.density, t.task_id))


def ffd_partition(tasks: List[Task], num_cores: int, dcdu: bool = False) -> Tuple[Dict[int, int], List[int], List[float]]:
    """Returns (allocation task_id -> core, failed task ids, density load per core)."""
    load = [0.0] * num_cores
    allocation: Dict[int, int] = {}
    failures: List[int] = []
    for task in allocation_order(rt_tasks(tasks), dcdu):
        placed = False
        for core in range(num_cores):
            if load[core] + task.density <= 1.0 + EPS:
                allocation[task.task_id] = core
                load[core] += task.density
                placed = True
                break
        if not placed:
            failures.append(task.task_id)
            core = min(range(num_cores), key=lambda c: (load[c], c))
            allocation[task.task_id] = core
            load[core] += task.density
    return allocation, failures, load


def run_partitioned_edf(tasks: List[Task], num_cores: int, horizon: int, miss_policy: str = "continue",
                        crit_mode: bool = False, gap_pct: int = 50, seed: int = 1) -> SimulationResult:
    rt = rt_tasks(tasks)
    allocation, failures, load = ffd_partition(rt, num_cores, dcdu=crit_mode)
    runs, checks = [], []
    for core in range(num_cores):
        core_tasks = [t for t in rt if allocation[t.task_id] == core]
        runs.append(edf_schedule(core_tasks, [core], horizon, miss_policy, crit_mode, gap_pct, seed))
        checks.append(high_group_check(core_tasks, 1, f"Core {core}"))
    jobs, timeline = merge_group_runs(runs, num_cores, horizon)
    ap_jobs = serve_aperiodic(ap_tasks(tasks), timeline, horizon)

    names = {t.task_id: t.name for t in rt}
    if failures:
        detail = ("FFD could not fit " + ", ".join(names[i] for i in failures)
                  + " on any core (density would exceed 1); placed on the least-loaded core, now overloaded.")
    else:
        detail = "All tasks fit by FFD; every core has density <= 1 (" + \
                 ", ".join(f"core {c}: {l:.2f}" for c, l in enumerate(load)) + ")."
    return SimulationResult(
        algorithm="Partitioned EDF", tasks=list(tasks), num_cores=num_cores, horizon=horizon,
        jobs=jobs, timeline=timeline, clusters=[[c] for c in range(num_cores)],
        allocation=allocation, allocation_failures=failures,
        analytic_test={"name": "FFD + uniprocessor EDF density test (sufficient; exact for D = T)",
                       "passed": not failures, "detail": detail},
        high_check=summarize_high(checks),
        miss_policy=miss_policy, cluster_size=1, crit_mode=crit_mode,
        aperiodic_jobs=ap_jobs, gap_pct=gap_pct, seed=seed,
    )
