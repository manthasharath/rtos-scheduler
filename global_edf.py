"""
global_edf.py
-------------
Global EDF (G-EDF).

  * All real-time tasks share ONE ready queue over all m cores.
  * At every time unit the m best jobs run (plain EDF: earliest deadline;
    criticality-first: High jobs first, then EDF).
  * A job may continue on a different core after preemption (migration).
    The engine prefers the core it last used (affinity).

Analytic indicator - GFB test (Goossens, Funk & Baruah, 2003), density form
for constrained deadlines:
        sum(density) <= m - (m - 1) * max(density)
A SUFFICIENT test: passing guarantees no deadline miss under G-EDF; failing
does not prove a miss. The same test on the High tasks alone is the
"High tasks guaranteed" check of criticality-first mode.
"""

from __future__ import annotations

from typing import List, Tuple

from scheduler import SimulationResult, edf_schedule, serve_aperiodic
from task_model import Task, ap_tasks, rt_tasks

EPS = 1e-9


def gfb_test(tasks: List[Task], m: int) -> Tuple[bool, float, float]:
    """Returns (passed, total density, bound)."""
    tasks = rt_tasks(tasks)
    if not tasks:
        return True, 0.0, float(m)
    total = sum(t.density for t in tasks)
    dmax = max(t.density for t in tasks)
    bound = m - (m - 1) * dmax
    return total <= bound + EPS, total, bound


def high_group_check(tasks: List[Task], cores: int, label: str) -> dict:
    """High-only test for one core group (density test for 1 core, GFB otherwise)."""
    high = [t for t in rt_tasks(tasks) if t.criticality == "High"]
    if not high:
        return {"group": label, "passed": True, "detail": "no High tasks"}
    ok, total, bound = gfb_test(high, cores)
    names = ", ".join(t.name for t in high)
    if cores == 1:
        detail = f"High density {total:.3f} {'<=' if ok else '>'} 1  ({names})"
    else:
        detail = f"High density {total:.3f} {'<=' if ok else '>'} GFB bound {bound:.3f}  ({names})"
    return {"group": label, "passed": ok, "detail": detail}


def summarize_high(groups: List[dict]) -> dict:
    return {"passed": all(g["passed"] for g in groups), "groups": groups}


def run_global_edf(tasks: List[Task], num_cores: int, horizon: int, miss_policy: str = "continue",
                   crit_mode: bool = False, gap_pct: int = 50, seed: int = 1) -> SimulationResult:
    cores = list(range(num_cores))
    rt = rt_tasks(tasks)
    jobs, timeline = edf_schedule(rt, cores, horizon, miss_policy, crit_mode, gap_pct, seed)
    ap_jobs = serve_aperiodic(ap_tasks(tasks), timeline, horizon)
    passed, total, bound = gfb_test(rt, num_cores)
    return SimulationResult(
        algorithm="Global EDF", tasks=list(tasks), num_cores=num_cores, horizon=horizon,
        jobs=jobs, timeline=timeline, clusters=[cores],
        allocation={t.task_id: 0 for t in tasks}, allocation_failures=[],
        analytic_test={"name": "GFB test (sufficient only)", "passed": passed,
                       "detail": f"Total density {total:.3f} {'<=' if passed else '>'} bound {bound:.3f} "
                                 f"= m - (m-1)*max density."},
        high_check=summarize_high([high_group_check(rt, num_cores, f"All {num_cores} cores")]),
        miss_policy=miss_policy, cluster_size=num_cores, crit_mode=crit_mode,
        aperiodic_jobs=ap_jobs, gap_pct=gap_pct, seed=seed,
    )
