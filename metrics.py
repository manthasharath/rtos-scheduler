"""
metrics.py
----------
Performance metrics and timeline segments computed from a SimulationResult.

Definitions (all over the finite simulated horizon [0, H)):
  Released jobs      : real-time jobs released before H.
  Completed jobs     : real-time jobs that received all C units before H.
  Evaluated jobs     : completed jobs + unfinished jobs whose deadline <= H
                       (unfinished jobs with deadline > H have no verdict yet).
  Deadline misses    : evaluated jobs that did not finish by their deadline.
  Miss ratio         : misses / evaluated jobs.
  Response time      : finish - release (completed jobs).
  Core utilization   : busy time units on that core / H (includes aperiodic work).
  Load imbalance     : max core utilization - min core utilization.
  Overhead events    : preemptions, migrations and context switches are
                       COUNTED; their time cost is modelled as zero here
                       (the Raspberry Pi version measures the real cost).
  Aperiodic jobs     : background service; response time = finish - arrival.
  Schedulable in H   : no real-time deadline missed inside the horizon
                       (a simulation observation, not a proof).
  High tasks guaranteed : the High-only test passed on every core/cluster;
                       a guarantee only in criticality-first mode.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

from scheduler import SimulationResult
from task_model import CRITICALITY_LEVELS


def core_busy_units(result: SimulationResult) -> Dict[int, int]:
    return {c: sum(1 for x in row if x is not None) for c, row in result.timeline.items()}


def context_switches(result: SimulationResult) -> int:
    """Times a core starts running a job different from the last job it ran."""
    count = 0
    for row in result.timeline.values():
        last = None
        for key in row:
            if key is not None and key != last:
                if last is not None:
                    count += 1
                last = key
    return count


def _ratio(a: int, b: int) -> float:
    return round(a / b, 4) if b else 0.0


def _crit_of(result: SimulationResult) -> Dict[int, str]:
    return {t.task_id: t.criticality for t in result.tasks}


def high_guaranteed_text(result: SimulationResult) -> str:
    ok = result.high_check.get("passed", False)
    if result.crit_mode:
        return "YES" if ok else "NO"
    return "not in plain EDF"


def compute_metrics(result: SimulationResult) -> Dict:
    H, m = result.horizon, result.num_cores
    jobs = result.jobs
    crit_of = _crit_of(result)
    busy = core_busy_units(result)
    per_core = {c: busy[c] / H for c in sorted(busy)}
    utils = np.array(list(per_core.values()))

    completed = [j for j in jobs if j.finish is not None]
    evaluated = [j for j in jobs if j.finish is not None or j.abs_deadline <= H]
    missed = [j for j in jobs if j.missed]
    rts = [j.response_time for j in completed]
    ap = result.aperiodic_jobs
    ap_done = [j for j in ap if j.finish is not None]
    ap_rts = [j.response_time for j in ap_done]
    rt_keys = {j.key for j in jobs}
    rt_units = sum(1 for row in result.timeline.values() for k in row if k is not None and k in rt_keys)

    from task_model import total_density, total_utilization
    out = {
        "Algorithm": result.algorithm,
        "EDF mode": result.mode_name,
        "Cores": m,
        "Cluster size": result.cluster_size,
        "Horizon": H,
        "Task utilization (sum C/T)": round(total_utilization(result.tasks), 4),
        "Normalized utilization (U/m)": round(total_utilization(result.tasks) / m, 4),
        "Total density": round(total_density(result.tasks), 4),
        "CPU utilization (observed)": round(sum(busy.values()) / (m * H), 4),
        "Real-time CPU utilization": round(rt_units / (m * H), 4),
        "Per-core utilization": {f"Core {c}": round(u, 4) for c, u in per_core.items()},
        "Released jobs": len(jobs),
        "Completed jobs": len(completed),
        "Evaluated jobs": len(evaluated),
        "Deadline misses": len(missed),
        "Aborted jobs": sum(1 for j in jobs if j.aborted),
        "Deadline miss ratio": _ratio(len(missed), len(evaluated)),
        "Average response time": round(float(np.mean(rts)), 3) if rts else None,
        "Maximum response time": int(max(rts)) if rts else None,
        "Preemptions": sum(j.preemptions for j in jobs),
        "Migrations": sum(j.migrations for j in jobs),
        "Context switches": context_switches(result),
        "Load imbalance (max-min)": round(float(utils.max() - utils.min()), 4) if len(utils) else 0.0,
        "Load std. deviation": round(float(utils.std()), 4) if len(utils) else 0.0,
        "Allocation failures": len(result.allocation_failures),
        "Analytic test": result.analytic_test.get("name", ""),
        "Analytic test passed": bool(result.analytic_test.get("passed", False)),
        "Schedulable in horizon": len(missed) == 0,
        "High tasks guaranteed": high_guaranteed_text(result),
        "Aperiodic jobs": len(ap),
        "Aperiodic completed": len(ap_done),
        "Aperiodic avg response": round(float(np.mean(ap_rts)), 3) if ap_rts else None,
        "Aperiodic max response": int(max(ap_rts)) if ap_rts else None,
    }
    for lvl in ["High", "Medium", "Low"]:
        ev = [j for j in evaluated if crit_of[j.task_id] == lvl]
        mi = [j for j in missed if crit_of[j.task_id] == lvl]
        out[f"{lvl} misses"] = len(mi)
        out[f"{lvl} miss ratio"] = _ratio(len(mi), len(ev))
    return out


def metrics_row(result: SimulationResult) -> Dict:
    """Flat metrics (per-core dict expanded) for tables and CSV export."""
    m = compute_metrics(result)
    per_core = m.pop("Per-core utilization")
    m.update({f"Util {k}": v for k, v in per_core.items()})
    return m


def comparison_table(results: Dict[str, SimulationResult]) -> pd.DataFrame:
    return pd.DataFrame([metrics_row(r) for r in results.values()])


def criticality_metrics(result: SimulationResult) -> pd.DataFrame:
    """Results per criticality level (High, Medium, Low) for the real-time jobs."""
    H = result.horizon
    crit_of = _crit_of(result)
    rows = []
    for lvl in ["High", "Medium", "Low"]:
        js = [j for j in result.jobs if crit_of[j.task_id] == lvl]
        done = [j for j in js if j.finish is not None]
        ev = [j for j in js if j.finish is not None or j.abs_deadline <= H]
        miss = [j for j in js if j.missed]
        rts = [j.response_time for j in done]
        rows.append({
            "Algorithm": result.algorithm,
            "EDF mode": result.mode_name,
            "Criticality": lvl,
            "Tasks": sum(1 for t in result.tasks if t.is_realtime and t.criticality == lvl),
            "Released jobs": len(js),
            "Completed jobs": len(done),
            "Deadline misses": len(miss),
            "Miss ratio": _ratio(len(miss), len(ev)),
            "Avg response": round(float(np.mean(rts)), 3) if rts else None,
            "Max response": max(rts) if rts else None,
        })
    df = pd.DataFrame(rows)
    df["Max response"] = df["Max response"].astype("Int64")
    return df


def aperiodic_table(result: SimulationResult) -> pd.DataFrame:
    rows = []
    for j in result.aperiodic_jobs:
        rows.append({
            "Algorithm": result.algorithm,
            "Job": j.label,
            "Arrival": j.release,
            "C": j.wcet,
            "Start": j.start,
            "Finish": j.finish,
            "Response time": j.response_time,
            "Executed": j.executed,
            "Cores used": ", ".join(map(str, sorted(j.cores_used))),
            "Status": "completed" if j.finish is not None else f"waiting ({j.executed}/{j.wcet} done)",
        })
    return pd.DataFrame(rows, columns=["Algorithm", "Job", "Arrival", "C", "Start", "Finish", "Response time",
                                       "Executed", "Cores used", "Status"])


def high_check_table(result: SimulationResult) -> pd.DataFrame:
    rows = [{"Algorithm": result.algorithm, "Core group": g["group"], "High tasks fit alone": g["passed"],
             "Detail": g["detail"]} for g in result.high_check.get("groups", [])]
    return pd.DataFrame(rows)


def task_metrics(result: SimulationResult) -> pd.DataFrame:
    """Per-task statistics (real-time tasks)."""
    by_task = defaultdict(list)
    for j in result.jobs:
        by_task[j.task_id].append(j)
    rows = []
    for task in result.tasks:
        if not task.is_realtime:
            continue
        js = by_task.get(task.task_id, [])
        done = [j for j in js if j.finish is not None]
        rts = [j.response_time for j in done]
        cores = sorted(set().union(*[j.cores_used for j in js])) if js else []
        group = result.allocation.get(task.task_id)
        if result.algorithm == "Partitioned EDF":
            placed = f"Core {group}"
        elif result.algorithm == "Clustered EDF":
            placed = f"Cluster {group + 1} (cores {result.clusters[group][0]}-{result.clusters[group][-1]})"
        else:
            placed = "Any core"
        rows.append({
            "Task": task.name, "Type": task.kind, "C": task.wcet, "T": task.period, "D": task.deadline,
            "Priority": task.priority, "Criticality": task.criticality,
            "Utilization": round(task.utilization, 3),
            "Assigned to": placed,
            "Allocation failed": task.task_id in result.allocation_failures,
            "Cores used": ", ".join(map(str, cores)),
            "Released": len(js), "Completed": len(done),
            "Missed": sum(1 for j in js if j.missed),
            "Avg response": round(float(np.mean(rts)), 2) if rts else None,
            "Max response": max(rts) if rts else None,
            "Preemptions": sum(j.preemptions for j in js),
            "Migrations": sum(j.migrations for j in js),
        })
    return pd.DataFrame(rows)


def job_table(result: SimulationResult) -> pd.DataFrame:
    crit_of = _crit_of(result)
    rows = []
    for j in result.jobs:
        rows.append({
            "Algorithm": result.algorithm, "EDF mode": result.mode_name,
            "Job": j.label, "Task": j.task_name, "Type": j.kind, "Criticality": crit_of[j.task_id],
            "Release": j.release, "Absolute deadline": j.abs_deadline, "C": j.wcet,
            "Executed": j.executed, "Start": j.start, "Finish": j.finish,
            "Response time": j.response_time, "Missed deadline": j.missed, "Aborted": j.aborted,
            "Preemptions": j.preemptions, "Migrations": j.migrations,
            "Cores used": ", ".join(map(str, sorted(j.cores_used))),
        })
    return pd.DataFrame(rows)


def timeline_segments(result: SimulationResult, include_idle: bool = True) -> pd.DataFrame:
    """
    Compress the per-tick timeline into continuous segments. Each execution
    segment gets an 'End event': Completed / Preempted / Migrated / Aborted /
    Running at horizon end.
    """
    H = result.horizon
    jmap = result.job_map()
    crit_of = _crit_of(result)
    raw = []
    for core in sorted(result.timeline):
        row = result.timeline[core]
        s = 0
        for t in range(1, H + 1):
            if t == H or row[t] != row[s]:
                raw.append((core, s, t, row[s]))
                s = t
    per_job = defaultdict(list)
    for core, s, e, key in raw:
        if key is not None:
            per_job[key].append((s, e, core))
    for key in per_job:
        per_job[key].sort()

    cluster_of = {c: i for i, cl in enumerate(result.clusters) for c in cl}
    rows = []
    for core, s, e, key in raw:
        base = {"Algorithm": result.algorithm, "Core": core, "Cluster": cluster_of.get(core, 0) + 1,
                "Start": s, "End": e, "Duration": e - s}
        if key is None:
            if include_idle:
                rows.append({**base, "Type": "Idle", "Task": "Idle", "Job": "", "Kind": "", "Criticality": "",
                             "Release": None, "Abs deadline": None, "Migrated in": False, "End event": "",
                             "Deadline missed": False})
            continue
        job = jmap[key]
        segs = per_job[key]
        idx = segs.index((s, e, core))
        migrated_in = idx > 0 and segs[idx - 1][2] != core
        executed_by_end = sum(se - ss for ss, se, _ in segs[: idx + 1])
        if executed_by_end >= job.wcet:
            event = "Completed"
        elif idx + 1 < len(segs):
            nxt = segs[idx + 1]
            event = "Migrated" if (nxt[0] == e and nxt[2] != core) else "Preempted"
        elif e == H:
            event = "Running at horizon end"
        elif job.aborted:
            event = "Aborted at deadline"
        else:
            event = "Preempted"
        rows.append({**base, "Type": "Aperiodic" if job.kind == "A" else "Execution", "Task": job.task_name,
                     "Job": job.label, "Kind": job.kind,
                     "Criticality": "" if job.kind == "A" else crit_of[job.task_id],
                     "Release": job.release, "Abs deadline": None if job.kind == "A" else job.abs_deadline,
                     "Migrated in": migrated_in, "End event": event, "Deadline missed": job.missed})
    return pd.DataFrame(rows)


def cumulative_misses(result: SimulationResult) -> pd.DataFrame:
    """Number of deadline misses detected up to each time instant (for line charts)."""
    H = result.horizon
    counts = [0] * (H + 1)
    for j in result.jobs:
        if j.missed and j.abs_deadline <= H:
            counts[j.abs_deadline] += 1
    return pd.DataFrame({"Time": range(H + 1), "Cumulative misses": np.cumsum(counts),
                         "Algorithm": result.algorithm})


def mode_comparison(tasks, num_cores, horizon, cluster_size, miss_policy, gap_pct, seed, algorithms) -> pd.DataFrame:
    """Run plain and criticality-first EDF on the same task set; misses per criticality."""
    from algorithms import run_algorithm
    rows = []
    for alg in algorithms:
        for crit in (False, True):
            r = run_algorithm(alg, tasks, num_cores, horizon, cluster_size, miss_policy, crit, gap_pct, seed)
            m = compute_metrics(r)
            rows.append({"Algorithm": alg, "EDF mode": r.mode_name,
                         "High misses": m["High misses"], "Medium misses": m["Medium misses"],
                         "Low misses": m["Low misses"], "Total misses": m["Deadline misses"],
                         "High guaranteed": m["High tasks guaranteed"],
                         "Avg response": m["Average response time"],
                         "Aperiodic avg response": m["Aperiodic avg response"]})
    return pd.DataFrame(rows)
