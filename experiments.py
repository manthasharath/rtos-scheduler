"""
experiments.py
--------------
Batch experiments: generate many random task sets and run the chosen
algorithms (and EDF modes) on EXACTLY the same task sets (fair comparison).

For every normalized utilization level U/m in `util_levels`, `sets_per_level`
task sets are generated (e.g. 7 levels x 50 sets = 350 experiments). Each set
is simulated for its hyperperiod (or a fixed horizon). Nothing is hard-coded:
every number in the output comes from a simulation run.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from algorithms import ALGORITHMS, run_algorithm
from metrics import compute_metrics
from task_generator import DEFAULT_PERIODS, generate_taskset
from task_model import hyperperiod, total_utilization

MODE_NAMES = {False: "Plain EDF", True: "Criticality-first EDF"}


def run_batch(
    num_tasks: int,
    num_cores: int,
    util_levels: Sequence[float],
    sets_per_level: int,
    seed: int = 42,
    cluster_size: int = 2,
    miss_policy: str = "continue",
    deadline_type: str = "implicit",
    periods: Sequence[int] = DEFAULT_PERIODS,
    horizon: Optional[int] = None,
    progress: Optional[Callable[[float], None]] = None,
    modes: Sequence[bool] = (False,),
    algorithms: Sequence[str] = tuple(ALGORITHMS),
    sporadic_frac: float = 0.0,
    n_aperiodic: int = 0,
    gap_pct: int = 50,
) -> Dict[str, pd.DataFrame]:
    """modes: False = plain EDF, True = criticality-first EDF."""
    rng = np.random.default_rng(seed)
    result_rows: List[dict] = []
    task_rows: List[dict] = []
    total = len(util_levels) * sets_per_level
    exp_id = 0
    for level in util_levels:
        target_u = level * num_cores
        for _ in range(sets_per_level):
            exp_id += 1
            tasks = generate_taskset(num_tasks, target_u, rng, periods, deadline_type,
                                     sporadic_frac=sporadic_frac, n_aperiodic=n_aperiodic)
            H = horizon or hyperperiod(tasks)
            actual_u = total_utilization(tasks)
            for t in tasks:
                task_rows.append({"Experiment": exp_id, "Target U/m": level, "Task": t.name, "Type": t.kind,
                                  "C": t.wcet, "T": t.period if t.is_realtime else None,
                                  "D": t.deadline if t.is_realtime else None,
                                  "Priority": t.priority if t.is_realtime else None,
                                  "Criticality": t.criticality if t.is_realtime else None,
                                  "Arrival": None if t.is_realtime else t.arrival})
            for crit in modes:
                for alg in algorithms:
                    res = run_algorithm(alg, tasks, num_cores, H, cluster_size, miss_policy, crit, gap_pct, seed)
                    m = compute_metrics(res)
                    result_rows.append({
                        "Experiment": exp_id, "Algorithm": alg, "EDF mode": res.mode_name,
                        "Target U/m": level, "Actual U": round(actual_u, 4),
                        "Actual U/m": round(actual_u / num_cores, 4),
                        "Tasks": num_tasks, "Cores": num_cores, "Cluster size": res.cluster_size, "Horizon": H,
                        **{k: m[k] for k in [
                            "Released jobs", "Completed jobs", "Deadline misses", "Deadline miss ratio",
                            "CPU utilization (observed)", "Average response time", "Maximum response time",
                            "Preemptions", "Migrations", "Load std. deviation", "Allocation failures",
                            "Analytic test passed", "Schedulable in horizon", "High misses", "Medium misses",
                            "Low misses", "High miss ratio", "Medium miss ratio", "Low miss ratio",
                            "Aperiodic jobs", "Aperiodic completed", "Aperiodic avg response"]},
                        "High tasks guaranteed": res.high_check.get("passed", False),
                    })
            if progress:
                progress(exp_id / total)
    results = pd.DataFrame(result_rows)
    return {
        "results": results,
        "tasksets": pd.DataFrame(task_rows),
        "summary": summarize(results),
        "by_level": summarize_by_level(results),
    }


def _agg(g: pd.DataFrame) -> pd.Series:
    w = g["Actual U"]
    return pd.Series({
        "Experiments": g["Experiment"].nunique(),
        "Schedulability ratio (simulation)": round(g["Schedulable in horizon"].mean(), 4),
        "Weighted schedulability": round(float((g["Schedulable in horizon"] * w).sum() / w.sum()), 4) if w.sum() else 0.0,
        "Analytic test pass ratio": round(g["Analytic test passed"].mean(), 4),
        "High-only test pass ratio": round(g["High tasks guaranteed"].mean(), 4),
        "Avg deadline miss ratio": round(g["Deadline miss ratio"].mean(), 4),
        "Avg CPU utilization": round(g["CPU utilization (observed)"].mean(), 4),
        "Avg response time": round(g["Average response time"].mean(), 3),
        "Max response time": g["Maximum response time"].max(),
        "Avg preemptions": round(g["Preemptions"].mean(), 2),
        "Avg migrations": round(g["Migrations"].mean(), 2),
        "Avg load std. dev.": round(g["Load std. deviation"].mean(), 4),
        "Avg High miss ratio": round(g["High miss ratio"].mean(), 4),
        "Avg Medium miss ratio": round(g["Medium miss ratio"].mean(), 4),
        "Avg Low miss ratio": round(g["Low miss ratio"].mean(), 4),
        "Avg aperiodic response": round(g["Aperiodic avg response"].mean(), 3)
        if g["Aperiodic avg response"].notna().any() else None,
    })


def _ints(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("Experiments", "Max response time"):
        if col in df:
            df[col] = df[col].astype("Int64")
    return df


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode in MODE_NAMES.values():
        for alg in ALGORITHMS:
            g = results[(results["Algorithm"] == alg) & (results["EDF mode"] == mode)]
            if len(g):
                rows.append({"Algorithm": alg, "EDF mode": mode, **_agg(g).to_dict()})
    return _ints(pd.DataFrame(rows))


def summarize_by_level(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (level, mode, alg), g in results.groupby(["Target U/m", "EDF mode", "Algorithm"], sort=True):
        series = alg if mode == "Plain EDF" else f"{alg} (criticality-first)"
        rows.append({"Target U/m": level, "Algorithm": alg, "EDF mode": mode, "Series": series,
                     **_agg(g).to_dict()})
    return _ints(pd.DataFrame(rows))
