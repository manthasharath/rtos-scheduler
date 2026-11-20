"""
clustered_edf.py
----------------
Clustered EDF (C-EDF).

Clusters: the m cores are split into consecutive groups of k cores
(cluster 1 = cores 0..k-1, ...). If m is not a multiple of k the last
cluster is smaller. k = 1 behaves like Partitioned EDF, k = m like Global EDF.

Allocation (static, task -> cluster) - First Fit Decreasing:
  order as in partitioned_edf (density, or DCDU in criticality-first mode);
  each task goes to the FIRST cluster that still passes the GFB test
      sum(density) <= c - (c - 1) * max(density)      (c = cores in cluster)
  If none passes: ALLOCATION FAILURE, placed in the cluster with the most
  spare capacity and reported.

Run time: global EDF inside each cluster; migration only inside the cluster.
Aperiodic jobs: background service on any idle core.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from global_edf import gfb_test, high_group_check, summarize_high
from partitioned_edf import allocation_order
from scheduler import SimulationResult, edf_schedule, merge_group_runs, serve_aperiodic
from task_model import Task, ap_tasks, rt_tasks


def make_clusters(num_cores: int, cluster_size: int) -> List[List[int]]:
    if cluster_size < 1 or cluster_size > num_cores:
        raise ValueError(f"cluster size must be between 1 and {num_cores}")
    return [list(range(s, min(s + cluster_size, num_cores))) for s in range(0, num_cores, cluster_size)]


def allocate_to_clusters(tasks: List[Task], clusters: List[List[int]], dcdu: bool = False
                         ) -> Tuple[Dict[int, int], List[int]]:
    """Returns (allocation task_id -> cluster index, failed task ids)."""
    members: List[List[Task]] = [[] for _ in clusters]
    allocation: Dict[int, int] = {}
    failures: List[int] = []
    for task in allocation_order(rt_tasks(tasks), dcdu):
        placed = False
        for ci, cores in enumerate(clusters):
            ok, _, _ = gfb_test(members[ci] + [task], len(cores))
            if ok:
                members[ci].append(task)
                allocation[task.task_id] = ci
                placed = True
                break
        if not placed:
            failures.append(task.task_id)
            spare = [len(cores) - sum(t.density for t in members[ci]) for ci, cores in enumerate(clusters)]
            ci = max(range(len(clusters)), key=lambda i: (spare[i], -i))
            members[ci].append(task)
            allocation[task.task_id] = ci
    return allocation, failures


def run_clustered_edf(tasks: List[Task], num_cores: int, horizon: int, cluster_size: int = 2,
                      miss_policy: str = "continue", crit_mode: bool = False, gap_pct: int = 50,
                      seed: int = 1) -> SimulationResult:
    rt = rt_tasks(tasks)
    clusters = make_clusters(num_cores, cluster_size)
    allocation, failures = allocate_to_clusters(rt, clusters, dcdu=crit_mode)
    runs, checks = [], []
    for ci, cores in enumerate(clusters):
        cl_tasks = [t for t in rt if allocation[t.task_id] == ci]
        runs.append(edf_schedule(cl_tasks, cores, horizon, miss_policy, crit_mode, gap_pct, seed))
        checks.append(high_group_check(cl_tasks, len(cores), f"Cluster {ci + 1} (cores {cores[0]}-{cores[-1]})"))
    jobs, timeline = merge_group_runs(runs, num_cores, horizon)
    ap_jobs = serve_aperiodic(ap_tasks(tasks), timeline, horizon)

    names = {t.task_id: t.name for t in rt}
    if failures:
        detail = ("No cluster passed the GFB test for " + ", ".join(names[i] for i in failures)
                  + "; placed in the cluster with most spare capacity (overloaded).")
    else:
        detail = (f"All tasks allocated to {len(clusters)} cluster(s) of size {cluster_size}; "
                  "every cluster passes the GFB density test.")
    return SimulationResult(
        algorithm="Clustered EDF", tasks=list(tasks), num_cores=num_cores, horizon=horizon,
        jobs=jobs, timeline=timeline, clusters=clusters,
        allocation=allocation, allocation_failures=failures,
        analytic_test={"name": "FFD to clusters + per-cluster GFB test (sufficient only)",
                       "passed": not failures, "detail": detail},
        high_check=summarize_high(checks),
        miss_policy=miss_policy, cluster_size=cluster_size, crit_mode=crit_mode,
        aperiodic_jobs=ap_jobs, gap_pct=gap_pct, seed=seed,
    )
