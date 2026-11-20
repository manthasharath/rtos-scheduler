"""
Algorithm tests: allocation policies, migration restrictions, known
textbook cases, and schedule invariants on hundreds of random task sets.
"""

import numpy as np
import pytest

from algorithms import ALGORITHMS, run_algorithm
from clustered_edf import allocate_to_clusters, make_clusters
from global_edf import gfb_test
from partitioned_edf import ffd_partition
from task_generator import generate_taskset, uunifast, uunifast_discard
from task_model import Task, hyperperiod


def dens_tasks(densities, period=10):
    return [Task(i, f"T{i}", int(d * period), period, period) for i, d in enumerate(densities)]


# ---------------- allocation ----------------
def test_ffd_allocation_known_result():
    # densities 0.4, 0.6, 0.3, 0.5 -> sorted 0.6,0.5,0.4,0.3
    # 0.6->core0, 0.5->core1, 0.4->core0 (=1.0), 0.3->core1 (=0.8)
    tasks = dens_tasks([0.4, 0.6, 0.3, 0.5])
    alloc, fails, load = ffd_partition(tasks, 2)
    assert alloc == {1: 0, 3: 1, 0: 0, 2: 1}
    assert fails == [] and load == pytest.approx([1.0, 0.8])


def test_ffd_reports_failure():
    tasks = dens_tasks([0.6, 0.6, 0.6])
    alloc, fails, _ = ffd_partition(tasks, 2)
    assert fails == [2]
    res = run_algorithm("Partitioned EDF", tasks, 2, 10)
    assert res.allocation_failures == [2] and res.analytic_test["passed"] is False


def test_make_clusters():
    assert make_clusters(4, 2) == [[0, 1], [2, 3]]
    assert make_clusters(5, 2) == [[0, 1], [2, 3], [4]]
    assert make_clusters(3, 3) == [[0, 1, 2]]
    with pytest.raises(ValueError):
        make_clusters(4, 5)


def test_gfb_test_values():
    ok, total, bound = gfb_test(dens_tasks([0.5, 0.5, 0.5]), 2)
    assert ok and total == pytest.approx(1.5) and bound == pytest.approx(1.5)
    ok, _, _ = gfb_test(dens_tasks([0.2, 0.2, 1.0]), 2)
    assert not ok  # bound = 2 - 1*1.0 = 1.0 < 1.4


def test_cluster_allocation_policy():
    # densities 0.5,0.5,0.4,0.4 into two 2-core clusters, GFB bound = 2 - max density
    # T0 -> cl0 (0.5); T1 -> cl0 (1.0 <= 1.5); T2 -> cl0 (1.4 <= 1.5); T3: 1.8 > 1.5 -> cl1
    alloc, fails = allocate_to_clusters(dens_tasks([0.5, 0.5, 0.4, 0.4]), [[0, 1], [2, 3]])
    assert alloc == {0: 0, 1: 0, 2: 0, 3: 1} and fails == []
    # GFB is only sufficient: two heavy tasks (0.9) fill a 2-core cluster on their own
    # (1.8 > 2 - 0.9), so light tasks fit nowhere and are reported as allocation failures.
    alloc, fails = allocate_to_clusters(dens_tasks([0.9, 0.9, 0.3, 0.3]), [[0, 1], [2, 3]])
    assert alloc[0] == 0 and alloc[1] == 1 and fails == [2, 3]


# ---------------- textbook cases ----------------
def test_dhall_effect_global_misses_partitioned_meets():
    # Two light tasks + one heavy task on 2 cores (Dhall & Liu, 1978).
    tasks = [Task(0, "L1", 2, 10, 10), Task(1, "L2", 2, 10, 10), Task(2, "H", 10, 11, 11)]
    g = run_algorithm("Global EDF", tasks, 2, 110)
    p = run_algorithm("Partitioned EDF", tasks, 2, 110)
    heavy0 = [j for j in g.jobs if j.label == "H.J1"][0]
    assert heavy0.missed and heavy0.finish == 12
    assert not any(j.missed for j in p.jobs)


def test_global_succeeds_where_partitioning_fails():
    # Three tasks of utilization 0.6 on 2 cores: FFD cannot pack them, G-EDF meets all deadlines.
    tasks = [Task(0, "A", 3, 5, 5), Task(1, "B", 6, 10, 10), Task(2, "C", 6, 10, 10)]
    g = run_algorithm("Global EDF", tasks, 2, 10)
    p = run_algorithm("Partitioned EDF", tasks, 2, 10)
    assert p.allocation_failures and any(j.missed for j in p.jobs)
    assert not any(j.missed for j in g.jobs)


def test_cluster_size_extremes_match_partitioned_and_global():
    rng = np.random.default_rng(7)
    for _ in range(20):
        tasks = generate_taskset(8, 3.2, rng)
        H = hyperperiod(tasks)
        p = run_algorithm("Partitioned EDF", tasks, 4, H)
        c1 = run_algorithm("Clustered EDF", tasks, 4, H, cluster_size=1)
        g = run_algorithm("Global EDF", tasks, 4, H)
        c4 = run_algorithm("Clustered EDF", tasks, 4, H, cluster_size=4)
        assert p.timeline == c1.timeline
        assert g.timeline == c4.timeline


# ---------------- invariants on random task sets ----------------
def check_invariants(res):
    H = res.horizon
    jm = res.job_map()
    task = res.task_by_id()
    units = {k: 0 for k in jm}
    for t in range(H):
        running = [res.timeline[c][t] for c in res.timeline if res.timeline[c][t] is not None]
        # a job (and a task) never on two cores at the same time
        assert len(running) == len(set(running))
        assert len({k[0] for k in running}) == len(running)
        for k in running:
            j = jm[k]
            assert t >= j.release, "executed before release"
            assert j.finish is None or t < j.finish, "executed after completion"
            if j.aborted:
                assert t < j.abs_deadline, "aborted job executed after deadline"
            units[k] += 1
    for k, j in jm.items():
        assert units[k] == j.executed <= j.wcet
        assert j.abs_deadline == j.release + task[j.task_id].deadline
        if j.finish is not None:
            assert j.executed == j.wcet
        should_miss = (j.finish is not None and j.finish > j.abs_deadline) or \
                      (j.finish is None and j.abs_deadline <= H)
        assert j.missed == should_miss
    # migration restrictions
    for j in res.jobs:
        group = res.clusters[res.allocation[j.task_id]]
        assert j.cores_used <= set(group)
        if res.algorithm == "Partitioned EDF":
            assert j.migrations == 0


@pytest.mark.parametrize("policy", ["continue", "abort"])
@pytest.mark.parametrize("alg", ALGORITHMS)
def test_invariants_random(alg, policy):
    rng = np.random.default_rng(2026)
    for level in (0.5, 0.8, 1.0):
        for _ in range(15):
            m = int(rng.integers(2, 7))
            tasks = generate_taskset(int(rng.integers(m + 1, 3 * m + 2)), level * m, rng,
                                     deadline_type="constrained" if rng.random() < 0.5 else "implicit")
            res = run_algorithm(alg, tasks, m, hyperperiod(tasks), cluster_size=2, miss_policy=policy)
            check_invariants(res)


def test_unknown_algorithm():
    with pytest.raises(ValueError):
        run_algorithm("RMS", [Task(0, "A", 1, 2, 2)], 1, 4)


# ---------------- generator ----------------
def test_uunifast_sums_to_target():
    rng = np.random.default_rng(1)
    for n, U in [(5, 0.8), (10, 3.0), (3, 2.5)]:
        u = uunifast(n, U, rng)
        assert len(u) == n and sum(u) == pytest.approx(U) and min(u) >= 0


def test_uunifast_discard_caps_each_task():
    rng = np.random.default_rng(3)
    u = uunifast_discard(6, 4.5, rng)
    assert max(u) <= 1 and sum(u) == pytest.approx(4.5)
    with pytest.raises(ValueError):
        uunifast_discard(3, 3.5, rng)


def test_generated_tasks_are_valid():
    rng = np.random.default_rng(5)
    for dt in ("implicit", "constrained"):
        for t in generate_taskset(12, 4.0, rng, deadline_type=dt):
            assert 1 <= t.wcet <= t.deadline <= t.period
            if dt == "implicit":
                assert t.deadline == t.period
