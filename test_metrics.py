"""
Metric tests on hand-computed schedules.
"""

import pytest

from algorithms import run_algorithm
from experiments import run_batch
from export_results import to_excel_bytes
from metrics import compute_metrics, timeline_segments, task_metrics, context_switches
from task_model import Task


def test_utilization_and_response_times_single_core():
    # Schedule (see test_scheduler): A B B _ A _ B B A _ _ _  -> 7 busy units of 12
    tasks = [Task(0, "A", 1, 4, 4), Task(1, "B", 2, 6, 6)]
    m = compute_metrics(run_algorithm("Partitioned EDF", tasks, 1, 12))
    assert m["CPU utilization (observed)"] == pytest.approx(7 / 12, abs=1e-4)
    assert m["Task utilization (sum C/T)"] == pytest.approx(1 / 4 + 2 / 6, abs=1e-4)
    assert m["Released jobs"] == 5 and m["Completed jobs"] == 5
    # response times A:1,1,1  B:3,2
    assert m["Average response time"] == pytest.approx(1.6)
    assert m["Maximum response time"] == 3
    assert m["Deadline misses"] == 0 and m["Deadline miss ratio"] == 0
    assert m["Schedulable in horizon"] is True


def test_per_core_utilization_and_balance():
    # FFD: A (0.75) -> core0; B (0.5) does not fit on core0 (1.25) -> core1
    tasks = [Task(0, "A", 3, 4, 4), Task(1, "B", 2, 4, 4)]
    m = compute_metrics(run_algorithm("Partitioned EDF", tasks, 2, 8))
    assert m["Per-core utilization"] == {"Core 0": 0.75, "Core 1": 0.5}
    assert m["CPU utilization (observed)"] == pytest.approx(0.625)
    assert m["Load imbalance (max-min)"] == pytest.approx(0.25)


def test_miss_ratio():
    # One core, two C=3,T=4 tasks, H=8, continue policy.
    tasks = [Task(0, "A", 3, 4, 4), Task(1, "B", 3, 4, 4)]
    res = run_algorithm("Partitioned EDF", tasks, 1, 8)
    m = compute_metrics(res)
    # jobs: A1 ok, B1 late (finish 6), A2 (d8) runs 6,7 -> unfinished at 8 -> miss,
    # B2 (d8) never runs -> miss. Evaluated = 4, misses = 3.
    assert m["Released jobs"] == 4
    assert m["Deadline misses"] == 3
    assert m["Deadline miss ratio"] == pytest.approx(0.75)
    assert m["Schedulable in horizon"] is False


def test_segments_and_events():
    tasks = [Task(0, "A", 1, 3, 3), Task(1, "B", 3, 6, 6)]
    res = run_algorithm("Partitioned EDF", tasks, 1, 6)
    seg = timeline_segments(res)
    ex = seg[seg["Type"] == "Execution"]
    assert list(ex["Job"]) == ["A.J1", "B.J1", "A.J2", "B.J1"]
    assert list(ex["End event"]) == ["Completed", "Preempted", "Completed", "Completed"]
    assert seg["Duration"].sum() == 6
    assert context_switches(res) == 3


def test_task_metrics_table():
    tasks = [Task(0, "A", 3, 5, 5), Task(1, "B", 6, 10, 10), Task(2, "C", 6, 10, 10)]
    tm = task_metrics(run_algorithm("Partitioned EDF", tasks, 2, 10))
    assert tm["Allocation failed"].sum() == 1
    assert list(tm["Released"]) == [2, 1, 1]


def test_batch_runs_all_algorithms_on_same_sets():
    out = run_batch(num_tasks=6, num_cores=3, util_levels=[0.5, 0.9], sets_per_level=5, seed=11)
    res = out["results"]
    assert len(res) == 2 * 5 * 3
    # each experiment has exactly one row per algorithm with identical utilization
    for _, g in res.groupby("Experiment"):
        assert len(g) == 3 and g["Actual U"].nunique() == 1
    assert set(out["summary"]["Algorithm"]) == {"Partitioned EDF", "Global EDF", "Clustered EDF"}
    # reproducible with the same seed
    again = run_batch(num_tasks=6, num_cores=3, util_levels=[0.5, 0.9], sets_per_level=5, seed=11)
    assert res.equals(again["results"])
    assert to_excel_bytes({"r": res})[:2] == b"PK"  # valid xlsx (zip) bytes
