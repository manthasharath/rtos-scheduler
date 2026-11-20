"""
Engine tests on small task sets whose schedules can be worked out by hand.
"""

import pytest

from scheduler import edf_schedule
from task_model import Task, validate_task_table, default_task_table, hyperperiod
import pandas as pd


def names(timeline_row, jobs):
    jm = {j.key: j for j in jobs}
    return [None if k is None else jm[k].label for k in timeline_row]


def test_uniprocessor_edf_exact_timeline():
    # A: C=1,T=4   B: C=2,T=6   on one core, H = LCM = 12
    # t0 A0(d4) | t1-2 B0(d6) | t3 idle | t4 A1(d8) | t5 idle | t6-7 B1(d12) | t8 A2(d12) | idle
    tasks = [Task(0, "A", 1, 4, 4), Task(1, "B", 2, 6, 6)]
    jobs, tl = edf_schedule(tasks, [0], 12)
    assert names(tl[0], jobs) == [
        "A.J1", "B.J1", "B.J1", None, "A.J2", None, "B.J2", "B.J2", "A.J3", None, None, None
    ]
    assert all(not j.missed for j in jobs)
    assert [j.response_time for j in jobs if j.task_name == "B"] == [3, 2]


def test_absolute_deadlines_and_releases():
    tasks = [Task(0, "A", 1, 5, 3), Task(1, "B", 2, 7, 7)]
    jobs, _ = edf_schedule(tasks, [0], 35)
    for j in jobs:
        task = tasks[j.task_id]
        assert j.release == j.index * task.period
        assert j.abs_deadline == j.release + task.deadline
    assert len([j for j in jobs if j.task_id == 0]) == 7
    assert len([j for j in jobs if j.task_id == 1]) == 5


def test_preemption_counted():
    # A: C=1,T=3   B: C=3,T=6. At t=3 A1 (d6) ties with B0 (d6); task id breaks tie -> A preempts B.
    tasks = [Task(0, "A", 1, 3, 3), Task(1, "B", 3, 6, 6)]
    jobs, tl = edf_schedule(tasks, [0], 6)
    assert names(tl[0], jobs) == ["A.J1", "B.J1", "B.J1", "A.J2", "B.J1", None]
    b0 = [j for j in jobs if j.label == "B.J1"][0]
    assert b0.preemptions == 1 and b0.finish == 5


def test_priority_breaks_deadline_ties():
    # Same as above but B has better user priority (1 < 2) -> B keeps running at t=3.
    tasks = [Task(0, "A", 1, 3, 3, priority=2), Task(1, "B", 3, 6, 6, priority=1)]
    jobs, tl = edf_schedule(tasks, [0], 6)
    assert names(tl[0], jobs) == ["A.J1", "B.J1", "B.J1", "B.J1", "A.J2", None]


def test_deadline_miss_continue_policy():
    # Two tasks C=3,T=4 on one core: U = 1.5, B0 must miss at t=4.
    tasks = [Task(0, "A", 3, 4, 4), Task(1, "B", 3, 4, 4)]
    jobs, _ = edf_schedule(tasks, [0], 8, "continue")
    b0 = [j for j in jobs if j.label == "B.J1"][0]
    assert b0.missed and not b0.aborted
    assert b0.finish == 6  # ran t=3, then late at t=4,5


def test_deadline_miss_abort_policy():
    tasks = [Task(0, "A", 3, 4, 4), Task(1, "B", 3, 4, 4)]
    jobs, tl = edf_schedule(tasks, [0], 8, "abort")
    b0 = [j for j in jobs if j.label == "B.J1"][0]
    assert b0.missed and b0.aborted and b0.finish is None and b0.executed == 1
    assert names(tl[0], jobs)[4] == "A.J2"  # aborted job no longer runs


def test_no_miss_reported_for_job_on_time():
    tasks = [Task(0, "A", 2, 4, 4)]
    jobs, _ = edf_schedule(tasks, [0], 4)
    assert not jobs[0].missed and jobs[0].finish == 2


def test_unfinished_job_with_deadline_at_horizon_is_missed():
    tasks = [Task(0, "A", 4, 4, 4), Task(1, "B", 1, 4, 4)]
    jobs, _ = edf_schedule(tasks, [0], 4)
    assert sum(j.missed for j in jobs) == 1


def test_one_task_never_runs_on_two_cores_even_when_late():
    # Overloaded: three C=2,T=2 tasks on 2 cores. Late jobs pile up, but two jobs
    # of the same task must never run in parallel.
    tasks = [Task(i, f"X{i}", 2, 2, 2) for i in range(3)]
    jobs, tl = edf_schedule(tasks, [0, 1], 40, "continue")
    for t in range(40):
        keys = [tl[c][t] for c in (0, 1) if tl[c][t] is not None]
        assert len({k[0] for k in keys}) == len(keys)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        edf_schedule([Task(0, "A", 1, 2, 2)], [0], 5, "drop")
    with pytest.raises(ValueError):
        edf_schedule([Task(0, "A", 1, 2, 2)], [], 5)


def test_validation_messages():
    df = pd.DataFrame([
        {"Name": "A", "C": 5, "T": 10, "D": 4, "Priority": 1, "Criticality": "Low"},     # C > D
        {"Name": "B", "C": 1, "T": 10, "D": 12, "Priority": 1, "Criticality": "Low"},    # D > T
        {"Name": "B", "C": 1, "T": 0, "D": 1, "Priority": 1, "Criticality": "Low"},      # dup + T<1
        {"Name": "C", "C": 1.5, "T": 10, "D": 10, "Priority": 1, "Criticality": "Ultra"},
    ])
    tasks, errors, _ = validate_task_table(df)
    assert tasks == []
    text = " ".join(errors)
    assert "larger than D" in text and "D <= T" in text
    assert "duplicate" in text and "at least 1" in text
    assert "whole number" in text and "criticality" in text


def test_default_table_is_valid():
    tasks, errors, warnings = validate_task_table(default_task_table())
    assert not errors and len(tasks) == 15 and hyperperiod(tasks) == 100
