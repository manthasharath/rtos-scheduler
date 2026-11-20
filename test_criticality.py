"""
Tests for criticality-first EDF, the High-only guarantee check, sporadic and
aperiodic tasks, the tie-break order, automatic priorities and the examples.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from algorithms import ALGORITHMS, run_algorithm
from examples import DEFAULT_EXAMPLE, EXAMPLES
from metrics import compute_metrics
from scheduler import edf_key, edf_schedule, sporadic_gap
from task_generator import generate_taskset
from task_model import Job, Task, fill_priorities, hyperperiod, validate_task_table


def labels(row, jobs):
    jm = {j.key: j for j in jobs}
    return [None if k is None else jm[k].task_name for k in row]


def test_hand_example_high_first():
    # One core. L: C=3, D=4 (Low).  H: C=4, D=6 (High). Both cannot meet.
    tasks = [Task(0, "L", 3, 10, 4, 1, "Low"), Task(1, "H", 4, 10, 6, 2, "High")]
    jobs, tl = edf_schedule(tasks, [0], 8)
    assert labels(tl[0], jobs)[:7] == ["L", "L", "L", "H", "H", "H", "H"]     # plain EDF
    assert [j.missed for j in jobs] == [False, True]                          # H misses
    jobs, tl = edf_schedule(tasks, [0], 8, crit_mode=True)
    assert labels(tl[0], jobs)[:7] == ["H", "H", "H", "H", "L", "L", "L"]     # High band first
    assert [j.missed for j in jobs] == [True, False]                          # L misses, H safe


def test_medium_and_low_share_one_band():
    # Medium and Low are ordered by EDF together (only High is a separate band).
    tasks = [Task(0, "M", 2, 10, 8, 1, "Medium"), Task(1, "L", 2, 10, 3, 2, "Low")]
    jobs, tl = edf_schedule(tasks, [0], 10, crit_mode=True)
    assert labels(tl[0], jobs)[:4] == ["L", "L", "M", "M"]


@pytest.mark.parametrize("alg", ALGORITHMS)
@pytest.mark.parametrize("policy", ["continue", "abort"])
def test_high_check_passed_means_no_high_miss(alg, policy):
    """Whenever the High-only design-time check says YES, criticality-first EDF
    never misses a High deadline, even when the whole set is overloaded."""
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(40):
        tasks = generate_taskset(8, float(rng.uniform(3.6, 5.0)), rng, sporadic_frac=0.25)
        crits = rng.choice(["High", "Medium", "Low"], size=len(tasks), p=[0.3, 0.35, 0.35])
        tasks = [dataclasses.replace(t, criticality=str(c)) for t, c in zip(tasks, crits)]
        res = run_algorithm(alg, tasks, 4, hyperperiod(tasks), 2, policy, crit_mode=True)
        if res.high_check["passed"]:
            checked += 1
            assert compute_metrics(res)["High misses"] == 0
    assert checked >= 5


def test_sporadic_gap_deterministic_and_at_least_T():
    assert sporadic_gap(1, 3, 4, 5) == sporadic_gap(1, 3, 4, 5)
    assert all(0 <= sporadic_gap(1, i, k, 5) <= 5 for i in range(5) for k in range(20))
    assert sporadic_gap(1, 0, 0, 0) == 0
    tasks = [Task(0, "S", 1, 10, 10, 1, "Low", kind="S")]
    jobs, _ = edf_schedule(tasks, [0], 200, gap_pct=50, seed=3)
    rel = [j.release for j in jobs]
    assert rel[0] == 0
    gaps = [b - a for a, b in zip(rel, rel[1:])]
    assert all(10 <= g <= 15 for g in gaps) and len(set(gaps)) > 1
    jobs2, _ = edf_schedule(tasks, [0], 200, gap_pct=50, seed=3)
    assert [j.release for j in jobs2] == rel


@pytest.mark.parametrize("alg", ALGORITHMS)
def test_aperiodic_never_changes_realtime_schedule(alg):
    ex = EXAMPLES[DEFAULT_EXAMPLE]
    tasks, errors, _ = validate_task_table(ex["df"])
    assert not errors
    rt_only = [t for t in tasks if t.kind != "A"]
    H = hyperperiod(tasks)
    a = run_algorithm(alg, tasks, 4, H, 2)
    b = run_algorithm(alg, rt_only, 4, H, 2)
    ap_keys = {j.key for j in a.aperiodic_jobs}
    for c in range(4):
        stripped = [None if k in ap_keys else k for k in a.timeline[c]]
        assert stripped == b.timeline[c]
    assert all(j.finish is not None for j in a.aperiodic_jobs)
    for j in a.aperiodic_jobs:
        assert j.start >= j.release


def test_tie_break_order():
    def job(tid, dl, crit, prio):
        return Job(task_id=tid, task_name=f"T{tid}", index=0, release=0, abs_deadline=dl, wcet=1,
                   remaining=1, priority=prio, crit_rank={"High": 3, "Medium": 2, "Low": 1}[crit])
    jobs = [job(0, 10, "Low", 1), job(1, 10, "High", 5), job(2, 10, "High", 2), job(3, 9, "Low", 9)]
    order = [j.task_id for j in sorted(jobs, key=edf_key)]
    assert order == [3, 2, 1, 0]       # deadline, then criticality, then priority number


def test_fill_priorities_by_period():
    df = pd.DataFrame({"Name": ["A", "B", "C", "X"], "Type": ["P", "P", "S", "A"], "C": [1, 1, 1, 2],
                       "T": [20, 5, 10, None], "D": [20, 5, 10, None], "Priority": [None, None, 7, None],
                       "Criticality": ["Low"] * 3 + [None], "Arrival": [None, None, None, 3]})
    out = fill_priorities(df)
    assert list(out["Priority"][:3]) == [3, 1, 7]          # blanks = rank by period; given value kept
    assert pd.isna(out["Priority"][3]) or out["Priority"][3] in ("", None)


@pytest.mark.parametrize("name", list(EXAMPLES))
def test_examples_valid(name):
    ex = EXAMPLES[name]
    tasks, errors, _ = validate_task_table(ex["df"])
    assert not errors and tasks


def _misses(name, alg, crit):
    ex = EXAMPLES[name]
    tasks, _, _ = validate_task_table(ex["df"])
    m = compute_metrics(run_algorithm(alg, tasks, ex["cores"], hyperperiod(tasks), ex["cluster"], crit_mode=crit))
    return m["Deadline misses"], m["High misses"], m["High tasks guaranteed"]


def test_example_claims():
    n = list(EXAMPLES)
    dhall, partfail, heavy, overload = n[1], n[2], n[3], n[4]
    assert _misses(dhall, "Global EDF", False)[0] > 0 and _misses(dhall, "Partitioned EDF", False)[0] == 0
    assert _misses(partfail, "Partitioned EDF", False)[0] > 0 and _misses(partfail, "Global EDF", False)[0] == 0
    assert _misses(heavy, "Clustered EDF", False)[0] == 0
    assert _misses(heavy, "Partitioned EDF", False)[0] > 0 and _misses(heavy, "Global EDF", False)[0] > 0
    for alg in ALGORITHMS:
        assert _misses(overload, alg, False)[1] > 0
        miss, high, guar = _misses(overload, alg, True)
        assert high == 0 and guar == "YES"
        assert _misses(DEFAULT_EXAMPLE, alg, False)[0] == 0
        assert _misses(DEFAULT_EXAMPLE, alg, True)[1] == 0


def test_fields_not_used_are_cleared():
    from task_model import clean_task_table
    df = pd.DataFrame({"Name": ["T1", "Diag"], "Type": ["P", "A"], "C": [1, 3], "T": [10, 10],
                       "D": [10, 5], "Priority": [None, 2], "Criticality": ["Low", "High"],
                       "Arrival": [4, 5]})
    clean, notes = clean_task_table(df)
    assert len(notes) == 2
    assert pd.isna(clean.loc[0, "Arrival"])
    assert all(pd.isna(clean.loc[1, c]) for c in ["T", "D", "Priority", "Criticality"])
    assert int(clean.loc[1, "Arrival"]) == 5
    tasks, errors, warnings = validate_task_table(clean)
    assert not errors and not warnings and len(tasks) == 2
