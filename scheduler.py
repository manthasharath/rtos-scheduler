"""
scheduler.py
------------
The discrete-time, preemptive EDF engine shared by ALL three algorithms (and
re-implemented line by line in hardware/rtos_hw.c for the Raspberry Pi).

edf_schedule(tasks, cores, horizon, ...) schedules the REAL-TIME tasks
(periodic + sporadic) of one group of cores:

    Partitioned EDF : one group per core     (tasks fixed to 1 core)
    Global EDF      : one group, all cores   (tasks may migrate anywhere)
    Clustered EDF   : one group per cluster  (tasks migrate only inside cluster)

Per time unit t = 0, 1, ..., horizon-1:
  1. RELEASE  periodic task: a job when t is a multiple of T.
              sporadic task: first job at 0, next job at previous release
              + T + extra gap (0 .. T*gap% /100, from sporadic_gap()).
  2. MISSES   an unfinished job whose absolute deadline <= t has missed;
              "abort" drops it, "continue" lets it finish late.
  3. ORDER    Plain EDF: earliest absolute deadline first.
              Criticality-first EDF: ALL High jobs before Medium/Low jobs;
              earliest deadline first inside each of the two levels.
              Ties: higher criticality, then priority number (1 = highest),
              then task order, then job number.
  4. PICK     up to (cores in group) jobs; only the OLDEST unfinished job of a
              task is eligible (jobs of a task run in order, never in parallel).
  5. ASSIGN   a job keeps the core it last ran on if that core is free,
              otherwise the lowest free core (a change of core = migration).
  6. COUNT    a preemption for every job that ran at t-1, is unfinished and
              was not picked now; then run every picked job for one unit.

Aperiodic jobs are added afterwards by serve_aperiodic(): every core that is
idle in a time unit runs the oldest waiting aperiodic job (background
service). They only use idle time, so they never change the real-time
schedule.

Guarantee of criticality-first EDF: lower jobs never take a core that a High
job wants, so High tasks behave as if they were alone. If the High-only test
of their core/cluster passes (see high_only_check), no High job can miss.

Assumptions: zero context-switch, migration and scheduling overhead; no shared
resources or blocking; independent tasks; integer time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from task_model import Job, Task

MISS_POLICIES = ("continue", "abort")
MASK = 0xFFFFFFFF

JobKey = Tuple[int, int]
Timeline = Dict[int, List[Optional[JobKey]]]


def sporadic_gap(seed: int, task_id: int, k: int, max_gap: int) -> int:
    """
    Extra gap after job k of a sporadic task: an integer in [0, max_gap].
    A fixed integer hash (identical in rtos_hw.c), so the same seed gives the
    same arrivals on the PC and on the Raspberry Pi.
    """
    if max_gap <= 0:
        return 0
    h = (seed * 0x9E3779B1 + (task_id + 1) * 0x85EBCA77 + (k + 1) * 0xC2B2AE3D) & MASK
    h ^= h >> 16
    h = (h * 0x7FEB352D) & MASK
    h ^= h >> 15
    h = (h * 0x846CA68B) & MASK
    h ^= h >> 16
    return h % (max_gap + 1)


def edf_key(job: Job):
    """EDF order. Equal deadlines: higher criticality, then priority number, task, job."""
    return (job.abs_deadline, -job.crit_rank, job.priority, job.task_id, job.index)


def crit_first_key(job: Job):
    """Criticality-first: High level before the Medium+Low level, EDF inside each."""
    return (0 if job.crit_rank == 3 else 1,) + edf_key(job)


@dataclass
class SimulationResult:
    algorithm: str
    tasks: List[Task]                         # all tasks (incl. aperiodic)
    num_cores: int
    horizon: int
    jobs: List[Job]                           # real-time jobs
    timeline: Timeline                        # core -> [job key or None] per time unit
    clusters: List[List[int]]                 # groups of cores used by the algorithm
    allocation: Dict[int, int] = field(default_factory=dict)      # task_id -> group index
    allocation_failures: List[int] = field(default_factory=list)  # task ids that did not fit
    analytic_test: Dict = field(default_factory=dict)             # full task set test
    high_check: Dict = field(default_factory=dict)                # High-only test
    miss_policy: str = "continue"
    cluster_size: Optional[int] = None
    crit_mode: bool = False
    aperiodic_jobs: List[Job] = field(default_factory=list)
    gap_pct: int = 50
    seed: int = 1

    @property
    def mode_name(self) -> str:
        return "Criticality-first EDF" if self.crit_mode else "Plain EDF"

    def job_map(self) -> Dict[JobKey, Job]:
        m = {j.key: j for j in self.jobs}
        m.update({j.key: j for j in self.aperiodic_jobs})
        return m

    def task_by_id(self) -> Dict[int, Task]:
        return {t.task_id: t for t in self.tasks}


def edf_schedule(
    tasks: Sequence[Task],
    cores: Sequence[int],
    horizon: int,
    miss_policy: str = "continue",
    crit_mode: bool = False,
    gap_pct: int = 50,
    seed: int = 1,
) -> Tuple[List[Job], Timeline]:
    """Preemptive (global) EDF of the real-time `tasks` on `cores` over [0, horizon)."""
    if miss_policy not in MISS_POLICIES:
        raise ValueError(f"miss_policy must be one of {MISS_POLICIES}")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    cores = list(cores)
    if not cores:
        raise ValueError("at least one core is required")
    tasks = [t for t in tasks if t.is_realtime]
    key = crit_first_key if crit_mode else edf_key

    timeline: Timeline = {c: [None] * horizon for c in cores}
    jobs: List[Job] = []
    ready: List[Job] = []
    next_index = {t.task_id: 0 for t in tasks}
    next_release = {t.task_id: 0 for t in tasks}
    running_prev: Dict[JobKey, int] = {}

    for t in range(horizon):
        # 1. releases
        for task in tasks:
            if t == next_release[task.task_id]:
                k = next_index[task.task_id]
                next_index[task.task_id] += 1
                gap = sporadic_gap(seed, task.task_id, k, task.period * gap_pct // 100) if task.kind == "S" else 0
                next_release[task.task_id] = t + task.period + gap
                job = Job(task_id=task.task_id, task_name=task.name, index=k, release=t,
                          abs_deadline=t + task.deadline, wcet=task.wcet, remaining=task.wcet,
                          priority=task.priority, crit_rank=task.crit_rank, kind=task.kind)
                jobs.append(job)
                ready.append(job)

        # 2. deadline misses
        for job in ready:
            if not job.missed and t >= job.abs_deadline:
                job.missed = True
                if miss_policy == "abort":
                    job.aborted = True
        if miss_policy == "abort":
            ready = [j for j in ready if not j.aborted]

        # 3 + 4. order and pick (oldest pending job of each task only)
        oldest = {}
        for job in ready:
            if job.task_id not in oldest or job.index < oldest[job.task_id]:
                oldest[job.task_id] = job.index
        ready.sort(key=key)
        selected: List[Job] = []
        for job in ready:
            if len(selected) == len(cores):
                break
            if job.index == oldest[job.task_id]:
                selected.append(job)
        selected.sort(key=edf_key)          # core assignment always in EDF order

        # 5. core assignment with affinity
        free = list(cores)
        assignment: Dict[JobKey, int] = {}
        for job in selected:
            if job.last_core is not None and job.last_core in free:
                assignment[job.key] = job.last_core
                free.remove(job.last_core)
        for job in selected:
            if job.key not in assignment:
                assignment[job.key] = free.pop(0)

        # 6. preemptions, then execute one time unit
        selected_keys = {j.key for j in selected}
        for job in ready:
            if job.key in running_prev and job.key not in selected_keys:
                job.preemptions += 1
        running_now: Dict[JobKey, int] = {}
        for job in selected:
            core = assignment[job.key]
            if job.last_core is not None and job.last_core != core:
                job.migrations += 1
            if job.start is None:
                job.start = t
            job.last_core = core
            job.cores_used.add(core)
            job.remaining -= 1
            job.executed += 1
            timeline[core][t] = job.key
            running_now[job.key] = core
            if job.remaining == 0:
                job.finish = t + 1
        ready = [j for j in ready if j.remaining > 0]
        running_prev = running_now

    for job in ready:
        if not job.missed and job.abs_deadline <= horizon:
            job.missed = True
    return jobs, timeline


def serve_aperiodic(ap: Sequence[Task], timeline: Timeline, horizon: int) -> List[Job]:
    """
    Background service: in every time unit, idle cores run the oldest waiting
    aperiodic jobs (arrival order, then table order). One core per job; a job
    keeps its previous core if that core is idle, otherwise the lowest idle core.
    Modifies `timeline` in place. Never touches real-time slots.
    """
    jobs = [Job(task_id=a.task_id, task_name=a.name, index=0, release=a.arrival, abs_deadline=-1,
                wcet=a.wcet, remaining=a.wcet, priority=0, crit_rank=0, kind="A") for a in ap]
    cores = sorted(timeline)
    for t in range(horizon):
        waiting = sorted([j for j in jobs if j.release <= t and j.remaining > 0], key=lambda j: (j.release, j.task_id))
        if not waiting:
            continue
        idle = [c for c in cores if timeline[c][t] is None]
        chosen = waiting[:len(idle)]
        assignment: Dict[JobKey, int] = {}
        for j in chosen:
            if j.last_core is not None and j.last_core in idle:
                assignment[j.key] = j.last_core
                idle.remove(j.last_core)
        for j in chosen:
            if j.key not in assignment:
                assignment[j.key] = idle.pop(0)
        for j in chosen:
            core = assignment[j.key]
            if j.last_core is not None and j.last_core != core:
                j.migrations += 1
            if j.start is None:
                j.start = t
            j.last_core = core
            j.cores_used.add(core)
            j.remaining -= 1
            j.executed += 1
            timeline[core][t] = j.key
            if j.remaining == 0:
                j.finish = t + 1
    return jobs


def merge_group_runs(
    runs: List[Tuple[List[Job], Timeline]], num_cores: int, horizon: int
) -> Tuple[List[Job], Timeline]:
    """Combine the results of several independent groups into one result."""
    jobs: List[Job] = []
    timeline: Timeline = {c: [None] * horizon for c in range(num_cores)}
    for group_jobs, group_tl in runs:
        jobs.extend(group_jobs)
        for core, row in group_tl.items():
            timeline[core] = row
    jobs.sort(key=lambda j: (j.release, j.task_id, j.index))
    return jobs, timeline
