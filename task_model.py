"""
task_model.py
-------------
Data model: tasks, the jobs they release, and validation of the ONE task
table used by every version (website, terminal, Raspberry Pi, CSV files).

Task table columns:  Name, Type, C, T, D, Priority, Criticality, Arrival

  Type P = periodic : a job every T time units (first at t = 0)
  Type S = sporadic : first job at t = 0, then each next job after
                      T + a random extra gap (T is the MINIMUM gap)
  Type A = aperiodic: ONE job arriving at time 'Arrival'; served in the
                      background (only on idle cores). Uses C and Arrival only.

  C  = execution time (integer time units)
  T  = period / minimum gap,   D = relative deadline,   C <= D <= T
  Priority    = 1 = highest. Only used when two jobs have the same deadline
                AND the same criticality. Empty -> filled in automatically
                (shorter period = higher priority, Rate Monotonic rule).
  Criticality = Low / Medium / High.
                Plain EDF: used for tie-breaks and per-criticality results.
                Criticality-first EDF: High jobs always run before Medium/Low
                (see scheduler.py). This is NOT mixed-criticality scheduling
                (one execution time per task, no mode switch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from math import gcd
from typing import List, Optional, Set, Tuple

import pandas as pd

CRITICALITY_LEVELS = ["Low", "Medium", "High"]
CRIT_RANK = {"Low": 1, "Medium": 2, "High": 3}
TASK_TYPES = ["P", "S", "A"]
TYPE_NAMES = {"P": "Periodic", "S": "Sporadic", "A": "Aperiodic"}
MAX_CORES = 20
MAX_TASKS = 60
TASK_COLUMNS = ["Name", "Type", "C", "T", "D", "Priority", "Criticality", "Arrival"]


@dataclass(frozen=True)
class Task:
    """A task (immutable). kind: 'P' periodic, 'S' sporadic, 'A' aperiodic."""

    task_id: int
    name: str
    wcet: int                 # C
    period: int               # T (minimum gap for sporadic; 0 for aperiodic)
    deadline: int             # D (0 for aperiodic)
    priority: int = 1
    criticality: str = "Low"
    kind: str = "P"
    arrival: int = 0          # aperiodic only

    @property
    def is_realtime(self) -> bool:
        return self.kind != "A"

    @property
    def utilization(self) -> float:
        """u = C / T (0 for aperiodic jobs, they have no period)."""
        return self.wcet / self.period if self.is_realtime else 0.0

    @property
    def density(self) -> float:
        """delta = C / min(D, T). Equals utilization when D = T."""
        return self.wcet / min(self.deadline, self.period) if self.is_realtime else 0.0

    @property
    def crit_rank(self) -> int:
        return CRIT_RANK.get(self.criticality, 1)


@dataclass
class Job:
    """One job of a task (or the single job of an aperiodic task)."""

    task_id: int
    task_name: str
    index: int               # 0-based job number of this task
    release: int
    abs_deadline: int        # release + D  (-1 for aperiodic: no deadline)
    wcet: int
    remaining: int
    priority: int
    crit_rank: int = 1       # 1 Low, 2 Medium, 3 High
    kind: str = "P"
    start: Optional[int] = None
    finish: Optional[int] = None
    missed: bool = False
    aborted: bool = False
    executed: int = 0
    preemptions: int = 0
    migrations: int = 0
    last_core: Optional[int] = None
    cores_used: Set[int] = field(default_factory=set)

    @property
    def key(self) -> Tuple[int, int]:
        return (self.task_id, self.index)

    @property
    def label(self) -> str:
        return self.task_name if self.kind == "A" else f"{self.task_name}.J{self.index + 1}"

    @property
    def response_time(self) -> Optional[int]:
        return None if self.finish is None else self.finish - self.release


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def rt_tasks(tasks: List[Task]) -> List[Task]:
    """Real-time tasks (periodic + sporadic)."""
    return [t for t in tasks if t.is_realtime]


def ap_tasks(tasks: List[Task]) -> List[Task]:
    """Aperiodic jobs."""
    return [t for t in tasks if not t.is_realtime]


def lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def hyperperiod(tasks: List[Task]) -> int:
    """LCM of the periods of the real-time tasks."""
    return reduce(lcm, (t.period for t in rt_tasks(tasks)), 1)


def total_utilization(tasks: List[Task]) -> float:
    return sum(t.utilization for t in rt_tasks(tasks))


def total_density(tasks: List[Task]) -> float:
    return sum(t.density for t in rt_tasks(tasks))


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and v.strip() == "")


def normalize_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bring any task table / CSV into the standard column layout.
    Old files without Type/Arrival are treated as periodic; criticality may be
    written as L/M/H; Type may be lower case or written out.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in TASK_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[TASK_COLUMNS].dropna(how="all").reset_index(drop=True)
    df = df.astype(object)
    short = {"l": "Low", "low": "Low", "m": "Medium", "medium": "Medium", "h": "High", "high": "High"}
    kinds = {"p": "P", "periodic": "P", "s": "S", "sporadic": "S", "a": "A", "aperiodic": "A"}
    for i in range(len(df)):
        t = df.loc[i, "Type"]
        df.loc[i, "Type"] = "P" if _blank(t) else kinds.get(str(t).strip().lower(), str(t).strip())
        c = df.loc[i, "Criticality"]
        if not _blank(c):
            df.loc[i, "Criticality"] = short.get(str(c).strip().lower(), str(c).strip())
    return df


def fill_priorities(df: pd.DataFrame, overwrite: bool = False) -> pd.DataFrame:
    """
    Fill in EMPTY priorities of periodic/sporadic rows (all of them if
    overwrite=True): shorter period = higher priority, equal periods in table
    order. Aperiodic rows have no priority.
    """
    df = normalize_table(df)

    def period(i):
        try:
            return float(df.loc[i, "T"])
        except (TypeError, ValueError):
            return float("inf")
    rt_rows = [i for i in range(len(df)) if df.loc[i, "Type"] != "A"]
    rank = {i: r for r, i in enumerate(sorted(rt_rows, key=lambda i: (period(i), i)), start=1)}
    for i in rt_rows:
        if overwrite or _blank(df.loc[i, "Priority"]):
            df.loc[i, "Priority"] = rank[i]
    return df


def auto_priorities(df: pd.DataFrame) -> pd.DataFrame:
    """All priorities filled in automatically (shorter period first)."""
    return fill_priorities(df, overwrite=True)


def tasks_to_dataframe(tasks: List[Task]) -> pd.DataFrame:
    rows = []
    for t in tasks:
        if t.is_realtime:
            rows.append({"Name": t.name, "Type": t.kind, "C": t.wcet, "T": t.period, "D": t.deadline,
                         "Priority": t.priority, "Criticality": t.criticality, "Arrival": None})
        else:
            rows.append({"Name": t.name, "Type": "A", "C": t.wcet, "T": None, "D": None,
                         "Priority": None, "Criticality": None, "Arrival": t.arrival})
    return pd.DataFrame(rows, columns=TASK_COLUMNS)


def task_summary_dataframe(tasks: List[Task]) -> pd.DataFrame:
    df = tasks_to_dataframe(tasks)
    df["Utilization (C/T)"] = [round(t.utilization, 4) if t.is_realtime else None for t in tasks]
    df["Density (C/min(D,T))"] = [round(t.density, 4) if t.is_realtime else None for t in tasks]
    return df


# --------------------------------------------------------------------------
# Validation of the task table
# --------------------------------------------------------------------------
def _as_int(value, column: str, row_label: str, errors: List[str], minimum: int = 1) -> Optional[int]:
    if _blank(value):
        errors.append(f"{row_label}: '{column}' is empty.")
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        errors.append(f"{row_label}: '{column}' must be a whole number (got '{value}').")
        return None
    if f != int(f):
        errors.append(f"{row_label}: '{column}' must be a whole number (got {value}).")
        return None
    if int(f) < minimum:
        errors.append(f"{row_label}: '{column}' must be at least {minimum} (got {int(f)}).")
        return None
    return int(f)


def validate_task_table(df: pd.DataFrame) -> Tuple[List[Task], List[str], List[str]]:
    """
    Convert the task table into Task objects.
    Returns (tasks, errors, warnings). If `errors` is non-empty the task set
    must not be simulated.
    """
    errors: List[str] = []
    warnings: List[str] = []
    tasks: List[Task] = []

    if df is None or len(df) == 0:
        return [], ["The task table is empty. Add at least one task."], []
    df = fill_priorities(df)
    if len(df) == 0:
        return [], ["The task table is empty. Add at least one task."], []
    if len(df) > MAX_TASKS:
        errors.append(f"At most {MAX_TASKS} tasks are supported (got {len(df)}).")

    seen_names = set()
    for i in range(len(df)):
        row = df.loc[i]
        name = "" if _blank(row["Name"]) else str(row["Name"]).strip()
        row_label = f"Row {i + 1} ({name or 'unnamed'})"
        if not name:
            errors.append(f"Row {i + 1}: task name is empty.")
        elif name in seen_names:
            errors.append(f"{row_label}: duplicate task name '{name}'.")
        elif any(ch in name for ch in " ,"):
            errors.append(f"{row_label}: task name must not contain spaces or commas.")
        seen_names.add(name)
        kind = row["Type"]
        if kind not in TASK_TYPES:
            errors.append(f"{row_label}: Type must be P (periodic), S (sporadic) or A (aperiodic), got '{kind}'.")
            continue

        if kind == "A":
            c = _as_int(row["C"], "C (execution time)", row_label, errors)
            a = _as_int(row["Arrival"], "Arrival", row_label, errors, minimum=0)
            ignored = [col for col in ("T", "D", "Priority", "Criticality") if not _blank(row[col])]
            if ignored:
                warnings.append(f"{row_label}: aperiodic job - {', '.join(ignored)} not used (only C and Arrival).")
            if None in (c, a) or not name:
                continue
            tasks.append(Task(task_id=i, name=name, wcet=c, period=0, deadline=0, priority=0,
                              criticality="Low", kind="A", arrival=a))
            continue

        c = _as_int(row["C"], "C (execution time)", row_label, errors)
        t = _as_int(row["T"], "T (period / minimum gap)", row_label, errors)
        d = _as_int(row["D"], "D (deadline)", row_label, errors)
        p = _as_int(row["Priority"], "Priority", row_label, errors)
        crit = row["Criticality"]
        if crit not in CRITICALITY_LEVELS:
            errors.append(f"{row_label}: criticality must be High, Medium or Low (got '{crit}').")
        if not _blank(row["Arrival"]):
            warnings.append(f"{row_label}: Arrival is only used by aperiodic jobs (ignored).")
        if None in (c, t, d, p) or crit not in CRITICALITY_LEVELS or not name:
            continue
        if c > d:
            errors.append(f"{row_label}: C={c} is larger than D={d}; every job would miss. Make C <= D.")
            continue
        if d > t:
            errors.append(f"{row_label}: D={d} is larger than T={t}. Deadlines must satisfy D <= T.")
            continue
        tasks.append(Task(task_id=i, name=name, wcet=c, period=t, deadline=d, priority=p,
                          criticality=crit, kind=kind))

    if errors:
        return [], errors, warnings
    if not rt_tasks(tasks):
        return [], ["Add at least one periodic (P) or sporadic (S) task."], warnings
    hp = hyperperiod(tasks)
    if hp > 2000:
        warnings.append(f"The hyperperiod is {hp} time units; the horizon will be capped. Periods with common "
                        "factors (5, 10, 20, 25, 50, 100) keep it small.")
    return tasks, errors, warnings


NOT_USED = {"A": ("T", "D", "Priority", "Criticality"), "P": ("Arrival",), "S": ("Arrival",)}
NOT_USED_REASON = {"A": "aperiodic jobs have no period, deadline, priority or criticality",
                   "P": "periodic tasks always start at time 0",
                   "S": "sporadic tasks always start at time 0"}


def clean_task_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Empty the fields that do not belong to a task's type (used for CSV files and
    the website table): Type A keeps only Name, C, Arrival; P and S never use Arrival.
    Returns (cleaned table, notes saying what was removed).
    """
    df = normalize_table(df).reset_index(drop=True)
    notes: List[str] = []
    for i, row in df.iterrows():
        kind = str(row["Type"]).strip().upper() if not _blank(row["Type"]) else "P"
        cols = [c for c in NOT_USED.get(kind, ()) if not _blank(row[c])]
        if cols:
            for c in cols:
                df.at[i, c] = None
            notes.append(f"{row['Name']} ({TYPE_NAMES.get(kind, kind)}): {', '.join(cols)} removed - "
                         f"{NOT_USED_REASON[kind]}.")
    return df, notes


def display_table(df: pd.DataFrame) -> pd.DataFrame:
    """Copy for showing on screen: whole numbers, and '—' where a field is not used by that task type."""
    out = normalize_table(df).reset_index(drop=True).astype(object)
    for i, row in out.iterrows():
        kind = str(row["Type"]).strip().upper() if not _blank(row["Type"]) else "P"
        for c in TASK_COLUMNS:
            v = row[c]
            if c in NOT_USED.get(kind, ()):
                out.at[i, c] = "—"
            elif _blank(v):
                out.at[i, c] = ""
            elif c in ("C", "T", "D", "Priority", "Arrival"):
                try:
                    out.at[i, c] = str(int(float(v)))
                except (TypeError, ValueError):
                    out.at[i, c] = str(v)
            else:
                out.at[i, c] = str(v)
    out.index = range(1, len(out) + 1)
    return out


def read_task_csv(path_or_buffer) -> pd.DataFrame:
    """Read a task CSV (new or old format) into the standard table."""
    return fill_priorities(pd.read_csv(path_or_buffer, dtype=str, keep_default_na=False,
                                       na_values=[""]))


def default_task_table() -> pd.DataFrame:
    from examples import EXAMPLES, DEFAULT_EXAMPLE
    return EXAMPLES[DEFAULT_EXAMPLE]["df"].copy()
