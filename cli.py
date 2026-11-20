"""
cli.py  -  Terminal version of the multi-core RTOS scheduling simulator
=======================================================================
Run:   python cli.py

Same engine as the website. Menu (same numbers as the Raspberry Pi program):

   1  Number of cores                 6  Choose algorithms
   2  Cluster size                    7  Run: results + Gantt charts
   3  Enter tasks (manual/CSV/random/example)
   4  Simulation settings             8  Batch experiments (random task sets)
   5  Show / check / edit tasks       9  Export results (CSV + Excel)
                                      0  Exit

It starts with a GUIDED RUN: steps 1-6 in order, Enter keeps the value in [ ],
then the run, then "What next?" (run again, new run, compare modes, batch, export).
At ANY question you can type  X  for the full menu or  Q  to quit.
"""

from __future__ import annotations

import os
import string
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from algorithms import ALGORITHM_CHOICES, ALGORITHMS, run_algorithm
from clustered_edf import make_clusters
from examples import DEFAULT_EXAMPLE, EXAMPLES
from experiments import run_batch
from export_results import to_excel_bytes
from metrics import (aperiodic_table, comparison_table, compute_metrics, criticality_metrics, high_check_table,
                     job_table, mode_comparison, task_metrics, timeline_segments)
from task_generator import generate_taskset
from task_model import (MAX_CORES, MAX_TASKS, TASK_COLUMNS, clean_task_table, display_table, fill_priorities,
                        hyperperiod, read_task_csv,
                        tasks_to_dataframe, total_density, total_utilization, validate_task_table)

MAX_HORIZON = 5000
SYMBOLS = string.ascii_uppercase + string.ascii_lowercase + string.digits   # one character per task
LINE = "-" * 78

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 300)


class GoMenu(Exception):
    """User typed X: back to the main menu."""


class QuitApp(Exception):
    """User typed Q: leave the program."""


# =====================================================================
# Input helpers - every prompt re-asks until the value is valid.
# X = main menu, Q = quit, at every prompt.
# =====================================================================
def ask(prompt: str, default=None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    try:
        text = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        raise QuitApp()
    if text.upper() in ("X", "MENU"):
        raise GoMenu()
    if text.upper() == "Q":
        raise QuitApp()
    return str(default) if (text == "" and default is not None) else text


def ask_int(prompt: str, lo: int, hi: int, default=None, allow_blank=False):
    while True:
        text = ask(f"{prompt} ({lo}-{hi}{', Enter = automatic' if allow_blank else ''})", default)
        if allow_blank and text == "":
            return None
        try:
            value = int(text)
        except ValueError:
            print(f"  ! Please enter a whole number between {lo} and {hi}.")
            continue
        if lo <= value <= hi:
            return value
        print(f"  ! Value must be between {lo} and {hi}.")


def ask_float(prompt: str, lo: float, hi: float, default=None) -> float:
    while True:
        text = ask(f"{prompt} ({lo}-{hi})", default)
        try:
            value = float(text)
        except ValueError:
            print(f"  ! Please enter a number between {lo} and {hi}.")
            continue
        if lo <= value <= hi:
            return value
        print(f"  ! Value must be between {lo} and {hi}.")


def ask_choice(prompt: str, options: dict, default=None) -> str:
    for key, label in options.items():
        print(f"   {key}. {label}")
    while True:
        choice = ask(prompt, default).upper()
        if choice in options:
            return choice
        print(f"  ! Please choose one of: {', '.join(options)}")


def ask_yes_no(prompt: str, default="y") -> bool:
    while True:
        a = ask(prompt + " (y/n)", default).lower()
        if a in ("y", "yes"):
            return True
        if a in ("n", "no"):
            return False
        print("  ! Please type y or n.")


def ask_criticality(prompt: str, default="Low") -> str:
    short = {"l": "Low", "low": "Low", "m": "Medium", "medium": "Medium", "h": "High", "high": "High"}
    while True:
        a = ask(prompt + " (H/M/L)", default).lower()
        if a in short:
            return short[a]
        print("  ! Type H (High), M (Medium) or L (Low).")


def ask_type(prompt: str, default="P") -> str:
    kinds = {"p": "P", "periodic": "P", "s": "S", "sporadic": "S", "a": "A", "aperiodic": "A"}
    while True:
        a = ask(prompt + " (P = periodic, S = sporadic, A = aperiodic)", default).lower()
        if a in kinds:
            return kinds[a]
        print("  ! Type P, S or A.")


def ask_name(prompt: str, used, default=None) -> str:
    while True:
        name = ask(prompt, default)
        if not name or any(ch in name for ch in " ,"):
            print("  ! Name must not be empty and must not contain spaces or commas.")
        elif name in used:
            print(f"  ! '{name}' is already used. Choose another name.")
        else:
            return name


def header(title: str):
    print("\n" + "=" * 78)
    print(" " + title)
    print("=" * 78)


def pause():
    ask("\nPress Enter to continue (X = menu, Q = quit)", "")


# =====================================================================
# Session state
# =====================================================================
class Session:
    def __init__(self):
        ex = EXAMPLES[DEFAULT_EXAMPLE]
        self.cores = ex["cores"]
        self.cluster = ex["cluster"]
        self.tasks_df = ex["df"].copy()
        self.source = f"example: {DEFAULT_EXAMPLE}"
        self.horizon_mode = "hyperperiod"   # or an int
        self.miss_policy = "continue"
        self.crit_mode = False
        self.gap_pct = 50
        self.seed = 1
        self.algo_choice = "7"
        self.results = {}
        self.mode_table = None
        self.batch = None

    @property
    def algorithms(self):
        return ALGORITHM_CHOICES[self.algo_choice][1]

    def tasks(self):
        return validate_task_table(self.tasks_df)

    def horizon(self, tasks) -> int:
        if self.horizon_mode == "hyperperiod":
            return min(hyperperiod(tasks), MAX_HORIZON)
        return int(self.horizon_mode)

    def mode_text(self) -> str:
        return "Criticality-first EDF" if self.crit_mode else "Plain EDF"

    def invalidate(self):
        self.results = {}
        self.mode_table = None


# =====================================================================
# 1, 2  Cores and cluster size
# =====================================================================
def set_cores(s: Session):
    header("1. NUMBER OF CORES")
    print(f" Up to {MAX_CORES} cores. (Our comparisons use 4 and 8 cores; the Raspberry Pi has 4.)")
    s.cores = ask_int("Number of cores", 1, MAX_CORES, s.cores)
    if s.cluster > s.cores:
        s.cluster = s.cores
        print(f" Cluster size reduced to {s.cluster} (cannot be larger than the number of cores).")
    s.invalidate()
    print(f" OK: {s.cores} cores.")


def set_cluster(s: Session):
    header("2. CLUSTER SIZE (used by Clustered EDF)")
    print(" Cores are split into groups of this size. 1 = like Partitioned, "
          f"{s.cores} = like Global.")
    s.cluster = ask_int("Cores per cluster", 1, s.cores, min(s.cluster, s.cores))
    s.invalidate()
    groups = make_clusters(s.cores, s.cluster)
    print(" Clusters: " + "  ".join(f"C{i + 1}=cores {g[0]}-{g[-1]}" for i, g in enumerate(groups)))


# =====================================================================
# 3  Task entry
# =====================================================================
def enter_one_task(used, number: int) -> dict:
    print(f"\n Task {number}:")
    name = ask_name("   Name (no spaces)", used, f"T{number}")
    kind = ask_type("   Type", "P")
    if kind == "A":
        c = ask_int("   C = execution time", 1, 1000, 2)
        arr = ask_int("   Arrival time", 0, MAX_HORIZON, 0)
        return {"Name": name, "Type": "A", "C": c, "T": None, "D": None, "Priority": None,
                "Criticality": None, "Arrival": arr}
    word = "minimum gap between releases" if kind == "S" else "period"
    c = ask_int("   C = execution time (WCET)", 1, 1000, 1)
    t = ask_int(f"   T = {word}", max(c, 1), 10000, max(10, c))
    d = ask_int("   D = relative deadline (C <= D <= T)", c, t, t)
    p = ask_int("   Priority, 1 = highest (only breaks deadline ties)", 1, 999, None, allow_blank=True)
    crit = ask_criticality("   Criticality", "Low")
    return {"Name": name, "Type": kind, "C": c, "T": t, "D": d, "Priority": p, "Criticality": crit,
            "Arrival": None}


def enter_tasks_manually(s: Session):
    n = ask_int("How many tasks", 1, MAX_TASKS, 4)
    rows, used = [], set()
    for i in range(n):
        row = enter_one_task(used, i + 1)
        used.add(row["Name"])
        rows.append(row)
    s.tasks_df = fill_priorities(pd.DataFrame(rows, columns=TASK_COLUMNS))
    s.source = "entered manually"


def load_tasks_csv(s: Session):
    print(" CSV columns: Name,Type,C,T,D,Priority,Criticality,Arrival")
    print("   periodic/sporadic row:  EngineCtl,P,2,5,5,,High,")
    print("   aperiodic row:          DiagScan,A,6,,,,,12")
    print(" Priority may be empty (= automatic, shorter period first). Old 6-column files also work.")
    print(" Example files are in the folder example_csv")
    while True:
        path = ask("CSV file path", "task_configuration.csv").strip('"')
        if not os.path.exists(path):
            print(f"  ! File not found: {os.path.abspath(path)}")
            continue
        try:
            df, notes = clean_task_table(read_task_csv(path))
        except Exception as exc:                      # noqa: BLE001 - show any read problem
            print(f"  ! Could not read the CSV: {exc}")
            continue
        for n in notes:
            print("  note: " + n)
        s.tasks_df = df
        s.source = f"CSV: {path}"
        return


def generate_random_tasks(s: Session):
    print(" UUniFast-Discard: total utilization U is split randomly over the tasks.")
    n = ask_int("Number of real-time tasks (P + S)", 2, MAX_TASKS, 8)
    u = ask_float(f"Total utilization U (e.g. {0.7 * s.cores:.1f} = 70 % of {s.cores} cores)",
                  0.1, float(min(n, s.cores * 1.5)), round(0.7 * s.cores, 2))
    sp = ask_int("How many % of them sporadic", 0, 100, 0)
    na = ask_int("Number of aperiodic jobs", 0, max(0, MAX_TASKS - n), 0)
    seed = ask_int("Random seed (same seed = same task set)", 0, 10 ** 6, 42)
    tasks = generate_taskset(n, u, np.random.default_rng(seed), sporadic_frac=sp / 100, n_aperiodic=na)
    s.tasks_df = tasks_to_dataframe(tasks)
    s.source = f"random (U={u}, seed={seed})"


def load_example(s: Session):
    names = list(EXAMPLES)
    opts = {str(i + 1): n for i, n in enumerate(names)}
    k = ask_choice("Example", opts, "1")
    ex = EXAMPLES[opts[k]]
    s.tasks_df = ex["df"].copy()
    if ex["cores"] != s.cores and ask_yes_no(f" This example is made for {ex['cores']} cores "
                                             f"(you chose {s.cores}). Use {ex['cores']} cores?", "y"):
        s.cores, s.cluster = ex["cores"], ex["cluster"]
    s.cluster = min(s.cluster, s.cores)
    s.source = f"example: {opts[k]}"
    print(f"\n Cores = {s.cores}, cluster size = {s.cluster}.")
    print(" " + ex["description"])


def task_entry(s: Session):
    header("3. ENTER TASKS")
    k = ask_choice("How", {"A": "Manual entry (type each task)",
                           "B": "Load a CSV file",
                           "C": "Random task set (UUniFast)",
                           "D": "Built-in example (5 prepared cases)"}, "D")
    {"A": enter_tasks_manually, "B": load_tasks_csv, "C": generate_random_tasks, "D": load_example}[k](s)
    s.invalidate()
    show_tasks(s, edit=False)


# =====================================================================
# 4  Settings
# =====================================================================
def settings(s: Session):
    header("4. SIMULATION SETTINGS")
    print(f" Now: horizon = {s.horizon_mode}, miss policy = {s.miss_policy}, EDF mode = {s.mode_text()}, "
          f"sporadic extra gap = 0-{s.gap_pct} % of T, seed = {s.seed}")
    k = ask_choice("Horizon", {"1": "One hyperperiod (recommended)", "2": "Fixed number of time units"},
                   "1" if s.horizon_mode == "hyperperiod" else "2")
    s.horizon_mode = "hyperperiod" if k == "1" else ask_int("Horizon", 1, MAX_HORIZON, 100)
    k = ask_choice("When a job misses its deadline", {
        "1": "Continue: it keeps running late (shows how late)",
        "2": "Abort: it is dropped at the deadline"}, "1" if s.miss_policy == "continue" else "2")
    s.miss_policy = "continue" if k == "1" else "abort"
    k = ask_choice("EDF mode", {
        "1": "Plain EDF: earliest deadline first, criticality only breaks ties",
        "2": "Criticality-first EDF: High jobs always first; Medium + Low by EDF"},
        "2" if s.crit_mode else "1")
    s.crit_mode = k == "2"
    s.gap_pct = ask_int("Sporadic tasks: max extra gap after T, in % of T", 0, 200, s.gap_pct)
    s.seed = ask_int("Seed for sporadic gaps (Pi uses the same rule)", 0, 10 ** 6, s.seed)
    s.invalidate()
    print(" Settings saved.")


# =====================================================================
# 5  Show / check / edit
# =====================================================================
def analysis_lines(s: Session, tasks):
    rt = [t for t in tasks if t.is_realtime]
    u, dens = total_utilization(tasks), total_density(tasks)
    print(f"\n Real-time tasks: {len(rt)} (P {sum(t.kind == 'P' for t in rt)}, "
          f"S {sum(t.kind == 'S' for t in rt)})   Aperiodic jobs: {len(tasks) - len(rt)}")
    print(f" Total utilization U = {u:.3f}  ->  {u / s.cores * 100:.1f} % of {s.cores} cores"
          f"   (density {dens:.3f})   Hyperperiod = {hyperperiod(tasks)}")
    if u > s.cores:
        print(" ! U > number of cores: overloaded, deadline misses are certain.")
    hi = [t for t in rt if t.criticality == "High"]
    print(f" High tasks: {len(hi)}  (load {sum(t.utilization for t in hi):.2f})")


def show_tasks(s: Session, edit=True):
    while True:
        header(f"5. TASK SET  ({s.source})   cores = {s.cores}, cluster = {s.cluster}")
        df = fill_priorities(s.tasks_df)
        print(display_table(df).to_string())
        print(" (— = not used by that task type)")
        tasks, errors, warnings = validate_task_table(df)
        for w in warnings:
            print(" warning: " + w)
        if errors:
            print("\n ERRORS - fix them before running:")
            for e in errors:
                print("   * " + e)
        else:
            analysis_lines(s, tasks)
        if not edit:
            if errors:
                print(" Use menu 5 to edit the table.")
            return
        k = ask_choice("Action", {"C": "Values are correct - confirm",
                                  "E": "Edit a task", "A": "Add a task", "D": "Delete a task"}, "C")
        if k == "C":
            if errors:
                print("  ! There are still errors; please fix them first (or X for menu).")
                continue
            s.tasks_df = df
            print(" Confirmed.")
            return
        if k == "A":
            if len(df) >= MAX_TASKS:
                print(f"  ! At most {MAX_TASKS} tasks.")
                continue
            row = enter_one_task(set(df["Name"].astype(str)), len(df) + 1)
            s.tasks_df = fill_priorities(pd.concat([df, pd.DataFrame([row], columns=TASK_COLUMNS)],
                                                   ignore_index=True))
        elif k == "D":
            i = ask_int("Row number to delete", 1, len(df))
            s.tasks_df = df.drop(index=i - 1).reset_index(drop=True)
        else:
            i = ask_int("Row number to edit", 1, len(df))
            used = set(df["Name"].astype(str)) - {str(df.loc[i - 1, "Name"])}
            print(" Enter the new values for this row:")
            row = enter_one_task(used, i)
            df.loc[i - 1] = [row[c] for c in TASK_COLUMNS]
            s.tasks_df = fill_priorities(df)
        s.source += " (edited)"
        s.invalidate()


# =====================================================================
# 6  Algorithm choice
# =====================================================================
def choose_algorithms(s: Session):
    header("6. CHOOSE ALGORITHMS")
    s.algo_choice = ask_choice("Run which", {k: v[0] for k, v in ALGORITHM_CHOICES.items()}, s.algo_choice)
    s.invalidate()
    print(" Chosen: " + ", ".join(s.algorithms))


# =====================================================================
# 7  Run, results, Gantt
# =====================================================================
def run(s: Session) -> bool:
    tasks, errors, _ = s.tasks()
    if errors:
        print("\n The task table has errors (see menu 5):")
        for e in errors:
            print("   * " + e)
        return False
    H = s.horizon(tasks)
    s.results = {a: run_algorithm(a, tasks, s.cores, H, s.cluster, s.miss_policy, s.crit_mode, s.gap_pct, s.seed)
                 for a in s.algorithms}
    return True


def print_metrics(r):
    m = compute_metrics(r)
    header(f"RESULTS - {r.algorithm}, {r.mode_name}   ({r.num_cores} cores, horizon {r.horizon})")
    verdict = ("NO deadline misses" if m["Schedulable in horizon"]
               else f"{m['Deadline misses']} DEADLINE MISS(ES)  (High {m['High misses']}, "
                    f"Medium {m['Medium misses']}, Low {m['Low misses']})")
    print(f" Simulation         : {verdict}")
    at = r.analytic_test
    print(f" Design-time test   : {at['name']} -> {'PASSED' if at['passed'] else 'NOT PASSED'}")
    print(f"                      {at['detail']}")
    print(f" High tasks guaranteed (criticality-first, High-only test): {m['High tasks guaranteed']}")
    rows = [
        ("CPU utilization (observed)", f"{m['CPU utilization (observed)'] * 100:.1f} %"),
        ("Task-set utilization U (sum C/T)", f"{m['Task utilization (sum C/T)']:.3f}"),
        ("Released / completed jobs", f"{m['Released jobs']} / {m['Completed jobs']}"),
        ("Deadline miss ratio", f"{m['Deadline miss ratio'] * 100:.2f} %"),
        ("Average / maximum response time", f"{m['Average response time']} / {m['Maximum response time']}"),
        ("Preemptions", m["Preemptions"]),
        ("Migrations", m["Migrations"]),
        ("Context switches (zero cost modelled)", m["Context switches"]),
        ("Load imbalance (max - min core util.)", f"{m['Load imbalance (max-min)'] * 100:.1f} %"),
        ("Allocation failures", m["Allocation failures"]),
    ]
    print()
    for label, value in rows:
        print(f"   {label:<40} {value}")
    print("\n Per-core utilization:")
    for core, u in m["Per-core utilization"].items():
        print(f"   {core:<8} {'#' * int(round(u * 40)):<40} {u * 100:5.1f} %")
    print("\n By criticality:")
    print(criticality_metrics(r).drop(columns=["Algorithm", "EDF mode"]).to_string(index=False))
    hc = high_check_table(r)
    if len(hc):
        print("\n High-only check per core group:")
        print(hc.drop(columns=["Algorithm"]).to_string(index=False))
    print("\n Per task:")
    tm = task_metrics(r)[["Task", "Type", "C", "T", "D", "Priority", "Criticality", "Assigned to", "Cores used",
                          "Released", "Completed", "Missed", "Avg response", "Max response", "Preemptions",
                          "Migrations"]]
    print(tm.to_string(index=False))
    ap = aperiodic_table(r)
    if len(ap):
        print(f"\n Aperiodic jobs (served in background = idle cores only): "
              f"{m['Aperiodic completed']}/{m['Aperiodic jobs']} done, avg response {m['Aperiodic avg response']}")
        print(ap.drop(columns=["Algorithm"]).to_string(index=False))
    missed = [j for j in r.jobs if j.missed]
    if missed:
        print("\n Jobs that missed their deadline:")
        for j in missed[:30]:
            fin = j.finish if j.finish is not None else "not finished"
            print(f"   {j.label:<16} released {j.release:<5} deadline {j.abs_deadline:<5} "
                  f"finished {fin}  (executed {j.executed}/{j.wcet})")
        if len(missed) > 30:
            print(f"   ... and {len(missed) - 30} more")


def symbols(r):
    return {t.task_id: SYMBOLS[i % len(SYMBOLS)] for i, t in enumerate(r.tasks)}


def text_gantt(r, start=0, end=None, width=60, title=True):
    """Text Gantt chart: one character = one time unit."""
    end = min(r.horizon, end if end is not None else r.horizon)
    sym = symbols(r)
    jm = r.job_map()
    if title:
        header(f"GANTT CHART - {r.algorithm}, {r.mode_name}   (t = {start} to {end})")
        print(" Legend: " + "  ".join(f"{sym[t.task_id]}={t.name}{'(A)' if t.kind == 'A' else ''}"
                                      for t in r.tasks) + "  .=idle")
        print(" Marks below a core row:  X = deadline miss,  m = job migrated onto this core\n")
    cluster_of = {c: i + 1 for i, cl in enumerate(r.clusters) for c in cl}
    miss_at = {}
    for j in r.jobs:
        if j.missed and j.abs_deadline <= r.horizon:
            core = j.last_core if j.last_core is not None else min(r.clusters[r.allocation.get(j.task_id, 0)])
            miss_at.setdefault(core, set()).add(min(j.abs_deadline, r.horizon - 1))
    seg = timeline_segments(r, include_idle=False)
    mig_at = {}
    if len(seg):
        for _, row in seg[seg["Migrated in"]].iterrows():
            mig_at.setdefault(int(row["Core"]), set()).add(int(row["Start"]))
    for chunk in range(start, end, width):
        stop = min(chunk + width, end)
        ruler = "".join("|" if t % 10 == 0 else ("+" if t % 5 == 0 else " ") for t in range(chunk, stop))
        nums, t = "", chunk
        while t < stop:
            if t % 10 == 0:
                nums += str(t)
                t += len(str(t))
            else:
                nums += " "
                t += 1
        print(f" {'time':<12} {nums[:stop - chunk]}")
        print(f" {'':<12} {ruler}")
        for core in sorted(r.timeline):
            label = f"Core {core}" + (f" C{cluster_of[core]}" if r.algorithm == "Clustered EDF" else "")
            row = "".join("." if r.timeline[core][t] is None else sym[jm[r.timeline[core][t]].task_id]
                          for t in range(chunk, stop))
            print(f" {label:<12} {row}")
            marks = "".join("X" if t in miss_at.get(core, ()) else ("m" if t in mig_at.get(core, ()) else " ")
                            for t in range(chunk, stop))
            if marks.strip():
                print(f" {'':<12} {marks}")
        print()


def tick_events(r, t):
    ev = []
    for j in r.jobs + r.aperiodic_jobs:
        if j.release == t:
            ev.append(f"{j.label} released")
        if j.finish == t + 1:
            ev.append(f"{j.label} finished")
        if j.missed and j.abs_deadline == t + 1:
            ev.append(f"{j.label} MISSED deadline")
    return ev


def step_by_step(results):
    """Build the Gantt charts of all chosen algorithms one tick at a time."""
    H = max(r.horizon for r in results.values())
    print("\n Step mode: Enter = next tick, a number = jump that many ticks, P = play automatically, "
          "S = stop stepping")
    t, auto = 0, False
    while t < H:
        for r in results.values():
            start = max(0, t + 1 - 60)
            print(f"\n --- {r.algorithm}, t = {t} ---")
            text_gantt(r, start, t + 1, title=False)
            ev = tick_events(r, t)
            if ev:
                print("   events: " + "; ".join(ev))
        if auto:
            time.sleep(0.3)
            t += 1
            continue
        a = ask("Next", "")
        if a.upper() == "S":
            return
        if a.upper() == "P":
            auto = True
            t += 1
        elif a.isdigit() and int(a) > 0:
            t = min(H - 1, t + int(a))
        else:
            t += 1


def show_comparison(s: Session):
    header(f"COMPARISON - {s.mode_text()} (same task set, same horizon)")
    table = comparison_table(s.results)
    cols = ["Algorithm", "Deadline misses", "High misses", "Medium misses", "Low misses", "Deadline miss ratio",
            "CPU utilization (observed)", "Average response time", "Maximum response time", "Preemptions",
            "Migrations", "Allocation failures", "High tasks guaranteed", "Aperiodic avg response"]
    print(table[[c for c in cols if c in table]].set_index("Algorithm").T.to_string())


def run_and_show(s: Session) -> bool:
    header("RUN AND RESULTS")
    print(f" Algorithms: {', '.join(s.algorithms)}   Mode: {s.mode_text()}   Cores: {s.cores}")
    if not run(s):
        return False
    for r in s.results.values():
        print_metrics(r)
    if len(s.results) > 1:
        show_comparison(s)
    for r in s.results.values():
        text_gantt(r)
    return True


def compare_modes_now(s: Session):
    tasks, _, _ = s.tasks()
    s.mode_table = mode_comparison(tasks, s.cores, s.horizon(tasks), s.cluster, s.miss_policy,
                                   s.gap_pct, s.seed, s.algorithms)
    header("PLAIN vs CRITICALITY-FIRST EDF")
    print(s.mode_table.to_string(index=False))


def settings_line(s: Session) -> str:
    return (f" Settings: horizon {s.horizon_mode}, on miss {s.miss_policy}, {s.mode_text()}, "
            f"sporadic gap 0-{s.gap_pct} % of T, seed {s.seed}")


def guided_flow(s: Session):
    """Normal use: all steps in order; Enter keeps the value in [ ]."""
    full = True
    while True:
        if full:
            print("\n========== GUIDED RUN  (Enter = keep value in [ ],  X = full menu,  Q = quit) ==========")
            print("\n STEP 1/6  Cores")
            set_cores(s)
            print("\n STEP 2/6  Cluster size")
            set_cluster(s)
            print("\n STEP 3/6  Tasks")
            task_entry(s)
            print("\n STEP 4/6  Check the values")
            show_tasks(s, edit=True)
            print("\n STEP 5/6  Algorithms")
            choose_algorithms(s)
        print("\n STEP 6/6  Settings")
        print(settings_line(s))
        if ask_yes_no("Change settings?", "n"):
            settings(s)
        if not run_and_show(s):
            full = True
            continue
        exported = False
        while True:
            print("\n WHAT NEXT?\n   R. Run again (same tasks, change settings if you like)\n   N. New run from step 1\n"
                  "   S. Watch the Gantt charts build step by step\n"
                  "   C. Compare plain EDF vs criticality-first EDF (same tasks)\n"
                  "   B. Batch experiments (random task sets)\n   E. Export results (CSV + Excel)\n"
                  "   X. Full menu\n   Q. Quit")
            k = ask("Choice", "R" if exported else "E").upper()
            if k == "S":
                step_by_step(s.results)
            elif k == "C":
                compare_modes_now(s)
            elif k == "B":
                batch_experiments(s)
            elif k == "E":
                export(s)
                exported = True
            elif k in ("R", "N"):
                break
            else:
                print("  ! Choose R, N, S, C, B, E, X or Q.")
        full = k == "N"


def results_menu(s: Session):
    if not run_and_show(s):
        return
    while True:
        k = ask_choice("Next", {"1": "Watch the Gantt charts build step by step",
                                "2": "Compare plain EDF vs criticality-first EDF on this task set",
                                "3": "Show results again",
                                "X": "Main menu"}, "X")
        if k == "X":
            return
        if k == "1":
            step_by_step(s.results)
        elif k == "2":
            compare_modes_now(s)
        else:
            for r in s.results.values():
                print_metrics(r)
                text_gantt(r)


# =====================================================================
# 8  Batch
# =====================================================================
def batch_experiments(s: Session):
    header("8. BATCH EXPERIMENTS (random task sets, all algorithms on the SAME sets)")
    print(" Recommended for the report: 4 cores and 8 cores, 7 levels x 50 sets = 350 sets each.")
    cores = ask_int("Cores", 2, MAX_CORES, s.cores if s.cores in (4, 8) else 4)
    cluster = ask_int("Cluster size", 1, cores, min(2, cores))
    n = ask_int("Real-time tasks per set", cores + 1, MAX_TASKS, max(cores + 2, 8))
    levels_txt = ask("Utilization levels U/m (comma separated)", "0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    try:
        levels = [float(x) for x in levels_txt.split(",") if x.strip()]
        assert levels and all(0 < x <= 1.0 for x in levels)
    except (ValueError, AssertionError):
        print("  ! Use numbers between 0 and 1, e.g. 0.5,0.7,0.9")
        return
    per = ask_int("Task sets per level", 1, 500, 50)
    k = ask_choice("EDF mode", {"1": "Plain EDF", "2": "Criticality-first EDF", "3": "Both (compare)"}, "3")
    modes = {"1": (False,), "2": (True,), "3": (False, True)}[k]
    sp = ask_int("% sporadic tasks", 0, 100, 0)
    na = ask_int("Aperiodic jobs per set", 0, 20, 0)
    seed = ask_int("Seed", 0, 10 ** 6, 42)
    total = len(levels) * per
    print(f" Running {total} task sets x {len(ALGORITHMS)} algorithms x {len(modes)} mode(s) ...")
    t0 = time.time()

    def progress(f):
        sys.stdout.write(f"\r   {f * 100:5.1f} %")
        sys.stdout.flush()
    s.batch = run_batch(n, cores, levels, per, seed, cluster, s.miss_policy, modes=modes, progress=progress,
                        sporadic_frac=sp / 100, n_aperiodic=na, gap_pct=s.gap_pct)
    s.batch["settings"] = pd.DataFrame([{"Cores": cores, "Cluster size": cluster, "Tasks per set": n,
                                         "Levels": levels_txt, "Sets per level": per, "Sporadic %": sp,
                                         "Aperiodic per set": na, "Seed": seed, "Miss policy": s.miss_policy}])
    print(f"\n Done in {time.time() - t0:.1f} s.")
    header("BATCH SUMMARY")
    print(s.batch["summary"].set_index(["Algorithm", "EDF mode"]).T.to_string())
    header("SCHEDULABILITY RATIO BY UTILIZATION LEVEL")
    piv = s.batch["by_level"].pivot(index="Target U/m", columns="Series",
                                    values="Schedulability ratio (simulation)")
    print(piv.to_string())
    print("\n Export with menu 9 (CSV + Excel).")


# =====================================================================
# 9  Export
# =====================================================================
def export(s: Session):
    header("9. EXPORT RESULTS")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = ask("Folder to save into", os.path.join("results", stamp))
    os.makedirs(folder, exist_ok=True)
    saved = []
    sheets = {"Task configuration": s.tasks_df}

    def save(name, df):
        path = os.path.join(folder, f"{name}.csv")
        df.to_csv(path, index=False)
        saved.append(path)
    save("task_configuration", s.tasks_df)
    if s.results:
        res = list(s.results.values())
        tables = {
            "algorithm_comparison": comparison_table(s.results),
            "criticality_results": pd.concat([criticality_metrics(r) for r in res], ignore_index=True),
            "task_results": pd.concat([task_metrics(r).assign(Algorithm=r.algorithm) for r in res],
                                      ignore_index=True),
            "job_results": pd.concat([job_table(r) for r in res], ignore_index=True),
            "aperiodic_results": pd.concat([aperiodic_table(r) for r in res], ignore_index=True),
            "high_check": pd.concat([high_check_table(r) for r in res], ignore_index=True),
            "scheduling_timeline": pd.concat([timeline_segments(r) for r in res], ignore_index=True),
        }
        for name, df in tables.items():
            save(name, df)
            sheets[name.replace("_", " ").capitalize()] = df
    if s.mode_table is not None:
        save("plain_vs_criticality_first", s.mode_table)
        sheets["Plain vs crit-first"] = s.mode_table
    if s.batch:
        for name in ("summary", "by_level", "results", "tasksets", "settings"):
            save(f"batch_{name}", s.batch[name])
            sheets[f"Batch {name}"] = s.batch[name]
    path = os.path.join(folder, "rtos_scheduling_results.xlsx")
    with open(path, "wb") as f:
        f.write(to_excel_bytes(sheets))
    saved.append(path)
    print(" Saved:")
    for p in saved:
        print("   " + os.path.abspath(p))
    if not s.results:
        print(" (Run menu 7 first to also export results, timelines and comparisons.)")


# =====================================================================
# Main menu
# =====================================================================
MENU = [
    ("1", "Number of cores"),
    ("2", "Cluster size"),
    ("3", "Enter tasks  (A manual / B CSV file / C random / D example)"),
    ("4", "Simulation settings  (horizon, miss policy, EDF mode, sporadic gap)"),
    ("5", "Show tasks, check values, edit / add / delete"),
    ("6", "Choose algorithms  (P / G / C / P+G / P+C / G+C / all three)"),
    ("7", "Run: results + Gantt charts"),
    ("8", "Batch experiments  (random task sets)"),
    ("9", "Export results  (CSV + Excel)"),
    ("10", "Guided run  (all steps in order, like at the start)"),
    ("0", "Exit"),
]


def main():
    s = Session()
    actions = {"1": set_cores, "2": set_cluster, "3": task_entry, "4": settings, "5": show_tasks,
               "6": choose_algorithms, "7": results_menu, "8": batch_experiments, "9": export,
               "10": guided_flow}
    print("=" * 78)
    print("  MULTI-CORE RTOS SCHEDULING SIMULATOR  -  Partitioned / Global / Clustered EDF")
    print("=" * 78)
    try:
        guided_flow(s)
    except GoMenu:
        pass
    except QuitApp:
        print("\nBye.")
        return
    while True:
        try:
            tasks, errors, _ = s.tasks()
            n_tasks = len(s.tasks_df)
            print(f"\n{LINE}\n MAIN MENU   cores={s.cores}  cluster={s.cluster}  tasks={n_tasks}"
                  f"{' (ERRORS)' if errors else ''}  algorithms={ALGORITHM_CHOICES[s.algo_choice][0]}"
                  f"  mode={s.mode_text()}\n{LINE}")
            for k, label in MENU:
                print(f"  {k}. {label}")
            print("  (At any question: X = main menu, Q = quit)")
            choice = ask("Choose").strip()
            if choice == "0":
                raise QuitApp()
            if choice in actions:
                actions[choice](s)
                if choice in ("7", "8", "9"):
                    pause()
            else:
                print("  ! Choose 0-10.")
        except GoMenu:
            continue
        except QuitApp:
            print("\nBye.")
            return
        except KeyboardInterrupt:
            print("\n(Interrupted - back to main menu. Type Q to quit.)")


if __name__ == "__main__":
    main()
