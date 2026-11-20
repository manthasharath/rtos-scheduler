"""
compare_with_simulator.py
-------------------------
Compares the schedule executed on the Raspberry Pi (rtos_hw) with the
Python simulator, tick by tick and core by core.

Usage:
    python compare_with_simulator.py <folder with the hw_*.csv files>
    python compare_with_simulator.py <folder with many run folders>   (e.g. from collect_data.sh)

The folder must contain hw_run_info.csv, hw_tasks.csv and hw_trace_P/G/C.csv
(copied from the Pi). The simulator is re-run with exactly the same task set,
cores, cluster size, horizon, miss policy, EDF mode, sporadic gap and seed,
then the script reports:
  * how many (tick, core) slots match  (MATCH = identical schedule),
  * the first differences, if any,
  * simulator results next to the hardware measurements,
and writes hw_vs_simulator.csv into the same folder.
The website's "Raspberry Pi results" page uses the same function.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import pandas as pd

from algorithms import run_algorithm
from metrics import compute_metrics
from task_model import read_task_csv, validate_task_table

ALG = {"P": "Partitioned EDF", "G": "Global EDF", "C": "Clustered EDF"}


def simulated_labels(result) -> Dict:
    jm = result.job_map()
    return {(t, core): ("Idle" if key is None else jm[key].label)
            for core, row in result.timeline.items() for t, key in enumerate(row)}


def run_settings(info: pd.Series) -> dict:
    def get(name, default):
        return info[name] if name in info.index and not pd.isna(info[name]) else default
    return {
        "cores": int(info["Cores"]), "cluster": int(info["ClusterSize"]), "horizon": int(info["Horizon"]),
        "policy": str(info["MissPolicy"]), "crit": bool(int(get("CritMode", 0))),
        "gap": int(get("GapPct", 50)), "seed": int(get("Seed", 1)), "tick_ms": int(get("Tick_ms", 0)),
    }


def compare_frames(info: pd.Series, tasks_df: pd.DataFrame, traces: Dict[str, pd.DataFrame],
                   summary: Optional[pd.DataFrame] = None, verbose: bool = False):
    """
    info: first row of hw_run_info.csv; tasks_df: hw_tasks.csv (read with read_task_csv);
    traces: {"P": df, "G": df, "C": df} from hw_trace_*.csv; summary: hw_summary.csv or None.
    Returns (comparison DataFrame, {algorithm: simulated result}, {algorithm: list of differences}).
    """
    tasks, errors, _ = validate_task_table(tasks_df)
    if errors:
        raise ValueError("hw_tasks.csv is invalid: " + "; ".join(errors))
    s = run_settings(info)
    rows, sims, diffs_all = [], {}, {}
    for code, name in ALG.items():
        if code not in traces:
            continue
        hw = traces[code]
        sim = run_algorithm(name, tasks, s["cores"], s["horizon"], s["cluster"], s["policy"], s["crit"],
                            s["gap"], s["seed"])
        sims[name] = sim
        expected = simulated_labels(sim)
        diffs = [(int(r.Tick), int(r.Core), str(r.Job), expected.get((int(r.Tick), int(r.Core)), "?"))
                 for r in hw.itertuples() if expected.get((int(r.Tick), int(r.Core))) != str(r.Job)]
        diffs_all[name] = diffs
        sm = compute_metrics(sim)
        busy = hw[hw["Job"] != "Idle"]
        total = len(hw)
        if verbose:
            print(f"{name}: {total - len(diffs)} of {total} slots identical -> {'MATCH' if not diffs else 'DIFFERENT'}")
            for t, c, got, exp in diffs[:5]:
                print(f"   tick {t}, core {c}: hardware ran {got}, simulator expected {exp}")
        row = {
            "Algorithm": name, "EDF mode": sim.mode_name,
            "Slots compared": total, "Slots identical": total - len(diffs),
            "Schedule match": "MATCH" if not diffs else "DIFFERENT",
            "Jobs on wrong physical CPU": int((busy["ActualCPU"] != busy["Core"]).sum()),
            "Sim deadline misses": sm["Deadline misses"], "Sim High misses": sm["High misses"],
            "Sim Medium misses": sm["Medium misses"], "Sim Low misses": sm["Low misses"],
            "Sim High guaranteed": sm["High tasks guaranteed"],
            "Sim avg response": sm["Average response time"], "Sim max response": sm["Maximum response time"],
            "Sim preemptions": sm["Preemptions"], "Sim migrations": sm["Migrations"],
            "Sim aperiodic avg response": sm["Aperiodic avg response"],
        }
        if summary is not None and len(summary):
            h = summary[summary["Algorithm"] == name]
            if len(h):
                h = h.iloc[-1]
                def g(col, cast=float):
                    return cast(h[col]) if col in h.index and not pd.isna(h[col]) else None
                row.update({
                    "HW deadline misses": g("DeadlineMisses", int), "HW High misses": g("HighMisses", int),
                    "HW Medium misses": g("MediumMisses", int), "HW Low misses": g("LowMisses", int),
                    "HW High guaranteed": h["HighGuaranteed"] if "HighGuaranteed" in h.index else None,
                    "HW avg response": g("AvgResponse"), "HW max response": g("MaxResponse", int),
                    "HW preemptions": g("Preemptions", int), "HW migrations": g("Migrations", int),
                    "HW aperiodic avg response": g("AperiodicAvgResponse"),
                    "HW decision avg (us)": g("DecisionAvg_us"), "HW decision max (us)": g("DecisionMax_us"),
                    "HW dispatch avg (us)": g("DispatchAvg_us"), "HW dispatch max (us)": g("DispatchMax_us"),
                    "HW tick overruns": g("TickOverruns", int), "HW wrong CPU": g("WrongCPU", int),
                    "HW real-time priority": h["RealTimePriority"] if "RealTimePriority" in h.index else None,
                })
        rows.append(row)
    if not rows:
        raise ValueError("No hw_trace_P/G/C.csv files found.")
    return pd.DataFrame(rows), sims, diffs_all


def compare_folder(folder: str) -> pd.DataFrame:
    info = pd.read_csv(os.path.join(folder, "hw_run_info.csv")).iloc[0]
    tasks_df = read_task_csv(os.path.join(folder, "hw_tasks.csv"))
    traces = {c: pd.read_csv(os.path.join(folder, f"hw_trace_{c}.csv"))
              for c in ALG if os.path.exists(os.path.join(folder, f"hw_trace_{c}.csv"))}
    sp = os.path.join(folder, "hw_summary.csv")
    summary = pd.read_csv(sp) if os.path.exists(sp) and os.path.getsize(sp) > 0 else None
    s = run_settings(info)
    print(f"Task set: {len(tasks_df)} tasks | cores {s['cores']} | cluster {s['cluster']} | horizon {s['horizon']} "
          f"| on miss: {s['policy']} | {'criticality-first EDF' if s['crit'] else 'plain EDF'} "
          f"| sporadic gap {s['gap']} % seed {s['seed']} | tick {s['tick_ms']} ms\n")
    df, _, _ = compare_frames(info, tasks_df, traces, summary, verbose=True)
    print("\n" + df.set_index("Algorithm").T.to_string())
    out = os.path.join(folder, "hw_vs_simulator.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")
    return df


def compare_tree(root: str) -> pd.DataFrame:
    """Compare every sub-folder of `root` that holds a Pi run (e.g. the folder made by collect_data.sh)."""
    frames = []
    for dirpath, _, files in sorted(os.walk(root)):
        if "hw_run_info.csv" in files:
            print("=" * 78 + f"\n{os.path.relpath(dirpath, root)}\n" + "=" * 78)
            try:
                df = compare_folder(dirpath)
                df.insert(0, "Run", os.path.relpath(dirpath, root))
                frames.append(df)
            except (ValueError, FileNotFoundError) as exc:
                print(f"   skipped: {exc}")
    if not frames:
        raise ValueError(f"No Pi runs (hw_run_info.csv) found under {root}")
    allr = pd.concat(frames, ignore_index=True)
    out = os.path.join(root, "all_hw_vs_simulator.csv")
    allr.to_csv(out, index=False)
    n_match = int((allr["Schedule match"] == "MATCH").sum())
    print(f"\nSUMMARY: {n_match} of {len(allr)} algorithm runs MATCH the simulator.  Saved {out}")
    return allr


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    try:
        if os.path.exists(os.path.join(target, "hw_run_info.csv")):
            compare_folder(target)
        else:
            compare_tree(target)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc))
