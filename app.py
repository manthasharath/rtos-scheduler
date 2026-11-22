"""
app.py - Website (Streamlit) of the multi-core RTOS scheduling simulator.
Run with:   python -m streamlit run app.py
"""

from __future__ import annotations

import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from algorithms import ALGORITHM_CHOICES, ALGORITHMS, SHORT, run_algorithm
from clustered_edf import make_clusters
from compare_with_simulator import compare_frames, run_settings
from examples import DEFAULT_EXAMPLE, EXAMPLES
from experiments import run_batch
from export_results import to_csv_bytes, to_excel_bytes
from metrics import (aperiodic_table, comparison_table, compute_metrics, criticality_metrics, high_check_table,
                     job_table, mode_comparison, schedule_events, task_metrics, timeline_segments)
from task_generator import generate_taskset
from task_model import (MAX_CORES, MAX_TASKS, TASK_COLUMNS, clean_task_table, display_table, fill_priorities,
                        hyperperiod, read_task_csv,
                        tasks_to_dataframe, total_density, total_utilization, validate_task_table)
from visualization import (ALGO_COLORS, animated_gantt, criticality_bar, cumulative_miss_chart, gantt_chart,
                           gantt_from_trace, hw_overhead_chart, level_line_chart, metric_bar, per_core_util_chart,
                           per_task_response_chart)

st.set_page_config(page_title="Multi-Core RTOS Scheduler", page_icon="⏱️", layout="wide")

MAX_HORIZON = 5000
PAGES = ["Home", "1 · Setup", "2 · Review & edit", "3 · Run", "4 · Results", "5 · Compare", "6 · Batch experiments",
         "7 · Export", "Raspberry Pi results"]
CHART_CFG = {"displaylogo": False, "toImageButtonOptions": {"scale": 2}}

# ------------------------------------------------------------------ style
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
<style>
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }
.hero { border-radius: 18px; padding: 28px 30px 24px; margin-bottom: 18px; color: #f4f7fb;
  background:
    linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px) 0 0 / 40px 40px,
    linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px) 0 0 / 40px 40px,
    linear-gradient(120deg, #0d1b2a 0%, #1b3a5c 55%, #2a78d6 100%); }
.hero h1 { color: #fff; margin: 0 0 6px 0; font-size: 2.0rem; }
.hero p { color: #d5e2f2; margin: 0; font-size: 1.02rem; }
.hero .lanes { margin-top: 18px; display: grid; gap: 6px; }
.hero .lane { display: flex; align-items: center; gap: 10px; font: 500 12px 'JetBrains Mono', monospace; color: #b8cbe3; }
.hero .lane span.bar { height: 14px; border-radius: 4px; display: inline-block; }
.card { border: 1px solid rgba(128,128,128,0.25); border-radius: 14px; padding: 16px 18px; height: 100%;
  background: rgba(128,128,128,0.04); }
.card h4 { margin: 0 0 6px 0; font-family: 'Space Grotesk', sans-serif; }
.card .tag { font: 500 11px 'JetBrains Mono', monospace; text-transform: uppercase; letter-spacing: .06em; opacity: .7; }
.kpi { border: 1px solid rgba(128,128,128,0.25); border-left: 5px solid var(--c, #2a78d6); border-radius: 12px;
  padding: 12px 14px; background: rgba(128,128,128,0.04); }
.kpi .name { font: 600 13px 'Space Grotesk', sans-serif; margin-bottom: 6px; }
.kpi .row { display: flex; justify-content: space-between; font-size: 13px; padding: 2px 0;
  border-bottom: 1px dashed rgba(128,128,128,0.18); }
.kpi .row b { font-family: 'JetBrains Mono', monospace; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font: 600 12px 'JetBrains Mono', monospace; }
.ok { background: rgba(0,131,0,0.14); color: #0b7a2a; } .bad { background: rgba(198,40,40,0.14); color: #c62828; }
.neutral { background: rgba(128,128,128,0.16); }
.flow { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; font: 500 12px 'JetBrains Mono', monospace; }
.flow span { border: 1px solid rgba(128,128,128,0.35); border-radius: 8px; padding: 4px 8px; }
section[data-testid="stSidebar"] .stRadio label p { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ state
def init_state():
    ss = st.session_state
    if "tasks_df" not in ss:
        ex = EXAMPLES[DEFAULT_EXAMPLE]
        ss.tasks_df = ex["df"].copy()
        ss.source = f"Example: {DEFAULT_EXAMPLE}"
        ss.cfg = {"cores": ex["cores"], "cluster": ex["cluster"], "horizon_mode": "One hyperperiod",
                  "custom_horizon": 100, "miss_policy": "continue", "crit_mode": False, "gap_pct": 50, "seed": 1,
                  "algo_choice": "7"}
    ss.setdefault("editor_version", 0)
    ss.setdefault("confirmed", False)
    ss.setdefault("results", {})
    ss.setdefault("results_sig", None)
    ss.setdefault("mode_table", None)
    ss.setdefault("batches", {})
    ss.setdefault("pi", None)


def set_tasks(df: pd.DataFrame, source: str, cores=None, cluster=None):
    ss = st.session_state
    ss.tasks_df = fill_priorities(clean_task_table(df)[0])
    ss.source = source
    if cores:
        ss.cfg["cores"] = cores
    if cluster:
        ss.cfg["cluster"] = min(cluster, ss.cfg["cores"])
    ss.editor_version += 1
    ss.confirmed = False


def current_config():
    ss = st.session_state
    cfg = ss.cfg
    tasks, errors, warnings = validate_task_table(ss.tasks_df)
    horizon = None
    if tasks:
        if cfg["horizon_mode"] == "One hyperperiod":
            hp = hyperperiod(tasks)
            horizon = min(hp, MAX_HORIZON)
            if hp > MAX_HORIZON:
                warnings.append(f"Hyperperiod {hp} exceeds {MAX_HORIZON}; the horizon is capped at {MAX_HORIZON}.")
        else:
            horizon = int(cfg["custom_horizon"])
    if cfg["cluster"] > cfg["cores"]:
        errors.append(f"Cluster size ({cfg['cluster']}) cannot exceed the number of cores ({cfg['cores']}).")
    sig = (ss.tasks_df.to_json(), cfg["cores"], cfg["cluster"], horizon, cfg["miss_policy"], cfg["crit_mode"],
           cfg["gap_pct"], cfg["seed"])
    return dict(tasks=tasks, horizon=horizon, errors=errors, warnings=warnings, sig=sig, **cfg)


def fresh_results(conf):
    ss = st.session_state
    return ss.results if ss.results and ss.results_sig == conf["sig"] else {}


def run_now(conf, names):
    ss = st.session_state
    ss.results = {n: run_algorithm(n, conf["tasks"], conf["cores"], conf["horizon"], conf["cluster"],
                                   conf["miss_policy"], conf["crit_mode"], conf["gap_pct"], conf["seed"])
                  for n in names}
    ss.results_sig = conf["sig"]
    ss.mode_table = None


def show_errors(conf) -> bool:
    for w in conf["warnings"]:
        st.warning(w)
    if conf["errors"]:
        st.error("Fix the task table on **2 · Review & edit** first:\n\n- " + "\n- ".join(conf["errors"]))
        return True
    return False


def badge(text, kind):
    return f"<span class='badge {kind}'>{text}</span>"


def need_results(conf):
    res = fresh_results(conf)
    if not res:
        st.info("No results for the current settings yet. Go to **3 · Run** and press **Run simulation**.")
    return res


def plot(fig, key=None):
    st.plotly_chart(fig, width="stretch", config=CHART_CFG, key=key)


# ------------------------------------------------------------------ sidebar
init_state()
with st.sidebar:
    st.markdown("### ⏱️ Multi-Core RTOS\n**Partitioned · Global · Clustered EDF**")
    page = st.radio("Go to", PAGES, label_visibility="collapsed", key="page")
    st.divider()
    _c = current_config()
    if _c["tasks"]:
        _U = total_utilization(_c["tasks"])
        st.caption("Current set-up")
        st.markdown(
            f"**{len(_c['tasks'])}** tasks · **{_c['cores']}** cores · cluster **{_c['cluster']}**  \n"
            f"Load U = **{_U:.2f}** ({_U / _c['cores'] * 100:.0f} % of cores)  \n"
            f"Horizon **{_c['horizon']}** · {'Criticality-first' if _c['crit_mode'] else 'Plain'} EDF  \n"
            f"Algorithms: **{ALGORITHM_CHOICES[_c['algo_choice']][0]}**")
    else:
        st.markdown(":red[Task table has errors]")
    st.caption(st.session_state.source)


# ------------------------------------------------------------------ pages
def page_home():
    lanes = "".join(
        f"<div class='lane'>Core {i} " + "".join(
            f"<span class='bar' style='width:{w}px;background:{c};margin-left:{g}px'></span>" for w, c, g in row)
        + "</div>"
        for i, row in enumerate([[(70, "#2a78d6", 0), (40, "#eb6834", 4), (90, "#1baf7a", 4)],
                                 [(30, "#eda100", 0), (110, "#e87ba4", 4), (40, "#2a78d6", 20)],
                                 [(120, "#6250d6", 0), (50, "#7d8794", 4)],
                                 [(50, "#1baf7a", 30), (60, "#eb6834", 4), (50, "#eda100", 4)]]))
    st.markdown(f"""<div class='hero'><h1>Partitioned vs Global vs Clustered EDF</h1>
<p>Performance comparison of multi-core real-time scheduling, simulated on the PC and executed on a Raspberry Pi.</p>
<div class='lanes'>{lanes}</div></div>""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown("<div class='card'><div class='tag'>no migration</div><h4>Partitioned EDF</h4>Each task is fixed to "
                "<b>one core</b> by First Fit Decreasing (density ≤ 1 per core). Each core runs its own EDF.</div>",
                unsafe_allow_html=True)
    c2.markdown("<div class='card'><div class='tag'>migration anywhere</div><h4>Global EDF</h4>One ready queue: the "
                "<b>m earliest-deadline jobs</b> run on the m cores. Jobs may move between any cores.</div>",
                unsafe_allow_html=True)
    c3.markdown("<div class='card'><div class='tag'>migration inside a cluster</div><h4>Clustered EDF</h4>Cores are "
                "split into clusters of <b>k</b>. Tasks are fixed to a cluster (FFD + GFB test); global EDF inside "
                "each cluster.</div>", unsafe_allow_html=True)
    st.markdown("#### How to use")
    st.markdown("<div class='flow'><span>1 Setup</span>→<span>2 Review &amp; edit</span>→<span>3 Run</span>→"
                "<span>4 Results</span>→<span>5 Compare</span>→<span>6 Batch</span>→<span>7 Export</span>→"
                "<span>Raspberry Pi results</span></div>", unsafe_allow_html=True)
    st.markdown("#### Two EDF modes")
    a, b = st.columns(2)
    a.markdown("**Plain EDF** (default) — earliest absolute deadline first. Criticality only breaks ties.")
    b.markdown("**Criticality-first EDF** — High jobs always run before Medium and Low jobs; EDF inside each group. "
               "In this mode Partitioned and Clustered place High tasks first (DCDU order).")
    st.markdown("**High tasks guaranteed** — a design-time check on the High tasks alone (density test per core for "
                "Partitioned, GFB test per cluster or on all cores). If it says **YES**, criticality-first EDF meets "
                "every High deadline, whatever the Medium and Low tasks do. Plain EDF gives no such guarantee.")
    with st.expander("Task model and assumptions"):
        st.markdown(
            "- One table for all tasks. **Type P** periodic (release every T), **S** sporadic (minimum gap T plus a "
            "random extra gap, same seed = same arrivals on PC and Pi), **A** aperiodic (one job with an arrival "
            "time, no deadline).\n"
            "- Aperiodic jobs are served in the **background**: only on cores left idle by the real-time jobs, "
            "oldest first. They never delay a periodic or sporadic job.\n"
            "- Deadlines C ≤ D ≤ T. Equal deadlines: higher criticality, then the **priority number** (1 = highest; "
            "empty = automatic, shorter period first), then task order.\n"
            "- Integer time, preemptive, re-evaluated every time unit. Preemptions, migrations and context "
            "switches are **counted** but cost no time in the simulator; the Raspberry Pi **measures** the real "
            "overheads.\n"
            "- A deadline miss either lets the job continue late or aborts it.")
    st.info("**Simulation vs proof.** \"No deadline misses\" is what happened in the simulated horizon. The tests shown "
            "(FFD density test, GFB test) are sufficient conditions. For Partitioned EDF with D = T one hyperperiod is "
            "conclusive; for Global and Clustered EDF a miss-free run is not a proof.")


def _int(v, default):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


TYPE_LABEL = {"P": "Periodic (P)", "S": "Sporadic (S)", "A": "Aperiodic (A)"}


def task_fields(key: str, row, default_name: str) -> dict:
    """Entry boxes for one task. Type first; only the boxes that type uses are shown."""
    kind0 = str(row["Type"]).upper() if row is not None else "P"
    kind = st.radio("Type", ["P", "S", "A"], index=["P", "S", "A"].index(kind0 if kind0 in "PSA" else "P"),
                    format_func=TYPE_LABEL.get, horizontal=True, key=f"{key}_type")
    k = f"{key}_{kind}"
    get = (lambda c, d: _int(row[c], d)) if row is not None else (lambda c, d: d)
    name = st.text_input("Name (no spaces or commas)", row["Name"] if row is not None else default_name,
                         key=f"{k}_name").strip()
    if kind == "A":
        st.caption("Aperiodic job: one job at its arrival time, served in the background on idle cores. "
                   "It has no period, deadline, priority or criticality.")
        a, b = st.columns(2)
        c = a.number_input("C = execution time", 1, 10000, get("C", 2), key=f"{k}_C")
        arr = b.number_input("Arrival time", 0, 100000, get("Arrival", 0), key=f"{k}_arr")
        return {"Name": name, "Type": "A", "C": int(c), "T": None, "D": None, "Priority": None,
                "Criticality": None, "Arrival": int(arr)}
    st.caption("Periodic: a new job every T." if kind == "P" else
               "Sporadic: events at least T apart (plus a random extra gap set on 3 · Run).")
    a, b, c_, d, e = st.columns(5)
    c = a.number_input("C = execution time", 1, 10000, get("C", 1), key=f"{k}_C")
    t = b.number_input("T = period" if kind == "P" else "T = minimum gap", 1, 100000, get("T", 10), key=f"{k}_T")
    dl = c_.number_input("D = deadline (C ≤ D ≤ T)", 1, 100000, get("D", int(t)), key=f"{k}_D")
    pr = d.number_input("Priority (0 = automatic)", 0, 999, get("Priority", 0) if row is not None else 0,
                        key=f"{k}_P", help="1 = highest. Only breaks ties of equal deadline and criticality.")
    crit0 = str(row["Criticality"]) if row is not None and str(row["Criticality"]) in ("High", "Medium", "Low") else "Low"
    cr = e.selectbox("Criticality", ["High", "Medium", "Low"], index=["High", "Medium", "Low"].index(crit0),
                     key=f"{k}_crit")
    return {"Name": name, "Type": kind, "C": int(c), "T": int(t), "D": int(dl), "Priority": int(pr) or None,
            "Criticality": cr, "Arrival": None}


def try_save(new_df: pd.DataFrame, old_df: pd.DataFrame, source: str) -> bool:
    """Save only if the change adds no new error."""
    _, before, _ = validate_task_table(old_df)
    _, after, _ = validate_task_table(new_df)
    added = [e for e in after if e not in before]
    if added:
        st.error("Not saved:\n\n- " + "\n- ".join(added))
        return False
    set_tasks(new_df, source)
    return True


def task_editor(key_prefix: str):
    """Task table (view only) + a form to add, edit or delete tasks."""
    ss = st.session_state
    df = fill_priorities(ss.tasks_df).reset_index(drop=True)
    st.dataframe(display_table(df), width="stretch")
    st.caption("— = not used by that task type (aperiodic: only C and Arrival; periodic/sporadic: no Arrival). "
               "Priority is filled automatically by period when left empty.")
    src = ss.source if "edited" in ss.source else ss.source + " (edited)"
    v = ss.editor_version
    action = st.radio("Change the table", ["➕ Add a task", "✏️ Edit a task", "🗑️ Delete tasks"], horizontal=True,
                      key=f"{key_prefix}_act")
    with st.container(border=True):
        if action.startswith("➕"):
            if len(df) >= MAX_TASKS:
                st.warning(f"At most {MAX_TASKS} tasks.")
            else:
                row = task_fields(f"{key_prefix}_add{v}", None, f"T{len(df) + 1}")
                if st.button("Add task", type="primary", key=f"{key_prefix}_addbtn"):
                    new = pd.concat([df, pd.DataFrame([row], columns=TASK_COLUMNS)], ignore_index=True)
                    if try_save(new, df, src):
                        st.rerun()
        elif action.startswith("✏️"):
            if not len(df):
                st.info("The table is empty.")
            else:
                i = st.selectbox("Task to edit", range(len(df)), format_func=lambda j: f"{j + 1}. {df.at[j, 'Name']}",
                                 key=f"{key_prefix}_pick")
                row = task_fields(f"{key_prefix}_ed{v}_{i}", df.loc[i], str(df.at[i, "Name"]))
                if st.button("Update task", type="primary", key=f"{key_prefix}_updbtn"):
                    new = df.copy().astype(object)
                    new.loc[i] = [row[c] for c in TASK_COLUMNS]
                    if try_save(new, df, src):
                        st.rerun()
        else:
            drop = st.multiselect("Tasks to delete", range(len(df)), format_func=lambda j: f"{j + 1}. {df.at[j, 'Name']}",
                                  key=f"{key_prefix}_del{v}")
            if st.button("Delete", type="primary", disabled=not drop, key=f"{key_prefix}_delbtn"):
                set_tasks(df.drop(index=drop).reset_index(drop=True), src)
                st.rerun()
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("↺ Automatic priorities", key=f"{key_prefix}_auto",
                 help="Fill every P/S priority by period (shorter period = 1)"):
        set_tasks(fill_priorities(df, overwrite=True), ss.source)
        st.rerun()
    if c2.button("🧹 Clear all tasks", key=f"{key_prefix}_clear"):
        set_tasks(pd.DataFrame(columns=TASK_COLUMNS), "Manual entry")
        st.rerun()


def page_setup():
    ss = st.session_state
    cfg = ss.cfg
    st.title("1 · Setup")
    st.subheader("Processors")
    c1, c2, c3 = st.columns([1, 1, 2])
    cores = c1.number_input("Number of cores", 1, MAX_CORES, cfg["cores"], 1,
                            help=f"Up to {MAX_CORES}. The Raspberry Pi has 4; our comparisons use 4 and 8.")
    cluster = c2.number_input("Cluster size k (Clustered EDF)", 1, int(cores), min(cfg["cluster"], int(cores)), 1,
                              help="k = 1 behaves like Partitioned, k = cores like Global.")
    if (cores, cluster) != (cfg["cores"], cfg["cluster"]):
        cfg["cores"], cfg["cluster"] = int(cores), int(cluster)
        st.rerun()
    c3.markdown("**Clusters**  \n" + " · ".join(f"C{i + 1} = cores {c[0]}–{c[-1]}"
                                               for i, c in enumerate(make_clusters(cfg["cores"], cfg["cluster"]))))

    st.subheader("Tasks")
    t1, t2, t3, t4 = st.tabs(["⭐ Built-in example", "✏️ Enter tasks", "📄 CSV file", "🎲 Random (UUniFast)"])
    with t1:
        name = st.selectbox("Example", list(EXAMPLES), index=list(EXAMPLES).index(DEFAULT_EXAMPLE))
        ex = EXAMPLES[name]
        st.caption(f"{ex['cores']} cores · cluster {ex['cluster']}")
        st.markdown(ex["description"])
        st.dataframe(display_table(ex["df"]), width="stretch")
        if st.button("Load this example", type="primary"):
            set_tasks(ex["df"], f"Example: {name}", ex["cores"], ex["cluster"])
            st.rerun()
    with t2:
        st.caption("Choose the Type first: only the boxes for that type appear. "
                   "Use 🧹 Clear all tasks to start from an empty table.")
        task_editor("setup_editor")
    with t3:
        st.markdown("Columns: `Name,Type,C,T,D,Priority,Criticality,Arrival`  "
                    "(same file works in the terminal version and on the Raspberry Pi; old 6-column files also work).")
        st.code("EngineCtl,P,2,5,5,,High,\nBrakePedal,S,1,10,5,,High,\nDiagScan,A,6,,,,,12", language="text")
        up = st.file_uploader("Upload a task CSV", type=["csv"])
        if up is not None:
            try:
                df, notes = clean_task_table(read_task_csv(up))
                st.dataframe(display_table(df), width="stretch")
                for n in notes:
                    st.info(n)
                if st.button("Use this file", type="primary"):
                    set_tasks(df, f"CSV: {up.name}")
                    st.rerun()
            except Exception as exc:          # noqa: BLE001 - show any read problem
                st.error(f"Could not read the CSV: {exc}")
        st.download_button("Download the current table as CSV", to_csv_bytes(ss.tasks_df), "task_configuration.csv")
    with t4:
        a, b, c = st.columns(3)
        n = a.number_input("Real-time tasks (P + S)", 2, MAX_TASKS, 8)
        um = b.slider("Load U/m (fraction of all cores)", 0.1, 1.2, 0.7, 0.05)
        seed = c.number_input("Seed", 0, 10 ** 6, 42)
        a2, b2 = st.columns(2)
        sp = a2.slider("% sporadic", 0, 100, 0, 10)
        na = b2.number_input("Aperiodic jobs", 0, 20, 0)
        if st.button("Generate", type="primary"):
            u = min(um * cfg["cores"], float(n))
            ts = generate_taskset(int(n), u, np.random.default_rng(int(seed)), sporadic_frac=sp / 100,
                                  n_aperiodic=int(na))
            set_tasks(tasks_to_dataframe(ts), f"Random (U={u:.2f}, seed={seed})")
            st.rerun()
    st.success(f"Loaded: **{ss.source}** → check it on **2 · Review & edit**.")


def analysis_cards(conf):
    tasks = conf["tasks"]
    rt = [t for t in tasks if t.is_realtime]
    U = total_utilization(tasks)
    hi = [t for t in rt if t.criticality == "High"]
    c = st.columns(5)
    c[0].metric("Tasks P / S / A", f"{sum(t.kind == 'P' for t in rt)} / {sum(t.kind == 'S' for t in rt)} / "
                                   f"{len(tasks) - len(rt)}")
    c[1].metric("Load U = ΣC/T", f"{U:.2f}", f"{U / conf['cores'] * 100:.0f} % of {conf['cores']} cores",
                delta_color="off")
    c[2].metric("Density Σ C/min(D,T)", f"{total_density(tasks):.2f}")
    c[3].metric("Hyperperiod", hyperperiod(tasks))
    c[4].metric("High tasks / load", f"{len(hi)} / {sum(t.utilization for t in hi):.2f}")
    if U > conf["cores"]:
        st.warning("U is larger than the number of cores: the system is overloaded and deadline misses are certain.")


def page_review():
    ss = st.session_state
    st.title("2 · Review & edit")
    st.caption(f"Source: {ss.source}")
    task_editor("review_editor")
    conf = current_config()
    if show_errors(conf):
        return
    analysis_cards(conf)
    st.subheader("Design-time checks")
    rows = []
    for alg in ALGORITHMS:
        r = run_algorithm(alg, conf["tasks"], conf["cores"], 1, conf["cluster"], crit_mode=False)
        rc = run_algorithm(alg, conf["tasks"], conf["cores"], 1, conf["cluster"], crit_mode=True)
        rows.append({"Algorithm": alg, "Full set test (plain allocation)": "PASSED" if r.analytic_test["passed"]
                     else "not passed", "High tasks guaranteed (criticality-first)": "YES" if rc.high_check["passed"]
                     else "NO", "Detail": r.analytic_test["detail"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Tests are sufficient conditions. 'High tasks guaranteed' = the High tasks alone pass the density / "
               "GFB test on their core group, so criticality-first EDF meets every High deadline.")
    if st.button("✅ Values are correct — confirm", type="primary"):
        ss.confirmed = True
    if ss.confirmed:
        st.success("Confirmed. Next: **3 · Run**.")


def page_run():
    ss = st.session_state
    cfg = ss.cfg
    st.title("3 · Run")
    conf = current_config()
    if show_errors(conf):
        return
    labels = {k: v[0] for k, v in ALGORITHM_CHOICES.items()}
    choice = st.radio("Algorithms", list(labels), format_func=labels.get, horizontal=True,
                      index=list(labels).index(cfg["algo_choice"]))
    a, b = st.columns(2)
    mode = a.radio("EDF mode", ["Plain EDF", "Criticality-first EDF"], index=1 if cfg["crit_mode"] else 0,
                   help="Criticality-first: High jobs always before Medium/Low jobs; EDF inside each group.")
    policy = b.radio("On a deadline miss", ["continue", "abort"], index=0 if cfg["miss_policy"] == "continue" else 1,
                     format_func={"continue": "Continue (job runs late)", "abort": "Abort (job dropped)"}.get)
    c, d, e, f = st.columns(4)
    hmode = c.selectbox("Horizon", ["One hyperperiod", "Custom"],
                        index=0 if cfg["horizon_mode"] == "One hyperperiod" else 1)
    hval = d.number_input("Custom horizon", 1, MAX_HORIZON, cfg["custom_horizon"], disabled=hmode != "Custom")
    gap = e.number_input("Sporadic extra gap (max % of T)", 0, 200, cfg["gap_pct"])
    seed = f.number_input("Sporadic seed", 0, 10 ** 6, cfg["seed"])
    cfg.update(algo_choice=choice, crit_mode=mode != "Plain EDF", miss_policy=policy, horizon_mode=hmode,
               custom_horizon=int(hval), gap_pct=int(gap), seed=int(seed))
    conf = current_config()
    st.caption(f"{len(conf['tasks'])} tasks · {conf['cores']} cores · horizon {conf['horizon']} time units")
    if st.button("▶ Run simulation", type="primary"):
        run_now(conf, ALGORITHM_CHOICES[choice][1])
        st.success("Done. Open **4 · Results**.")
    res = fresh_results(conf)
    if res:
        kpi_cards(res)


def kpi_cards(res):
    cols = st.columns(len(res))
    for col, (name, r) in zip(cols, res.items()):
        m = compute_metrics(r)
        miss = m["Deadline misses"]
        hg = m["High tasks guaranteed"]
        rows = [("Deadline misses", badge(miss, "ok" if miss == 0 else "bad")),
                ("High · Med · Low", f"<b>{m['High misses']} · {m['Medium misses']} · {m['Low misses']}</b>"),
                ("High guaranteed", badge(hg, "ok" if hg == "YES" else ("bad" if hg == "NO" else "neutral"))),
                ("Avg / max response", f"<b>{m['Average response time']} / {m['Maximum response time']}</b>"),
                ("Preemptions · migrations", f"<b>{m['Preemptions']} · {m['Migrations']}</b>"),
                ("CPU utilization", f"<b>{m['CPU utilization (observed)'] * 100:.1f} %</b>")]
        if m["Aperiodic jobs"]:
            rows.append(("Aperiodic avg response", f"<b>{m['Aperiodic avg response']}</b>"))
        html = "".join(f"<div class='row'><span>{k}</span>{v}</div>" for k, v in rows)
        col.markdown(f"<div class='kpi' style='--c:{ALGO_COLORS[name]}'><div class='name'>{name}"
                     f"<br><span style='font-weight:400;font-size:12px;opacity:.7'>{r.mode_name}</span></div>{html}</div>",
                     unsafe_allow_html=True)


def page_results():
    st.title("4 · Results")
    conf = current_config()
    if show_errors(conf):
        return
    res = need_results(conf)
    if not res:
        return
    kpi_cards(res)
    st.write("")
    n_rt = len([t for t in conf["tasks"] if t.is_realtime])
    color_by = st.segmented_control("Colour bars by", ["task", "criticality"],
                                    default="task" if n_rt <= 8 else "criticality",
                                    help="With more than 8 tasks colours repeat, so colouring by criticality is clearer. "
                                         "Hover a bar for its job.") or "task"
    o1, o2, _ = st.columns([1, 1, 2])
    show_pre = o1.checkbox("Show preemptions ▼", True, help="▼ where a job is stopped, 'by X' = the job that took "
                           "its core; dotted line = waiting until it resumes on the same core")
    show_mig = o2.checkbox("Show migrations →", True, help="Arrow from the core where a job stopped to the core "
                           "where it continues")
    tabs = st.tabs(["Gantt (full)", "Gantt (animated)", "Events (why)", "Per task", "By criticality", "Aperiodic",
                    "Design-time tests", "Jobs"])
    with tabs[0]:
        H = max(r.horizon for r in res.values())
        if H > 40:
            lo, hi = st.slider("Zoom (time window)", 0, H, (0, H))
        else:
            lo, hi = 0, H
        for name, r in res.items():
            plot(gantt_chart(r, window=(lo, hi), color_by=color_by, show_preemptions=show_pre,
                             show_migrations=show_mig), key=f"g_{name}")
        st.caption("▼ preempted (label = by which job) · dotted line = waiting, then resumes on the same core · "
                   "arrow → = migration to another core · ✕ deadline miss · hatched grey = aperiodic job. "
                   "Hover any mark for the reason.")
    with tabs[1]:
        a, b = st.columns([2, 3])
        speed = a.slider("Speed (ms per frame)", 30, 800, 150, 10)
        b.markdown("Press **⏮ Play from start** inside the chart. All chosen algorithms build in sync; "
                   "drag the time slider to any moment.")
        plot(animated_gantt(res, frame_ms=speed, color_by=color_by, show_preemptions=show_pre,
                            show_migrations=show_mig), key="anim")
        st.caption("Above each chart: live counters and the latest preemption / migration with its reason.")
    with tabs[2]:
        a, b = st.columns([1, 2])
        alg = a.selectbox("Algorithm", list(res), key="ev_alg")
        kinds = b.multiselect("Show", ["Preemption", "Migration", "Resume", "Deadline miss", "Release", "Finish"],
                              default=["Preemption", "Migration", "Deadline miss"], key="ev_kinds")
        ev = schedule_events(res[alg])
        ev = ev[ev["Event"].isin(kinds)]
        st.dataframe(ev[["Time", "Event", "Text", "Job", "Core", "To core", "By", "Reason"]], hide_index=True,
                     width="stretch", height=420)
        st.caption("Every preemption with the job that caused it and the reason (earlier deadline, High "
                   "criticality first, or idle-time-only aperiodic job), and every migration from core to core.")
    with tabs[3]:
        for name, r in res.items():
            st.markdown(f"**{name}**")
            st.dataframe(task_metrics(r), hide_index=True, width="stretch")
    with tabs[4]:
        crit = pd.concat([criticality_metrics(r) for r in res.values()], ignore_index=True)
        st.dataframe(crit, hide_index=True, width="stretch")
        plot(criticality_bar(crit), key="critbar")
    with tabs[5]:
        ap = pd.concat([aperiodic_table(r) for r in res.values()], ignore_index=True)
        if len(ap):
            st.dataframe(ap, hide_index=True, width="stretch")
            st.caption("Background service: aperiodic jobs only use cores left idle by periodic/sporadic jobs.")
        else:
            st.info("This task set has no aperiodic (Type A) jobs.")
    with tabs[6]:
        for name, r in res.items():
            at = r.analytic_test
            st.markdown(f"**{name}** — {at['name']}: {badge('PASSED' if at['passed'] else 'NOT PASSED', 'ok' if at['passed'] else 'bad')}",
                        unsafe_allow_html=True)
            st.caption(at["detail"])
            st.dataframe(high_check_table(r), hide_index=True, width="stretch")
    with tabs[7]:
        st.dataframe(pd.concat([job_table(r) for r in res.values()], ignore_index=True), hide_index=True,
                     width="stretch")


def page_compare():
    ss = st.session_state
    st.title("5 · Compare")
    conf = current_config()
    if show_errors(conf):
        return
    res = need_results(conf)
    if not res:
        return
    table = comparison_table(res)
    show = ["Algorithm", "EDF mode", "Deadline misses", "High misses", "Medium misses", "Low misses",
            "Deadline miss ratio", "Average response time", "Maximum response time", "Preemptions", "Migrations",
            "Context switches", "CPU utilization (observed)", "Load imbalance (max-min)", "Allocation failures",
            "High tasks guaranteed", "Aperiodic avg response"]
    st.dataframe(table[[c for c in show if c in table]].set_index("Algorithm").T.astype(str), width="stretch")
    if len(res) > 1:
        c1, c2 = st.columns(2)
        with c1:
            plot(metric_bar(table, "Deadline misses"), key="b1")
            plot(metric_bar(table, "Migrations"), key="b3")
        with c2:
            plot(metric_bar(table, "Average response time", fmt=".2f"), key="b2")
            plot(metric_bar(table, "Preemptions"), key="b4")
        plot(per_core_util_chart(res), key="b5")
        plot(cumulative_miss_chart(res), key="b6")
        plot(per_task_response_chart({n: task_metrics(r) for n, r in res.items()}), key="b7")
    else:
        st.info("Choose two or three algorithms on **3 · Run** to see comparison charts.")
    st.subheader("Plain EDF vs criticality-first EDF (same task set)")
    if st.button("Run both modes"):
        ss.mode_table = mode_comparison(conf["tasks"], conf["cores"], conf["horizon"], conf["cluster"],
                                        conf["miss_policy"], conf["gap_pct"], conf["seed"], list(res))
    if ss.mode_table is not None:
        mt = ss.mode_table
        st.dataframe(mt, hide_index=True, width="stretch")
        long = mt.melt(id_vars=["Algorithm", "EDF mode"], value_vars=["High misses", "Medium misses", "Low misses"],
                       var_name="Criticality", value_name="Deadline misses")
        long["Criticality"] = long["Criticality"].str.replace(" misses", "")
        plot(criticality_bar(long), key="modebar")


DEFAULT_LEVELS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def page_batch():
    ss = st.session_state
    st.title("6 · Batch experiments")
    st.caption("Random task sets (UUniFast-Discard); every algorithm and mode runs on exactly the same sets. "
               "Recommended for the report: 4 cores and 8 cores, 7 levels × 50 sets = 350 sets each.")
    a, b, c, d = st.columns(4)
    cores = a.selectbox("Cores", [2, 4, 8, 10, 16, 20], index=1)
    cluster = b.selectbox("Cluster size", list(range(1, cores + 1)), index=1 if cores >= 2 else 0)
    n = c.number_input("Tasks per set", cores + 1, MAX_TASKS, max(cores + 4, 8))
    per = d.number_input("Sets per level", 1, 500, 50)
    levels = st.multiselect("Utilization levels U/m", [round(x, 2) for x in np.arange(0.1, 1.01, 0.05)],
                            default=DEFAULT_LEVELS)
    e, f, g, h = st.columns(4)
    modes = e.selectbox("EDF mode", ["Both (compare)", "Plain EDF", "Criticality-first EDF"])
    sp = f.slider("% sporadic", 0, 100, 0, 10)
    na = g.number_input("Aperiodic jobs per set", 0, 20, 0)
    seed = h.number_input("Seed", 0, 10 ** 6, 42)
    mode_tuple = {"Both (compare)": (False, True), "Plain EDF": (False,), "Criticality-first EDF": (True,)}[modes]
    total = len(levels) * per
    st.caption(f"{total} task sets × 3 algorithms × {len(mode_tuple)} mode(s) = "
               f"{total * 3 * len(mode_tuple)} simulations")
    if levels and int(n) < 2 * max(levels) * int(cores):
        st.warning(f"With {int(cores)} cores and U/m up to {max(levels)}, the total load is "
                   f"{max(levels) * int(cores):.1f}. {int(n)} tasks per set is probably too few for the generator "
                   f"(every task must stay at or below 100 % of a core); about {int(2 * max(levels) * int(cores)) + 1} "
                   f"or more is safer.")
    if st.button("▶ Run batch", type="primary", disabled=not levels):
        bar = st.progress(0.0, "Running…")
        try:
            out = run_batch(int(n), int(cores), sorted(levels), int(per), int(seed), int(cluster),
                            ss.cfg["miss_policy"], modes=mode_tuple, progress=lambda x: bar.progress(min(x, 1.0)),
                            sporadic_frac=sp / 100, n_aperiodic=int(na), gap_pct=ss.cfg["gap_pct"])
        except (RuntimeError, ValueError) as err:
            bar.empty()
            st.error(f"Could not generate the task sets: {err} Increase 'Tasks per set', remove the highest "
                     f"U/m levels, or use fewer cores.")
            return
        label = f"{cores} cores · k={cluster} · {n} tasks · {modes} · {datetime.now():%H:%M:%S}"
        out["settings"] = pd.DataFrame([{"Cores": cores, "Cluster size": cluster, "Tasks per set": n,
                                         "Levels": ", ".join(map(str, sorted(levels))), "Sets per level": per,
                                         "Modes": modes, "Sporadic %": sp, "Aperiodic per set": na, "Seed": seed,
                                         "Miss policy": ss.cfg["miss_policy"]}])
        ss.batches[label] = out
        ss.batch_view = label
        bar.empty()
    if not ss.batches:
        return
    label = st.selectbox("Show batch", list(ss.batches), index=list(ss.batches).index(ss.get("batch_view",
                                                                                           list(ss.batches)[-1])))
    out = ss.batches[label]
    st.dataframe(out["summary"], hide_index=True, width="stretch")
    bl = out["by_level"]
    c1, c2 = st.columns(2)
    with c1:
        plot(level_line_chart(bl, "Schedulability ratio (simulation)", "Schedulability ratio vs load", True), "l1")
        plot(level_line_chart(bl, "Avg response time", "Average response time vs load"), "l3")
        plot(level_line_chart(bl, "Avg High miss ratio", "High-criticality miss ratio vs load", True), "l5")
    with c2:
        plot(level_line_chart(bl, "Avg deadline miss ratio", "Deadline miss ratio vs load", True), "l2")
        plot(level_line_chart(bl, "Avg migrations", "Migrations per task set vs load"), "l4")
        plot(level_line_chart(bl, "Analytic test pass ratio", "Design-time test pass ratio vs load", True), "l6")
    st.caption("Solid = plain EDF, dashed = criticality-first EDF. Every point is the average over the task sets "
               "at that load.")


def page_export():
    ss = st.session_state
    st.title("7 · Export")
    conf = current_config()
    sheets = {"Task configuration": ss.tasks_df}
    res = fresh_results(conf) if not conf["errors"] else {}
    if res:
        rl = list(res.values())
        sheets.update({
            "Algorithm comparison": comparison_table(res),
            "Per criticality": pd.concat([criticality_metrics(r) for r in rl], ignore_index=True),
            "Per task": pd.concat([task_metrics(r).assign(Algorithm=r.algorithm) for r in rl], ignore_index=True),
            "Per job": pd.concat([job_table(r) for r in rl], ignore_index=True),
            "Aperiodic": pd.concat([aperiodic_table(r) for r in rl], ignore_index=True),
            "High check": pd.concat([high_check_table(r) for r in rl], ignore_index=True),
            "Timeline": pd.concat([timeline_segments(r) for r in rl], ignore_index=True),
        })
    if ss.mode_table is not None:
        sheets["Plain vs crit-first"] = ss.mode_table
    for i, (label, out) in enumerate(ss.batches.items(), 1):
        for part in ("settings", "summary", "by_level", "results", "tasksets"):
            sheets[f"B{i} {part}"] = out[part]
    if ss.pi is not None:
        sheets["Pi vs simulator"] = ss.pi["cmp"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button("⬇ Download everything (Excel)", to_excel_bytes(sheets), f"rtos_results_{stamp}.xlsx",
                       type="primary")
    st.caption(f"{len(sheets)} sheets. Batch sheets: B1, B2 … in the order you ran them.")
    st.subheader("Single CSV files")
    for name, df in sheets.items():
        if df is not None and len(df):
            st.download_button(f"{name} (.csv)", to_csv_bytes(df), f"{name.lower().replace(' ', '_')}_{stamp}.csv",
                               key=f"dl_{name}")
    if not res:
        st.info("Run a simulation (3 · Run) to also export results, timelines and per-job tables.")


def page_pi():
    ss = st.session_state
    st.title("Raspberry Pi results")
    st.markdown("Upload the **hw_*.csv** files of one run from the Pi (one folder: `hw_run_info.csv`, `hw_tasks.csv`, "
                "`hw_summary.csv`, `hw_trace_P/G/C.csv`). The simulator re-runs the same task set with the same "
                "settings and compares the schedules slot by slot.")
    files = st.file_uploader("hw_*.csv files", type=["csv"], accept_multiple_files=True)
    if files:
        by = {f.name: f for f in files}
        need = ["hw_run_info.csv", "hw_tasks.csv"]
        missing = [n for n in need if n not in by]
        traces = {c: pd.read_csv(by[f"hw_trace_{c}.csv"]) for c in "PGC" if f"hw_trace_{c}.csv" in by}
        if missing or not traces:
            st.error("Missing: " + ", ".join(missing + ([] if traces else ["hw_trace_P/G/C.csv"])))
        else:
            try:
                info = pd.read_csv(by["hw_run_info.csv"]).iloc[0]
                tdf = read_task_csv(by["hw_tasks.csv"])
                summ = pd.read_csv(by["hw_summary.csv"]) if "hw_summary.csv" in by else None
                cmp, sims, diffs = compare_frames(info, tdf, traces, summ)
                ss.pi = dict(info=info, tasks=tdf, traces=traces, cmp=cmp, sims=sims, diffs=diffs)
            except Exception as exc:          # noqa: BLE001 - show any parse problem
                st.error(f"Could not compare: {exc}")
    if ss.pi is None:
        st.info("No Pi data loaded yet.")
        return
    pi = ss.pi
    s = run_settings(pi["info"])
    st.caption(f"{len(pi['tasks'])} tasks · {s['cores']} cores · cluster {s['cluster']} · horizon {s['horizon']} · "
               f"tick {s['tick_ms']} ms · {'criticality-first' if s['crit'] else 'plain'} EDF · on miss {s['policy']}")
    cmp = pi["cmp"]
    cols = st.columns(len(cmp))
    for col, (_, r) in zip(cols, cmp.iterrows()):
        ok = r["Schedule match"] == "MATCH"
        col.markdown(f"<div class='kpi' style='--c:{ALGO_COLORS[r['Algorithm']]}'><div class='name'>{r['Algorithm']}</div>"
                     f"<div class='row'><span>Schedule vs simulator</span>{badge(r['Schedule match'], 'ok' if ok else 'bad')}</div>"
                     f"<div class='row'><span>Identical slots</span><b>{r['Slots identical']} / {r['Slots compared']}</b></div>"
                     f"<div class='row'><span>Misses HW · Sim</span><b>{r.get('HW deadline misses', '—')} · {r['Sim deadline misses']}</b></div>"
                     f"<div class='row'><span>High guaranteed</span><b>{r.get('HW High guaranteed', r['Sim High guaranteed'])}</b></div>"
                     f"<div class='row'><span>Wrong CPU</span><b>{r['Jobs on wrong physical CPU']}</b></div></div>",
                     unsafe_allow_html=True)
    st.write("")
    t1, t2, t3 = st.tabs(["Gantt: Pi vs simulator", "Measured overhead", "Full table"])
    with t1:
        for code, trace in pi["traces"].items():
            name = {"P": "Partitioned EDF", "G": "Global EDF", "C": "Clustered EDF"}[code]
            a, b = st.columns(2)
            with a:
                plot(gantt_from_trace(trace, pi["tasks"], f"{name} · executed on the Pi", s["cluster"]), f"pi_{code}")
            with b:
                plot(gantt_chart(pi["sims"][name], show_idle=False, title=f"{name} · simulator"), f"sim_{code}")
            d = pi["diffs"].get(name, [])
            if d:
                st.warning(f"{len(d)} different slots, first: " + "; ".join(
                    f"t={t} core {c}: Pi {g}, sim {e}" for t, c, g, e in d[:3]))
    with t2:
        if "HW decision avg (us)" in cmp:
            a, b = st.columns(2)
            with a:
                plot(hw_overhead_chart(cmp, "decision"), "ov1")
            with b:
                plot(hw_overhead_chart(cmp, "dispatch"), "ov2")
            st.caption("The simulator assumes zero overhead; these are the real costs measured on the Pi "
                       "(tick overruns and wrong-CPU counts are in the full table).")
        else:
            st.info("Upload hw_summary.csv to see the measured overheads.")
        lv = []
        for _, r in cmp.iterrows():
            for lvl in ["High", "Medium", "Low"]:
                lv.append({"Algorithm": r["Algorithm"], "EDF mode": "Plain EDF" if not s["crit"] else
                           "Criticality-first EDF", "Criticality": lvl,
                           "Deadline misses": r.get(f"HW {lvl} misses", r[f"Sim {lvl} misses"])})
        plot(criticality_bar(pd.DataFrame(lv)), "pi_crit")
    with t3:
        st.dataframe(cmp.set_index("Algorithm").T.astype(str), width="stretch")
        st.download_button("Download hw_vs_simulator.csv", to_csv_bytes(cmp), "hw_vs_simulator.csv")


ROUTES = {PAGES[0]: page_home, PAGES[1]: page_setup, PAGES[2]: page_review, PAGES[3]: page_run,
          PAGES[4]: page_results, PAGES[5]: page_compare, PAGES[6]: page_batch, PAGES[7]: page_export,
          PAGES[8]: page_pi}
ROUTES[page]()
