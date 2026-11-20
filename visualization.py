"""
visualization.py
----------------
Plotly figures for the website.

Gantt chart encoding:
  * one row per core (grouped by cluster for Clustered EDF)
  * coloured bar  = a job executing; text = job label (e.g. Brake.J2)
  * hatched grey  = aperiodic job (background service in idle time)
  * light grey    = idle interval
  * diamond       = job MIGRATED onto this core
  * triangle      = job PREEMPTED at the end of this bar
  * x             = DEADLINE MISS, drawn at the job's absolute deadline
Hover any bar for release, deadline, interval and end event.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from metrics import cumulative_misses, timeline_segments
from scheduler import SimulationResult

# validated categorical order (blue, orange, aqua, yellow, magenta, green, violet, red)
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ALGO_COLORS = {"Partitioned EDF": CATEGORICAL[0], "Global EDF": CATEGORICAL[1], "Clustered EDF": CATEGORICAL[2]}
CRIT_COLORS = {"High": "#6250d6", "Medium": "#eb6834", "Low": "#1baf7a"}
APERIODIC_COLOR = "#7d8794"
IDLE_COLOR = "rgba(160,160,160,0.22)"
MISS_COLOR = "#c62828"
GRID = "rgba(128,128,128,0.18)"


def task_color_map(task_names: List[str]) -> Dict[str, str]:
    # more than 8 tasks re-use hues; every bar also carries its job label
    return {name: CATEGORICAL[i % len(CATEGORICAL)] for i, name in enumerate(task_names)}


def _core_label(result: SimulationResult, core: int) -> str:
    if result.algorithm == "Clustered EDF":
        for i, cl in enumerate(result.clusters):
            if core in cl:
                return f"Core {core} · Cl {i + 1}"
    return f"Core {core}"


def _base_layout(fig: go.Figure, title: str, height: int, x_range=None, x_title="Time (units)"):
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=15)),
        height=height,
        margin=dict(l=10, r=10, t=56, b=40),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(title=x_title, gridcolor=GRID, zeroline=False, range=x_range)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def _bar_colors(result: SimulationResult, ex: pd.DataFrame, color_by: str) -> List[str]:
    tmap = task_color_map([t.name for t in result.tasks if t.is_realtime])
    out = []
    for _, r in ex.iterrows():
        if r["Kind"] == "A":
            out.append(APERIODIC_COLOR)
        elif color_by == "criticality":
            out.append(CRIT_COLORS.get(r["Criticality"], CATEGORICAL[0]))
        else:
            out.append(tmap[r["Task"]])
    return out


def _miss_points(result: SimulationResult, until: Optional[int] = None):
    xs, ys, txt = [], [], []
    for j in result.jobs:
        if j.missed and j.abs_deadline <= result.horizon and (until is None or j.abs_deadline <= until):
            core = j.last_core if j.last_core is not None else min(result.clusters[result.allocation.get(j.task_id, 0)])
            xs.append(j.abs_deadline)
            ys.append(_core_label(result, core))
            txt.append(f"{j.label}: deadline {j.abs_deadline}, done {j.executed}/{j.wcet}")
    return xs, ys, txt


HOVER = ("<b>%{customdata[0]}</b> on %{y}<br>Runs t = %{customdata[1]} → %{customdata[2]}<br>"
         "Release %{customdata[3]} · Deadline %{customdata[4]}<br>%{customdata[6]}<br>"
         "End event: %{customdata[5]}<extra></extra>")


def _custom(ex: pd.DataFrame):
    info = ex.apply(lambda r: "Aperiodic job (background)" if r["Kind"] == "A"
                    else f"{r['Criticality']} criticality" + (" · MISSED deadline" if r["Deadline missed"] else ""),
                    axis=1) if len(ex) else []
    dl = ex["Abs deadline"].apply(lambda v: "—" if pd.isna(v) else int(v)) if len(ex) else []
    return list(zip(ex["Job"], ex["Start"], ex["End"], ex["Release"], dl, ex["End event"], info))


def gantt_chart(result: SimulationResult, show_idle: bool = True, window: Optional[tuple] = None,
                title: Optional[str] = None, color_by: str = "task") -> go.Figure:
    segs = timeline_segments(result, include_idle=True)
    core_order = [_core_label(result, c) for c in range(result.num_cores)]
    segs["CoreLabel"] = segs["Core"].map(lambda c: _core_label(result, c))
    fig = go.Figure()
    if show_idle:
        idle = segs[segs["Type"] == "Idle"]
        if len(idle):
            fig.add_trace(go.Bar(
                y=idle["CoreLabel"], x=idle["Duration"], base=idle["Start"], orientation="h",
                marker=dict(color=IDLE_COLOR, line=dict(width=0)), name="Idle",
                customdata=idle[["Start", "End"]].values,
                hovertemplate="<b>Idle</b><br>%{y}<br>t = %{customdata[0]} → %{customdata[1]}<extra></extra>",
            ))
    ex = segs[segs["Type"] != "Idle"]
    # one trace per legend entry (task, criticality level or aperiodic)
    if color_by == "criticality":
        groups = [(lvl, ex[(ex["Kind"] != "A") & (ex["Criticality"] == lvl)], CRIT_COLORS[lvl], "")
                  for lvl in ["High", "Medium", "Low"]]
    else:
        tmap = task_color_map([t.name for t in result.tasks if t.is_realtime])
        groups = [(t.name, ex[ex["Task"] == t.name], tmap[t.name], "") for t in result.tasks if t.is_realtime]
    groups.append(("Aperiodic", ex[ex["Kind"] == "A"], APERIODIC_COLOR, "/"))
    for name, d, color, pattern in groups:
        if not len(d):
            continue
        fig.add_trace(go.Bar(
            y=d["CoreLabel"], x=d["Duration"], base=d["Start"], orientation="h", name=name,
            marker=dict(color=color, line=dict(color="white", width=1),
                        pattern=dict(shape=pattern, fgcolor="white", size=6, fillmode="overlay") if pattern else None),
            text=d["Job"], textposition="inside", insidetextanchor="middle", textangle=0,
            textfont=dict(color="white", size=10), customdata=_custom(d), hovertemplate=HOVER,
        ))
    rt = ex[ex["Kind"] != "A"]
    mig = ex[ex["Migrated in"]]
    if len(mig):
        fig.add_trace(go.Scatter(
            x=mig["Start"], y=mig["CoreLabel"], mode="markers", name="Migration in",
            marker=dict(symbol="diamond", size=10, color="white", line=dict(color="#222", width=2)),
            customdata=mig[["Job"]].values,
            hovertemplate="<b>Migration</b>: %{customdata[0]} resumes on %{y} at t=%{x}<extra></extra>"))
    pre = rt[rt["End event"] == "Preempted"]
    if len(pre):
        fig.add_trace(go.Scatter(
            x=pre["End"], y=pre["CoreLabel"], mode="markers", name="Preemption",
            marker=dict(symbol="triangle-down", size=10, color="#333"),
            customdata=pre[["Job"]].values,
            hovertemplate="<b>Preempted</b>: %{customdata[0]} at t=%{x}<extra></extra>"))
    mx, my, mt = _miss_points(result)
    if mx:
        fig.add_trace(go.Scatter(
            x=mx, y=my, mode="markers", name="Deadline miss",
            marker=dict(symbol="x", size=13, color=MISS_COLOR, line=dict(width=1, color="white")),
            text=mt, hovertemplate="<b>DEADLINE MISS</b><br>%{text}<extra></extra>"))
    fig.update_layout(barmode="overlay", bargap=0.3, uniformtext=dict(mode="hide", minsize=8))
    fig.update_yaxes(categoryorder="array", categoryarray=core_order[::-1], title=None)
    x_range = list(window) if window else [0, result.horizon]
    n_legend = len(fig.data)
    _base_layout(fig, title or f"{result.algorithm} · {result.mode_name}",
                 height=190 + 46 * result.num_cores + 22 * math.ceil(n_legend / 6), x_range=x_range)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.16 - 0.01 * n_legend, x=0, xanchor="left"),
                      margin=dict(l=10, r=10, t=50, b=40 + 22 * math.ceil(n_legend / 6)))
    return fig


def animated_gantt(results: Dict[str, SimulationResult], frame_ms: int = 150, color_by: str = "task",
                   max_frames: int = 200) -> go.Figure:
    """All chosen algorithms in one figure (one row each), animated in sync.
    Starts showing the full chart; Play from start / Pause / Resume and a time slider."""
    res = list(results.values())
    H = max(r.horizon for r in res)
    step = max(1, math.ceil(H / max_frames))
    times = list(range(step, H, step)) + [H]
    heights = [r.num_cores for r in res]
    fig = make_subplots(rows=len(res), cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=heights, subplot_titles=[f"{r.algorithm} · {r.mode_name}" for r in res])
    per_alg = []
    for r in res:
        segs = timeline_segments(r, include_idle=False)
        segs["CoreLabel"] = segs["Core"].map(lambda c, r=r: _core_label(r, c))
        segs["Color"] = _bar_colors(r, segs, color_by) if len(segs) else []
        per_alg.append(segs)

    def bar_data(segs, tau):
        d = segs[segs["Start"] < tau].copy()
        d["End"] = d["End"].clip(upper=tau)
        d["Duration"] = d["End"] - d["Start"]
        return d

    def traces_at(tau):
        out = []
        for r, segs in zip(res, per_alg):
            d = bar_data(segs, tau)
            out.append(go.Bar(y=d["CoreLabel"], x=d["Duration"], base=d["Start"], orientation="h",
                              marker=dict(color=d["Color"], line=dict(color="white", width=1)),
                              text=d["Job"], textposition="inside", insidetextanchor="middle", textangle=0,
                              textfont=dict(color="white", size=10), customdata=_custom(d),
                              hovertemplate=HOVER, showlegend=False))
            mx, my, mt = _miss_points(r, until=tau)
            out.append(go.Scatter(x=mx, y=my, mode="markers", text=mt, showlegend=False,
                                  marker=dict(symbol="x", size=13, color=MISS_COLOR, line=dict(width=1, color="white")),
                                  hovertemplate="<b>DEADLINE MISS</b><br>%{text}<extra></extra>"))
        return out

    def cursor(tau):
        return [dict(type="line", x0=tau, x1=tau, xref=f"x{i + 1}" if i else "x", y0=0, y1=1,
                     yref=f"y{i + 1} domain" if i else "y domain", line=dict(color=MISS_COLOR, width=2, dash="dot"))
                for i in range(len(res))]

    for i, tr in enumerate(traces_at(H)):
        fig.add_trace(tr, row=i // 2 + 1, col=1)
    for i, r in enumerate(res):
        order = [_core_label(r, c) for c in range(r.num_cores)][::-1]
        fig.update_yaxes(categoryorder="array", categoryarray=order, row=i + 1, col=1, gridcolor=GRID)
        fig.update_xaxes(range=[0, H], gridcolor=GRID, row=i + 1, col=1)
    fig.update_xaxes(title="Time (units)", row=len(res), col=1)
    fig.frames = [go.Frame(name=str(t), data=traces_at(t), layout=dict(shapes=cursor(t)),
                           traces=list(range(2 * len(res)))) for t in times]
    anim = dict(frame=dict(duration=frame_ms, redraw=True), transition=dict(duration=0), mode="immediate",
                fromcurrent=True)
    fig.update_layout(
        barmode="overlay", bargap=0.3, height=170 + sum(44 * h + 50 for h in heights),
        margin=dict(l=10, r=10, t=110, b=40), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        uniformtext=dict(mode="hide", minsize=8),
        updatemenus=[dict(type="buttons", direction="left", x=0, y=1.13, xanchor="left", yanchor="bottom",
                          pad=dict(r=6), showactive=False, buttons=[
                              dict(label="⏮ Play from start", method="animate",
                                   args=[[str(t) for t in times], {**anim, "fromcurrent": False}]),
                              dict(label="⏸ Pause", method="animate",
                                   args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate",
                                                      transition=dict(duration=0))]),
                              dict(label="▶ Resume", method="animate", args=[None, anim]),
                          ])],
        sliders=[dict(active=len(times) - 1, x=0, y=-0.02, len=1.0, yanchor="top", pad=dict(t=30),
                      currentvalue=dict(prefix="t = ", font=dict(size=13)),
                      steps=[dict(label=str(t), method="animate",
                                  args=[[str(t)], dict(frame=dict(duration=0, redraw=True), mode="immediate",
                                                       transition=dict(duration=0))]) for t in times])],
        shapes=cursor(H),
    )
    return fig


def gantt_from_trace(trace: pd.DataFrame, tasks_df: pd.DataFrame, title: str, cluster_size: Optional[int] = None
                     ) -> go.Figure:
    """Gantt chart of the schedule the Raspberry Pi actually executed (hw_trace_X.csv)."""
    kinds = dict(zip(tasks_df["Name"].astype(str), tasks_df["Type"].astype(str)))
    rt_names = [n for n, k in kinds.items() if k != "A"]
    tmap = task_color_map(rt_names)
    algo = str(trace["Algorithm"].iloc[0]) if len(trace) else ""
    cores = sorted(trace["Core"].unique())

    def label(c):
        if algo == "Clustered EDF" and cluster_size:
            return f"Core {c} · Cl {c // cluster_size + 1}"
        return f"Core {c}"
    rows = []
    for c in cores:
        d = trace[trace["Core"] == c].sort_values("Tick")
        start, prev = None, None
        for tick, job, task in zip(d["Tick"], d["Job"], d["Task"]):
            if job != prev:
                if prev is not None and prev != "Idle":
                    rows.append((label(c), start, tick, prev, prev_task))
                start, prev, prev_task = tick, job, task
        if prev is not None and prev != "Idle":
            rows.append((label(c), start, int(d["Tick"].max()) + 1, prev, prev_task))
    seg = pd.DataFrame(rows, columns=["Core", "Start", "End", "Job", "Task"])
    fig = go.Figure()
    for name in list(dict.fromkeys(seg["Task"])):
        d = seg[seg["Task"] == name]
        ap = kinds.get(name) == "A"
        fig.add_trace(go.Bar(
            y=d["Core"], x=d["End"] - d["Start"], base=d["Start"], orientation="h",
            name=name + (" (aperiodic)" if ap else ""),
            marker=dict(color=APERIODIC_COLOR if ap else tmap.get(name, CATEGORICAL[0]),
                        line=dict(color="white", width=1),
                        pattern=dict(shape="/", fgcolor="white", size=6, fillmode="overlay") if ap else None),
            text=d["Job"], textposition="inside", insidetextanchor="middle", textangle=0,
            textfont=dict(color="white", size=10), customdata=d[["Job", "Start", "End"]].values,
            hovertemplate="<b>%{customdata[0]}</b> on %{y}<br>t = %{customdata[1]} → %{customdata[2]}<extra></extra>"))
    H = int(trace["Tick"].max()) + 1 if len(trace) else 1
    fig.update_layout(barmode="overlay", bargap=0.3, uniformtext=dict(mode="hide", minsize=8), showlegend=False)
    fig.update_yaxes(categoryorder="array", categoryarray=[label(c) for c in cores][::-1], title=None)
    return _base_layout(fig, title, height=150 + 46 * len(cores), x_range=[0, H])


def metric_bar(df: pd.DataFrame, metric: str, title: Optional[str] = None, fmt: str = "") -> go.Figure:
    vals = df[metric].tolist()
    fig = go.Figure(go.Bar(
        x=df["Algorithm"], y=vals,
        marker=dict(color=[ALGO_COLORS.get(a, CATEGORICAL[0]) for a in df["Algorithm"]]),
        text=[("—" if v is None or (isinstance(v, float) and math.isnan(v)) else (f"{v:{fmt}}" if fmt else str(v)))
              for v in vals],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{x}<br>" + metric + ": %{y}<extra></extra>",
    ))
    fig.update_yaxes(title=None, rangemode="tozero")
    return _base_layout(fig, title or metric, height=300, x_title=None)


def per_core_util_chart(results: Dict[str, SimulationResult]) -> go.Figure:
    fig = go.Figure()
    for name, r in results.items():
        cores = sorted(r.timeline)
        util = [sum(1 for x in r.timeline[c] if x is not None) / r.horizon for c in cores]
        fig.add_trace(go.Bar(x=[f"Core {c}" for c in cores], y=util, name=name, marker=dict(color=ALGO_COLORS.get(name)),
                             hovertemplate=name + "<br>%{x}: %{y:.1%}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05], title=None)
    return _base_layout(fig, "Per-core utilization (load distribution)", height=340, x_title=None)


def per_task_response_chart(task_tables: Dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for name, df in task_tables.items():
        fig.add_trace(go.Bar(x=df["Task"], y=df["Max response"], name=name, marker=dict(color=ALGO_COLORS.get(name)),
                             hovertemplate=name + "<br>%{x}: max response %{y}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_yaxes(title=None, rangemode="tozero")
    return _base_layout(fig, "Maximum response time per task", height=340, x_title=None)


def cumulative_miss_chart(results: Dict[str, SimulationResult]) -> go.Figure:
    fig = go.Figure()
    for name, r in results.items():
        df = cumulative_misses(r)
        fig.add_trace(go.Scatter(x=df["Time"], y=df["Cumulative misses"], name=name, mode="lines",
                                 line=dict(color=ALGO_COLORS.get(name), width=2, shape="hv"),
                                 hovertemplate=name + "<br>t=%{x}: %{y} misses so far<extra></extra>"))
    fig.update_yaxes(title=None, rangemode="tozero")
    fig.update_layout(hovermode="x unified")
    return _base_layout(fig, "Cumulative deadline misses over time", height=320)


def level_line_chart(by_level: pd.DataFrame, metric: str, title: str, percent: bool = False) -> go.Figure:
    fig = go.Figure()
    col = "Series" if "Series" in by_level.columns else "Algorithm"
    for name in by_level[col].unique():
        d = by_level[by_level[col] == name].sort_values("Target U/m")
        alg = d["Algorithm"].iloc[0]
        crit = "criticality" in name
        fig.add_trace(go.Scatter(
            x=d["Target U/m"], y=d[metric], name=name, mode="lines+markers",
            line=dict(color=ALGO_COLORS.get(alg), width=2, dash="dash" if crit else "solid"),
            marker=dict(size=8, symbol="diamond" if crit else "circle", color=ALGO_COLORS.get(alg),
                        line=dict(color="white", width=2)),
            hovertemplate=name + "<br>U/m=%{x}: %{y}<extra></extra>"))
    if percent:
        fig.update_yaxes(tickformat=".0%", range=[-0.03, 1.05])
    fig.update_yaxes(title=None)
    fig.update_xaxes(tickformat=".2f")
    fig.update_layout(hovermode="x unified")
    return _base_layout(fig, title, height=340, x_title="Normalized utilization U/m")


def criticality_bar(crit_df: pd.DataFrame, metric: str = "Deadline misses") -> go.Figure:
    """Grouped bars: one group per criticality level, one bar per algorithm and EDF mode (hatched = crit-first)."""
    fig = go.Figure()
    for (alg, mode), d in crit_df.groupby(["Algorithm", "EDF mode"], sort=False):
        crit = mode != "Plain EDF"
        fig.add_trace(go.Bar(
            x=d["Criticality"], y=d[metric], name=f"{alg} ({'criticality-first' if crit else 'plain'})",
            marker=dict(color=ALGO_COLORS.get(alg), pattern=dict(shape="/" if crit else "", fillmode="overlay", fgcolor="white")),
            hovertemplate=f"{alg} · {mode}<br>%{{x}}: %{{y}}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    fig.update_yaxes(title=None, rangemode="tozero")
    return _base_layout(fig, f"{metric} by criticality", height=360, x_title=None)


def hw_overhead_chart(cmp: pd.DataFrame, which: str) -> go.Figure:
    """which = 'decision' or 'dispatch': avg and max per algorithm, in microseconds (measured on the Pi)."""
    cols = [f"HW {which} avg (us)", f"HW {which} max (us)"]
    fig = go.Figure()
    for col, shade in zip(cols, ["", "/"]):
        if col not in cmp:
            continue
        fig.add_trace(go.Bar(
            x=cmp["Algorithm"], y=cmp[col], name=col.replace("HW ", "").replace(" (us)", ""),
            marker=dict(color=[ALGO_COLORS.get(a) for a in cmp["Algorithm"]], pattern=dict(shape=shade, fillmode="overlay", fgcolor="white")),
            text=[f"{v:.1f}" if v is not None and not pd.isna(v) else "" for v in cmp[col]],
            textposition="outside", cliponaxis=False,
            hovertemplate="%{x}<br>" + col + ": %{y:.1f} µs<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.3)
    fig.update_yaxes(title="µs", rangemode="tozero")
    title = ("Scheduler decision time per tick (measured)" if which == "decision"
             else "Dispatch latency: tick start → core starts job (measured)")
    return _base_layout(fig, title, height=320, x_title=None)
