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

from metrics import cumulative_misses, schedule_events, timeline_segments
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


def _event_layers(result: SimulationResult, ev: pd.DataFrame, pos: Dict[str, int], job_color: Dict[str, str],
                  until: Optional[int] = None, xref: str = "x", yref: str = "y",
                  show_preemptions: bool = True, show_migrations: bool = True):
    """Preemption markers (with 'by X' label), dotted waiting lines and migration arrows."""
    if until is not None:
        ev = ev[ev["Time"] <= until]
    lab = lambda c: _core_label(result, int(c))
    traces, arrows = [], []
    pre = ev[ev["Event"] == "Preemption"] if show_preemptions else ev.iloc[0:0]
    few = len(pre) <= 40
    traces.append(go.Scatter(
        x=list(pre["Time"]), y=[lab(c) for c in pre["Core"]], mode="markers+text" if few else "markers",
        name="Preemption ▼", text=[f"by {b}" if isinstance(b, str) else "" for b in pre["By"]] if few else None,
        textposition="top center", textfont=dict(size=10, color="#222"),
        marker=dict(symbol="triangle-down", size=12, color="#333", line=dict(color="white", width=1)),
        hovertext=list(pre["Text"]), hovertemplate="<b>PREEMPTION</b> at t=%{x}<br>%{hovertext}<extra></extra>"))
    res = ev[ev["Event"] == "Resume"] if show_preemptions else ev.iloc[0:0]
    xs, ys = [], []
    for _, r in res.iterrows():
        xs += [int(r["Since"]), int(r["Time"]), None]
        ys += [lab(r["Core"]), lab(r["Core"]), None]
    traces.append(go.Scatter(x=xs, y=ys, mode="lines", name="Waiting (preempted)", hoverinfo="skip",
                             line=dict(color="#888", width=2, dash="dot")))
    mig = ev[ev["Event"] == "Migration"] if show_migrations else ev.iloc[0:0]
    for _, r in mig.iterrows():
        arrows.append(dict(
            x=int(r["Time"]), y=pos[lab(r["To core"])], ax=int(r["Since"]), ay=pos[lab(r["Core"])],
            xref=xref, yref=yref, axref=xref, ayref=yref, text="", showarrow=True, arrowhead=2, arrowsize=1.3,
            arrowwidth=2.2, arrowcolor=job_color.get(r["Job"], "#222"), standoff=3, startstandoff=3,
            hovertext=f"t={int(r['Time'])}: {r['Text']}"))
    traces.append(go.Scatter(
        x=list(mig["Time"]), y=[lab(c) for c in mig["To core"]], mode="markers", name="Migration →",
        marker=dict(symbol="diamond", size=10, color="white", line=dict(color="#222", width=2)),
        hovertext=list(mig["Text"]), hovertemplate="<b>MIGRATION</b> at t=%{x}<br>%{hovertext}<extra></extra>"))
    return traces, arrows


def _job_colors(result: SimulationResult, segs: pd.DataFrame, color_by: str) -> Dict[str, str]:
    if not len(segs):
        return {}
    cols = _bar_colors(result, segs, color_by)
    return dict(zip(segs["Job"], cols))


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
                title: Optional[str] = None, color_by: str = "task", show_preemptions: bool = True,
                show_migrations: bool = True) -> go.Figure:
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
    order = core_order[::-1]
    pos = {c: i for i, c in enumerate(order)}
    ev = schedule_events(result)
    layers, arrows = _event_layers(result, ev, pos, _job_colors(result, ex, color_by),
                                   show_preemptions=show_preemptions, show_migrations=show_migrations)
    for tr in layers:
        if len(tr.x or []):
            fig.add_trace(tr)
    fig.update_layout(annotations=arrows)
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
                   max_frames: int = 200, show_preemptions: bool = True, show_migrations: bool = True) -> go.Figure:
    """All chosen algorithms in one figure (one row each), animated in sync.
    Shows bars, deadline misses, preemptions (▼ 'by X', dotted waiting line) and migration arrows as they
    happen, a live counter and the latest event per algorithm. Starts showing the full chart."""
    res = list(results.values())
    H = max(r.horizon for r in res)
    step = max(1, math.ceil(H / max_frames))
    times = list(range(step, H, step)) + [H]
    heights = [r.num_cores for r in res]
    n = len(res)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.16 if n > 1 else 0.1,
                        row_heights=heights, subplot_titles=[" "] * n)
    per_alg = []
    for r in res:
        segs = timeline_segments(r, include_idle=False)
        segs["CoreLabel"] = segs["Core"].map(lambda c, r=r: _core_label(r, c))
        segs["Color"] = _bar_colors(r, segs, color_by) if len(segs) else []
        order = [_core_label(r, c) for c in range(r.num_cores)][::-1]
        per_alg.append(dict(segs=segs, ev=schedule_events(r), order=order, pos={c: i for i, c in enumerate(order)},
                            colors=dict(zip(segs["Job"], segs["Color"])) if len(segs) else {}))
    title_anns = list(fig.layout.annotations)

    def bar_data(segs, tau):
        d = segs[segs["Start"] < tau].copy()
        d["End"] = d["End"].clip(upper=tau)
        d["Duration"] = d["End"] - d["Start"]
        return d

    def state_at(tau):
        traces, anns = [], []
        for i, (r, a) in enumerate(zip(res, per_alg)):
            xr, yr = ("x", "y") if i == 0 else (f"x{i + 1}", f"y{i + 1}")
            d = bar_data(a["segs"], tau)
            traces.append(go.Bar(y=d["CoreLabel"], x=d["Duration"], base=d["Start"], orientation="h",
                                 marker=dict(color=d["Color"], line=dict(color="white", width=1)),
                                 text=d["Job"], textposition="inside", insidetextanchor="middle", textangle=0,
                                 textfont=dict(color="white", size=10), customdata=_custom(d),
                                 hovertemplate=HOVER, showlegend=False))
            mx, my, mt = _miss_points(r, until=tau)
            traces.append(go.Scatter(x=mx, y=my, mode="markers", text=mt, showlegend=False,
                                     marker=dict(symbol="x", size=13, color=MISS_COLOR,
                                                 line=dict(width=1, color="white")),
                                     hovertemplate="<b>DEADLINE MISS</b><br>%{text}<extra></extra>"))
            layers, arrows = _event_layers(r, a["ev"], a["pos"], a["colors"], until=tau, xref=xr, yref=yr,
                                           show_preemptions=show_preemptions, show_migrations=show_migrations)
            for tr in layers:
                tr.showlegend = False
                traces.append(tr)
            anns += arrows
            ev = a["ev"][a["ev"]["Time"] <= tau]
            npre = int((ev["Event"] == "Preemption").sum())
            nmig = int((ev["Event"] == "Migration").sum())
            nmiss = int((ev["Event"] == "Deadline miss").sum())
            key = ev[ev["Event"].isin(["Preemption", "Migration", "Deadline miss"])]
            last = (f"t={int(key['Time'].iloc[-1])}: {key['Text'].iloc[-1]}" if len(key)
                    else "no preemption or migration yet")
            t_ann = dict(title_anns[i].to_plotly_json())
            t_ann["text"] = (f"<b>{r.algorithm} · {r.mode_name}</b>  —  t = {tau} · preemptions {npre} · "
                             f"migrations {nmig} · misses {nmiss}<br><span style='font-size:11px'>{last}</span>")
            anns.append(t_ann)
        return traces, anns

    def cursor(tau):
        return [dict(type="line", x0=tau, x1=tau, xref=f"x{i + 1}" if i else "x", y0=0, y1=1,
                     yref=f"y{i + 1} domain" if i else "y domain", line=dict(color=MISS_COLOR, width=2, dash="dot"))
                for i in range(n)]

    full_traces, full_anns = state_at(H)
    per = len(full_traces) // n
    for k, tr in enumerate(full_traces):
        fig.add_trace(tr, row=k // per + 1, col=1)
    for i, (r, a) in enumerate(zip(res, per_alg)):
        fig.update_yaxes(categoryorder="array", categoryarray=a["order"], row=i + 1, col=1, gridcolor=GRID)
        fig.update_xaxes(range=[0, H], gridcolor=GRID, row=i + 1, col=1)
    fig.update_xaxes(title="Time (units)", row=n, col=1)
    frames = []
    for t in times:
        tr, an = state_at(t)
        frames.append(go.Frame(name=str(t), data=tr, layout=dict(shapes=cursor(t), annotations=an),
                               traces=list(range(len(full_traces)))))
    fig.frames = frames
    anim = dict(frame=dict(duration=frame_ms, redraw=True), transition=dict(duration=0), mode="immediate",
                fromcurrent=True)
    fig.update_layout(
        barmode="overlay", bargap=0.3, height=200 + sum(44 * h + 80 for h in heights),
        margin=dict(l=10, r=10, t=130, b=40), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        uniformtext=dict(mode="hide", minsize=8), showlegend=False,
        updatemenus=[dict(type="buttons", direction="left", x=0, y=1.0, xanchor="left", yanchor="bottom",
                          pad=dict(b=70, r=6), showactive=False, buttons=[
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
    fig.layout.annotations = full_anns          # replace (not merge) the subplot-title placeholders
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
