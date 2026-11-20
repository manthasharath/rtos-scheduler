# Multi-Core RTOS Scheduling: Partitioned vs Global vs Clustered EDF

**Performance Comparison of Partitioned, Global and Clustered Scheduling Algorithms in Multi-Core RTOS**
(BITS Pilani, Real-Time Embedded Systems project). Pure Python + C, no licensed software.

Three versions, one set of rules:

| Version | Where | Start |
|---|---|---|
| **Website** (graphs, animated Gantt charts, batch experiments, Pi results page) | PC | double-click `Run_Simulator.bat`, or `python -m streamlit run app.py` |
| **Terminal** (same menu as the Pi) | PC | double-click `Run_Terminal_Version.bat`, or `python cli.py` |
| **Raspberry Pi** (real 4-core execution, LEDs, live Gantt, measured overhead) | Pi | `hardware/` folder, see `hardware/README_HARDWARE.md` |

The Pi program uses exactly the same scheduling rules as the simulator. `compare_with_simulator.py`
(or the website's *Raspberry Pi results* page) re-runs the simulator on the Pi's task set and compares
the two schedules slot by slot: **MATCH** = identical.

---

## 1. Install (PC, once)

Python 3.10 or newer. In the project folder:
```
python -m pip install -r requirements.txt
```
(`Run_Simulator.bat` does this automatically on the first start.)

## 2. Task table (same everywhere: website, terminal, Pi, CSV)

| Column | Meaning |
|---|---|
| Name | no spaces or commas |
| Type | **P** periodic · **S** sporadic · **A** aperiodic |
| C | execution time (worst case), time units |
| T | P: period. S: minimum gap between releases |
| D | relative deadline, C ≤ D ≤ T |
| Priority | 1 = highest. Only breaks ties of equal deadline and criticality. Empty = automatic (shorter period first) |
| Criticality | High / Medium / Low (H/M/L accepted) |
| Arrival | aperiodic jobs only: arrival time |

```
Name,Type,C,T,D,Priority,Criticality,Arrival
EngineCtl,P,2,5,5,,High,
BrakePedal,S,1,10,5,,High,
DiagScan,A,6,,,,,12
```
Ready files: `example_csv/` (the five built-in examples). Old 6-column files (`Name,C,T,D,Priority,Criticality`) still load.

## 3. How the scheduler works

* Discrete time, preemptive, decision at every time unit. Only the oldest unfinished job of a task may run.
* **Partitioned EDF** – First Fit Decreasing by density C/min(D,T), a core accepts while Σ density ≤ 1; EDF per core; no migration.
* **Global EDF** – one queue, the m earliest-deadline jobs run; a job keeps its core if free, else lowest free core.
* **Clustered EDF** – clusters of k cores; FFD to clusters with the GFB test Σδ ≤ k − (k−1)·δmax; global EDF inside a cluster.
* **EDF modes**
  * *Plain EDF* (default): earliest absolute deadline first.
  * *Criticality-first EDF*: High jobs always before Medium/Low jobs, EDF inside each group. Partitioned and
    Clustered allocate in **DCDU** order (High, then Medium, then Low; each by decreasing density).
  * Tie-break (both): deadline → higher criticality → priority number → task order → job number.
* **High tasks guaranteed** – design-time test on the High tasks alone (density ≤ 1 per core; GFB per cluster or on all
  cores). If **YES**, criticality-first EDF meets every High deadline whatever the Medium/Low tasks do (checked by the
  tests on hundreds of overloaded random sets). Plain EDF gives no such guarantee.
* **Sporadic**: next release = previous release + T + extra gap; the extra gap is 0 … gap% of T from a fixed hash of
  (seed, task, job) – the same on the PC and the Pi.
* **Aperiodic**: background service – only on cores left idle by P/S jobs, oldest arrival first. They never delay a
  real-time job. (A server such as TBS is future work.)
* Deadline miss: *continue* (late job keeps running) or *abort* (dropped at its deadline).
* Overheads are **counted** (preemptions, migrations, context switches) but cost no time in the simulator; the Pi
  **measures** them (decision time, dispatch latency, tick overruns).

**Proof vs observation:** "no deadline misses" means none in the simulated horizon. FFD/GFB tests are sufficient
conditions. For Partitioned EDF with D = T one hyperperiod is conclusive; for Global/Clustered EDF it is not a proof.

## 4. Menu (terminal and Pi)

```
1 Number of cores          6 Choose algorithms (P / G / C / P+G / P+C / G+C / all three)
2 Cluster size             7 Run: results + Gantt charts
3 Enter tasks              8 PC: batch experiments    Pi: plain vs criticality-first on the hardware
  A manual  B CSV          9 Export results
  C random  D example      0 Exit
4 Settings
5 Show / check / edit      At ANY question:  X = main menu,  Q = quit
```

## 5. Built-in examples (results as simulated)

| Example | Cores | Shows |
|---|---|---|
| Automotive controller (P + S + A) – default | 4 | Realistic mixed set. Plain EDF: no misses in all three. Criticality-first: no High miss; Partitioned misses 2 Medium jobs (price of High-first on a packed core) |
| Dhall effect | 2 | Global EDF misses at 65 % load; Partitioned meets all |
| Partitioning fails | 2 | 3 × 60 % tasks: FFD cannot place one, Partitioned misses; Global meets all |
| Heavy-task mix | 4 | Partitioned and Global miss; Clustered (k = 2) meets all (plain EDF) |
| Overload + criticality | 4 | 116 % load. Plain EDF misses High jobs in all three; criticality-first: High guaranteed YES, 0 High misses |

## 6. Files

```
app.py                 website (Streamlit)
cli.py                 terminal version
task_model.py          task table, validation, CSV
scheduler.py           discrete-time EDF engine, sporadic releases, aperiodic background service
partitioned_edf.py     FFD / DCDU allocation + per-core EDF
global_edf.py          global EDF, GFB test, High-only check
clustered_edf.py       clusters, FFD/DCDU + GFB, EDF per cluster
algorithms.py          single entry point, the 7 algorithm choices
metrics.py             all metrics and tables
experiments.py         batch experiments (UUniFast-Discard, same sets for every algorithm)
task_generator.py      UUniFast(-Discard) random task sets
examples.py            the 5 built-in examples
visualization.py       Plotly charts (Gantt, animated Gantt, comparisons, Pi overhead)
compare_with_simulator.py   Pi run(s) vs simulator
export_results.py      CSV / Excel export
hardware/              Raspberry Pi program (rtos_hw.c), Makefile, collect_data.sh, guide
example_csv/           example task files
tests/                 pytest tests (python -m pytest -q)
```

## 7. Verification

* `python -m pytest -q` → 56 tests (EDF order, allocation, misses, policies, criticality-first, High guarantee on
  overloaded random sets, sporadic gaps, aperiodic never changes the real-time schedule, examples' claims).
* Pi program vs simulator: 480 of 480 algorithm runs identical on random P/S/A task sets (both EDF modes, both miss
  policies, 2 and 4 cores).
