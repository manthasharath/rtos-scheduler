"""
examples.py
-----------
The 5 built-in examples (same in the website, terminal and Pi CSV files).
Each one was tuned on this simulator so that it shows its point; the
description states the actual simulated result (checked by
tests/test_examples.py, so it stays true).

Task values are chosen by us, inspired by typical systems; they are not real
industry data.
"""

from __future__ import annotations

import pandas as pd

from task_model import TASK_COLUMNS, fill_priorities


def _df(rows):
    return fill_priorities(pd.DataFrame(rows, columns=TASK_COLUMNS))


EXAMPLES = {
    "Automotive controller (P + S + A, 4 cores)": dict(
        cores=4, cluster=2,
        description=(
            "A car controller: High = engine, brakes, airbag, steering, brake pedal (sporadic) and collision "
            "alert (sporadic); Medium = cruise control, sensor fusion, battery; Low = dashboard, logger, "
            "infotainment; plus 3 aperiodic diagnostic jobs. Load 66 % of 4 cores. Plain EDF: all three "
            "algorithms meet every deadline. Criticality-first EDF: no High task misses in any algorithm, but "
            "Partitioned EDF misses 2 SensorFusion (Medium) jobs, because on its fully packed core the High "
            "jobs always run first - the price of protecting High tasks. Aperiodic jobs run only in idle time."),
        df=_df([
            ("EngineCtl", "P", 2, 5, 5, None, "High", None),
            ("BrakeCtl", "P", 3, 10, 10, None, "High", None),
            ("Airbag", "P", 2, 20, 10, None, "High", None),
            ("SteeringCtl", "P", 4, 20, 20, None, "High", None),
            ("BrakePedal", "S", 1, 10, 5, None, "High", None),
            ("CollisionAlert", "S", 2, 20, 10, None, "High", None),
            ("CruiseCtl", "P", 5, 20, 20, None, "Medium", None),
            ("SensorFusion", "P", 4, 10, 10, None, "Medium", None),
            ("BatteryMon", "P", 5, 50, 50, None, "Medium", None),
            ("Dashboard", "P", 10, 50, 50, None, "Low", None),
            ("DataLogger", "P", 20, 100, 100, None, "Low", None),
            ("Infotainment", "P", 15, 50, 50, None, "Low", None),
            ("DiagScan", "A", 6, None, None, None, None, 12),
            ("OTACheck", "A", 8, None, None, None, None, 40),
            ("FaultLog", "A", 3, None, None, None, None, 65),
        ])),
    "Dhall effect: Global EDF fails (2 cores)": dict(
        cores=2, cluster=1,
        description=(
            "Two light tasks and one heavy task at only 65 % load. Global EDF runs the two light jobs first "
            "(earlier deadlines) on both cores, so the heavy task H misses its deadline. Partitioned EDF puts H "
            "alone on one core and meets every deadline (Dhall & Liu, 1978)."),
        df=_df([
            ("L1", "P", 2, 10, 10, None, "Low", None),
            ("L2", "P", 2, 10, 10, None, "Low", None),
            ("H", "P", 10, 11, 11, None, "Medium", None),
        ])),
    "Partitioning fails: Global EDF wins (2 cores)": dict(
        cores=2, cluster=2,
        description=(
            "Three tasks of 60 % each (180 % of 2 cores). No core can take two of them, so First Fit "
            "Decreasing fails to place one task and Partitioned EDF misses deadlines. Global EDF shares both "
            "cores and meets every deadline."),
        df=_df([
            ("A", "P", 3, 5, 5, None, "Medium", None),
            ("B", "P", 6, 10, 10, None, "Medium", None),
            ("C", "P", 6, 10, 10, None, "Medium", None),
        ])),
    "Heavy-task mix: Clustered EDF wins (4 cores)": dict(
        cores=4, cluster=2,
        description=(
            "Six heavy tasks (40-80 % of a core each), 90 % load on 4 cores. Partitioned EDF cannot pack them "
            "and misses; Global EDF also misses; Clustered EDF (2 clusters of 2 cores) meets every deadline "
            "(plain EDF). Shows why Clustered EDF exists. In criticality-first mode all three miss 1 job, "
            "none of them High."),
        df=_df([
            ("Heavy1", "P", 4, 5, 5, None, "High", None),
            ("Heavy2", "P", 7, 10, 10, None, "Medium", None),
            ("Heavy3", "P", 3, 5, 5, None, "Low", None),
            ("Heavy4", "P", 3, 5, 5, None, "High", None),
            ("Heavy5", "P", 5, 10, 10, None, "Medium", None),
            ("Heavy6", "P", 2, 5, 5, None, "Low", None),
        ])),
    "Overload + criticality (4 cores)": dict(
        cores=4, cluster=2,
        description=(
            "116 % load on 4 cores, so some deadlines must be missed. Plain EDF misses High-criticality jobs in "
            "all three algorithms. Criticality-first EDF shows 'High tasks guaranteed: YES' and no High task "
            "misses; only Medium/Low tasks miss."),
        df=_df([
            ("Sensor1", "P", 2, 5, 5, None, "Medium", None),
            ("Logger1", "P", 3, 5, 5, None, "Low", None),
            ("Control1", "P", 3, 5, 5, None, "Medium", None),
            ("Planner", "P", 14, 20, 20, None, "Medium", None),
            ("SafetyMon", "P", 13, 20, 20, None, "High", None),
            ("MotorCtl", "P", 5, 10, 10, None, "High", None),
            ("Logger2", "P", 3, 5, 5, None, "Low", None),
            ("Watchdog", "P", 3, 5, 5, None, "High", None),
        ])),
}

DEFAULT_EXAMPLE = "Automotive controller (P + S + A, 4 cores)"
