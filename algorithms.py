"""
algorithms.py
-------------
Single entry point used by the website, the terminal version, the batch
experiments and the Pi comparison script.
"""

from __future__ import annotations

from typing import List

from clustered_edf import run_clustered_edf
from global_edf import run_global_edf
from partitioned_edf import run_partitioned_edf
from scheduler import SimulationResult
from task_model import Task

ALGORITHMS = ["Partitioned EDF", "Global EDF", "Clustered EDF"]
SHORT = {"Partitioned EDF": "P", "Global EDF": "G", "Clustered EDF": "C"}

# the 7 algorithm choices offered in every version (menu number -> algorithms)
ALGORITHM_CHOICES = {
    "1": ("Partitioned", ["Partitioned EDF"]),
    "2": ("Global", ["Global EDF"]),
    "3": ("Clustered", ["Clustered EDF"]),
    "4": ("Partitioned + Global", ["Partitioned EDF", "Global EDF"]),
    "5": ("Partitioned + Clustered", ["Partitioned EDF", "Clustered EDF"]),
    "6": ("Global + Clustered", ["Global EDF", "Clustered EDF"]),
    "7": ("All three (Partitioned + Global + Clustered)", list(ALGORITHMS)),
}


def run_algorithm(
    name: str,
    tasks: List[Task],
    num_cores: int,
    horizon: int,
    cluster_size: int = 2,
    miss_policy: str = "continue",
    crit_mode: bool = False,
    gap_pct: int = 50,
    seed: int = 1,
) -> SimulationResult:
    if name == "Partitioned EDF":
        return run_partitioned_edf(tasks, num_cores, horizon, miss_policy, crit_mode, gap_pct, seed)
    if name == "Global EDF":
        return run_global_edf(tasks, num_cores, horizon, miss_policy, crit_mode, gap_pct, seed)
    if name == "Clustered EDF":
        return run_clustered_edf(tasks, num_cores, horizon, cluster_size, miss_policy, crit_mode, gap_pct, seed)
    raise ValueError(f"Unknown algorithm '{name}'. Choose from {ALGORITHMS}.")
