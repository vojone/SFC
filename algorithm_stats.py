import threading
import numpy.typing
import time

import ant_algorithm as alg
from gui import GUI
from dataclasses import dataclass, field

@dataclass
class AlgorithmRun:
    seed : int
    algorithm : str | None = None
    finished : bool = False
    params : dict = field(default_factory=lambda: {})
    total_time : float = 0.0
    best_solution : tuple[list, float] = None
    best_len_history : list[list[float]] = field(default_factory=lambda: [])


class AlgorithmStats:
    def __init__(self):
        self.run = None
        self.run_history = {}
        self.run_groups = {}

    def add_best_solution(self, new_best_solution : float):
        self.run.best_len_history.append(new_best_solution)

    def set_best(self, best_path : int, best_path_len : float):
        self.run.best_solution = (best_path, best_path_len)

    def set_finished(self):
        self.run.finished = True

    def _generate_id(self, dictionary : dict):
        existing_ids = list(dictionary.keys())
        if not existing_ids:
            return 1

        existing_ids.sort()
        return existing_ids[-1] + 1

    def run_init(self, seed, algorithm, params):
        self.run = AlgorithmRun(
            seed=seed,
            algorithm=algorithm,
            params=params,
        )

    def store(self):
        id = self._generate_id(self.run_history)
        self.run_history[id] = self.run
        return id

    def make_group(self, run_ids : list[int]):
        id = self._generate_id(self.run_groups)
        self.run_groups[id] = run_ids
        return id

    def delete_group(self, id : int):
        if id in self.run_groups:
            del self.run_groups[id]

