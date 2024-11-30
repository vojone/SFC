import threading
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


class AlgorithmRunner:
    def __init__(
        self,
        algorithm: alg.AntAlgorithm,
        gui: GUI | None,
        algorithm_stats: AlgorithmStats,
        on_algorithm_done,
        iteration_done_update
    ):
        self.algorithm = algorithm
        self.gui = gui
        self.thread = threading.Thread(target=self.runner)
        self.sleep_event = threading.Event()
        self.finish_task_event = threading.Event()
        self.run_event = threading.Event()
        self.terminated_event = threading.Event()
        self.on_algorithm_done = on_algorithm_done
        self.iteration_done_update = iteration_done_update
        self.algorithm_stats = algorithm_stats

    @property
    def is_alive(self):
        return self.thread.is_alive()

    def start(self):
        self.thread.start()

    def stop(self):
        self.run_event.clear()

    def make_step(self):
        self.sleep_event.set()

    def run(self):
        self.run_event.set()
        self.sleep_event.set()

    def wait_for_result(self):
        self.finish_task_event.wait()
        self.finish_task_event.clear()

    def terminate(self):
        self.run_event.clear()
        self.terminated_event.set()
        self.sleep_event.set()
        while self.thread.is_alive():
            self.gui.root.update()

        self.thread.join()

    def runner(self):
        while not self.terminated_event.is_set():
            self.sleep_event.wait()
            self.sleep_event.clear()
            if self.terminated_event.is_set():
                break

            start_t = time.time()
            if self.run_event.is_set():
                while self.run_event.is_set() and self.algorithm.make_step():
                    self.algorithm_stats.run.total_time += time.time() - start_t
                    self.iteration_done_update()
                    start_t = time.time()
            else:
                self.algorithm.make_step()
                self.algorithm_stats.run.total_time += time.time() - start_t
                self.iteration_done_update()

            if not self.terminated_event.is_set():
                self.on_algorithm_done(not self.algorithm.is_finished)
            self.finish_task_event.set()

