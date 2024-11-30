from dataclasses import dataclass, field

@dataclass
class AlgorithmRun:
    seed : int
    name : str | None = None
    algorithm : str | None = None
    finished : bool = False
    params : dict = field(default_factory=lambda: {})
    total_time : float = 0.0
    best_solution : tuple[list, float] = None
    best_len_history : list[list[float]] = field(default_factory=lambda: [])
    group : str | None = None

@dataclass
class RunGroup:
    name : str | None = None
    runs : list[int] = field(default_factory=lambda: [])

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
        self.run_history[id].name = f"#{id}"
        return id

    def make_group(self, run_ids : list[int], name : str | None = None):
        id = self._generate_id(self.run_groups)
        self.run_groups[id] = RunGroup(f"G#{id}" if name is None else name, run_ids)
        for i in run_ids:
            self.run_history[i].group = id

        return id

    def delete_group(self, id : int):
        group = self.run_groups[id]
        for r in group.runs:
            self.run_history[r].group = None

        del self.run_groups[id]

    def delete_group_with_runs(self, id : int):
        group = self.run_groups[id]
        for r in group.runs:
            del self.run_history[r]

        del self.run_groups[id]

    def rename_group(self, id : int, new_name : str):
        group = self.run_groups[id]
        for r in group.runs:
            self.run_history[r].group = new_name

        self.run_groups[id] = new_name

    def delete_run(self, id : int):
        run = self.run_history[id]
        if run.group is not None:
            self.run_groups[run.group].runs.remove(id)

        del self.run_history[id]

    def rename_run(self, id : int, new_name : str):
        run = self.run_history[id]
        run.name = new_name
