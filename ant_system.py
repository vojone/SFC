import numpy

from ant import Ant
from ant_map import Map, Place

class AntSystem:
    def __init__(self, places : list[tuple[str, tuple]], ant_amount : int, iterations : int, pheronome_w : float, visibility_w : float, vaporization : float):
        self.map = Map([ Place(p[1], p[0]) for p in places ], pheronomone_vaporization=vaporization)
        self.ants = [
            Ant(
                self.map,
                next_place_choice_fn=Ant.ant_system_choice,
                pheromone_weight=pheronome_w,
                visibility_weight=visibility_w
            ) for _ in range(ant_amount)
        ]
        self.iterations = iterations
        self._current_iteration = 0
        self._best_path = None
        self._best_path_len = None

    @property
    def current_iteration(self):
        return self._current_iteration

    @property
    def is_finished(self):
        return self._current_iteration >= self.iterations

    @property
    def best_path(self):
        return self._best_path

    @property
    def best_path_len(self):
        return self._best_path_len

    def start(self):
        self._best_path = None
        self._best_path_len = None
        self._current_iteration = 0

    def make_step(self) -> bool:
        if self.is_finished:
            return False

        self._current_iteration += 1

        for a in self.ants:
            a.init()

        for a in self.ants:
            while not a.is_finished:
                a.make_step()

        pheronomone_update_matrix = numpy.zeros([len(self.map.places), len(self.map.places)])
        for a in self.ants:
            path_len = a.get_path_len()
            if self._best_path_len is None or path_len < self._best_path_len:
                self._best_path_len = path_len
                self._best_path = a.get_path()
                print(f"Best path len: {self._best_path_len} Best path: {self._best_path}")

            pheronomone_update_matrix += a.get_new_pheromone_matrix()
            a.reset()

        self.map.vaporize_pheromone()
        self.map.add_pheromone(pheronomone_update_matrix)

        return True

