import numpy
import abc
import logging

from ant import Ant
from ant_map import Map, Place


class AntAlgorithm:
    """Common class for all ant optimization algorithms (Ant System, Ant Colony)."""

    def __init__(self, iterations: int):
        self.iterations = iterations
        self._current_iteration = 0
        self._best_path = None
        self._best_path_len = None

    @classmethod
    def tuples_to_places(
        self, places: list[tuple[str, tuple[float, float]]]
    ) -> list[Place]:
        return [Place(p[1], p[0]) for p in places]

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

    @abc.abstractmethod
    def make_step(self) -> bool:
        return False


class AntSystemCommon(AntAlgorithm):
    """Common class for algorithms based on Ant System - Ant System itself, Ant
    Colonny, which differs only in way how ants choose the next place.
    """

    def __init__(self, places: list[Place], iterations: int, vaporization: float):
        self.map : Map = Map(places, pheronomone_vaporization=vaporization)
        super().__init__(iterations)

    def make_step(self) -> bool:
        if self.is_finished:
            return False

        self._current_iteration += 1

        for a in self.ants:
            a.init()

        for a in self.ants:
            while not a.is_finished:
                a.make_step()

        pheronomone_update_matrix = numpy.zeros(
            [len(self.map.places), len(self.map.places)]
        )
        for a in self.ants:
            path_len = a.get_path_len()
            if self._best_path_len is None or path_len < self._best_path_len:
                self._best_path_len = path_len
                self._best_path = a.get_path()
                logging.info(
                    f"improvement: it={self.current_iteration}, "
                    f"len={self._best_path_len:g}, path={self._best_path}"
                )

            pheronomone_update_matrix += a.get_new_pheromone_matrix()
            a.reset()

        self.map.vaporize_pheromone()
        self.map.add_pheromone(pheronomone_update_matrix)

        return True


class AntSystem(AntSystemCommon):
    """Implementation of the Ant System algorithm (the most basic one)."""

    def __init__(
        self,
        places: list[Place],
        iterations: int,
        ant_amount: int,
        pheronome_w: float,
        visibility_w: float,
        vaporization: float,
    ):
        # It is important to call parent initialization first, because it
        # initializes the map for ants
        super().__init__(places, iterations, vaporization)
        self.ants = [
            Ant(
                self.map,
                next_place_choice_fn=Ant.ant_system_choice,
                pheromone_fn=Ant.ant_system_pheromone,
                pheromone_weight=pheronome_w,
                visibility_weight=visibility_w,
            )
            for _ in range(ant_amount)
        ]


class AntDensity(AntSystemCommon):
    """Implementation of the Ant System algorithm (the most basic one)."""

    def __init__(
        self,
        places: list[Place],
        iterations: int,
        ant_amount: int,
        pheronome_w: float,
        visibility_w: float,
        vaporization: float,
    ):
        # It is important to call parent initialization first, because it
        # initializes the map for ants
        super().__init__(places, iterations, vaporization)
        self.ants = [
            Ant(
                self.map,
                next_place_choice_fn=Ant.ant_system_choice,
                pheromone_fn=Ant.ant_density_pheromone,
                pheromone_weight=pheronome_w,
                visibility_weight=visibility_w,
            )
            for _ in range(ant_amount)
        ]


class AntQuantity(AntSystemCommon):
    """Implementation of the Ant System algorithm (the most basic one)."""

    def __init__(
        self,
        places: list[Place],
        iterations: int,
        ant_amount: int,
        pheronome_w: float,
        visibility_w: float,
        vaporization: float,
    ):
        # It is important to call parent initialization first, because it
        # initializes the map for ants
        super().__init__(places, iterations, vaporization)
        self.ants = [
            Ant(
                self.map,
                next_place_choice_fn=Ant.ant_system_choice,
                pheromone_fn=Ant.ant_quantity_pheromone,
                pheromone_weight=pheronome_w,
                visibility_weight=visibility_w,
            )
            for _ in range(ant_amount)
        ]


class AntColony(AntSystemCommon):
    """Implementation of Ant Colony algorithm."""

    def __init__(
        self,
        places: list[Place],
        iterations: int,
        ant_amount: int,
        pheronome_w: float,
        visibility_w: float,
        vaporization: float,
        exploitation_coef: float,
    ):
        super().__init__(places, iterations, vaporization)
        self.ants = [
            Ant(
                self.map,
                next_place_choice_fn=self.next_place_choice,
                pheromone_fn=Ant.ant_system_pheromone,
                pheromone_weight=pheronome_w,
                visibility_weight=visibility_w,
            )
            for _ in range(ant_amount)
        ]
        self.exploitation_coef = exploitation_coef


    def next_place_choice(self, ant: Ant):
        """Uses the fixed threshold for decision whether prefer exploitation
        or exploration.
        """

        return Ant.ant_colony_choice(ant, self.exploitation_coef)


class ElitistStrategy(AntAlgorithm):
    """Contains Elitist Strategy of Ant System algorithm.
    """

    def __init__(
        self,
        places: list[Place],
        iterations: int,
        ant_amount: int,
        pheronome_w: float,
        visibility_w: float,
        vaporization: float,
    ):
        super().__init__(iterations)
        self.map : Map = Map(places, pheronomone_vaporization=vaporization)
        self.ants = [
            Ant(
                self.map,
                next_place_choice_fn=Ant.ant_system_choice,
                pheromone_fn=Ant.ant_system_pheromone,
                pheromone_weight=pheronome_w,
                visibility_weight=visibility_w,
            )
            for _ in range(ant_amount)
        ]

    def make_step(self) -> bool:
        if self.is_finished:
            return False

        self._current_iteration += 1

        for a in self.ants:
            a.init()

        for a in self.ants:
            while not a.is_finished:
                a.make_step()

        pheronomone_update_matrix = numpy.zeros(
            [len(self.map.places), len(self.map.places)]
        )
        traversed_edge_count = numpy.zeros(
            [len(self.map.places), len(self.map.places)]
        )
        for a in self.ants:
            path_len = a.get_path_len()
            path = a.get_path()
            for i, p in enumerate(path):
                place_i = p
                next_place_i = path[i + 1 if i + 1 < len(path) else 0]
                traversed_edge_count[place_i][next_place_i] += 1
                traversed_edge_count[next_place_i][place_i] += 1

            if self._best_path_len is None or path_len < self._best_path_len:
                self._best_path_len = path_len
                self._best_path = a.get_path()
                logging.info(
                    f"improvement: it={self.current_iteration}, "
                    f"len={self._best_path_len:g}, path={self._best_path}"
                )

            pheronomone_update_matrix += a.get_new_pheromone_matrix()
            a.reset()

        for i, p in enumerate(self._best_path):
            place_i = p
            next_place_i = path[i + 1 if i + 1 < len(path) else 0]
            e = traversed_edge_count[place_i][next_place_i]
            pheronomone_update_matrix[place_i][next_place_i] += e * (self.ants[0].pheronome_deposit / self._best_path_len)
            pheronomone_update_matrix[next_place_i][place_i] += e * (self.ants[0].pheronome_deposit / self._best_path_len)

        self.map.vaporize_pheromone()
        self.map.add_pheromone(pheronomone_update_matrix)

        return True

class MinMaxAntSystem(AntAlgorithm):
    def __init__(
        self,
        places: list[Place],
        iterations: int,
        ant_amount: int,
        pheronome_w: float,
        visibility_w: float,
        vaporization: float,
        min_pheromone: float,
        max_pheromone: float,
    ):
        super().__init__(iterations)
        self.map : Map = Map(places, pheronomone_vaporization=vaporization)
        self.ants = [
            Ant(
                self.map,
                next_place_choice_fn=Ant.ant_system_choice,
                pheromone_fn=Ant.ant_system_pheromone,
                pheromone_weight=pheronome_w,
                visibility_weight=visibility_w,
            )
            for _ in range(ant_amount)
        ]
        self.min_pheromone = min_pheromone
        self.max_pheromone = max_pheromone

    def make_step(self) -> bool:
        if self.is_finished:
            return False

        self._current_iteration += 1

        for a in self.ants:
            a.init()

        for a in self.ants:
            while not a.is_finished:
                a.make_step()

        pheronomone_update_matrix = numpy.zeros(
            [len(self.map.places), len(self.map.places)]
        )
        for a in self.ants:
            path_len = a.get_path_len()
            if self._best_path_len is None or path_len < self._best_path_len:
                self._best_path_len = path_len
                self._best_path = a.get_path()
                logging.info(
                    f"improvement: it={self.current_iteration}, "
                    f"len={self._best_path_len:g}, path={self._best_path}"
                )

                pheronomone_update_matrix = a.get_new_pheromone_matrix()
            a.reset()

        self.map.vaporize_pheromone()
        self.map.add_pheromone(pheronomone_update_matrix)
        self.map.pheromone_m = self.map.pheromone_m.clip(self.min_pheromone, self.max_pheromone)
        print(self.map.pheromone_m)
        return True