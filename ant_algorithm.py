# ant_algorithm.py
# Contains classes with implementations of ant algorithm
# Author: Vojtěch Dvořák (xdvora3o)

import numpy
import abc
import logging

from ant import Ant
from ant_map import Map, Place


class AntAlgorithm:
    """Base class for all ant optimization algorithms. Ants and map has to be
    initialized by subclasses.
    """

    def __init__(
        self,
        iterations: int,
        places: list[Place],
        vaporization: float,
        ant_amount: int,
        pheromone_w: float,
        visibility_w: float
    ):
        self.iterations = iterations
        self._current_iteration = 0
        self._best_path: None | list = None
        self._best_path_len: None | list = None
        self.map = self.create_map(places, vaporization)
        self.ants = self.create_ants(ant_amount, pheromone_w, visibility_w)

    @classmethod
    def tuples_to_places(
        self, coords: list[tuple[float, float]], names: list[str] = None
    ) -> list[Place]:
        """Provides conversion of list with coords and list with place
        names to list with Place objects.
        """

        assert len(coords) == len(names) or names is None
        result = [
            Place(c, names[i]) if names is not None else Place(c) for i, c in enumerate(coords)
        ]
        return result

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
        """Intializes state of the instance of algorithm."""

        self._best_path = None
        self._best_path_len = None
        self._current_iteration = 0

    def init_ants(self):
        """Prepares ants for the upcoming step (iteration)."""

        for a in self.ants:
            a.init()

    def release_ants(self):
        """Ants are forced to find their path. Includes computation of one
        iteration of the ant algorithm.
        """

        for a in self.ants:
            while not a.is_finished:
                a.make_step()

    @abc.abstractmethod
    def create_map(self, places: list[Place], vaporization: float) -> Map:
        """Instantiates the map."""
        pass

    @abc.abstractmethod
    def create_ants(
        self,
        ant_amount: int,
        pheromone_w: float,
        visibility_w: float
    ) -> list[Ant]:
        """Instantiates ants with specified params."""
        pass

    @abc.abstractmethod
    def make_step(self) -> bool:
        """Executes the one iteration of the algorithm."""
        pass


class AntSystemCommon(AntAlgorithm):
    """Common class for algorithms based on Ant System - Ant System itself, Ant
    Colony, which differs only in way how ants choose the next place.
    """

    def __init__(
        self,
        iterations: int,
        places: list[Place],
        vaporization: float,
        ant_amount: int,
        pheromone_w: float,
        visibility_w: float
    ):
        super().__init__(
            iterations,
            places,
            vaporization,
            ant_amount,
            pheromone_w,
            visibility_w
        )

    def create_map(self, places, vaporization) -> Map:
        return Map(places, pheronomone_vaporization=vaporization)

    def create_ants(
        self,
        ant_amount,
        pheromone_w,
        visibility_w,
    ) -> list[Ant]:
        return [
            Ant(
                self.map,
                next_place_choice_fn=Ant.ant_system_choice,
                pheromone_fn=Ant.ant_system_pheromone,
                pheromone_weight=pheromone_w,
                visibility_weight=visibility_w,
            )
            for _ in range(ant_amount)
        ]

    def make_step(self) -> bool:
        if self.is_finished:
            return False

        self._current_iteration += 1

        # Place ants to the random places on the map
        self.init_ants()

        # Ants will find the path through the map
        self.release_ants()

        # Prepare empty matrix with the new pheromone
        pheronomone_update_matrix = numpy.zeros(
            [len(self.map.places), len(self.map.places)]
        )

        # Process the result of each ant
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
        super().__init__(
            iterations,
            places,
            vaporization,
            ant_amount,
            pheronome_w,
            visibility_w,
        )


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
        super().__init__(
            iterations,
            places,
            vaporization,
            ant_amount,
            pheronome_w,
            visibility_w,
        )

    def create_ants(
        self,
        ant_amount,
        pheromone_w,
        visibility_w,
    ) -> list[Ant]:
        return [
            Ant(
                self.map,
                next_place_choice_fn=Ant.ant_system_choice,
                pheromone_fn=Ant.ant_density_pheromone,
                pheromone_weight=pheromone_w,
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
        super().__init__(
            iterations,
            places,
            vaporization,
            ant_amount,
            pheronome_w,
            visibility_w,
        )

    def create_ants(
        self,
        ant_amount,
        pheromone_w,
        visibility_w,
    ) -> list[Ant]:
        return [
            Ant(
                self.map,
                next_place_choice_fn=Ant.ant_system_choice,
                pheromone_fn=Ant.ant_quantity_pheromone,
                pheromone_weight=pheromone_w,
                visibility_weight=visibility_w,
            )
            for _ in range(ant_amount)
        ]


class AntColony(AntSystemCommon):
    """Implementation of Ant Colony algorithm. It is possible to specify
    exploitation coeficient which can be used for balancing exploration and
    exploitation of the solution space.
    """

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
        super().__init__(
            iterations,
            places,
            vaporization,
            ant_amount,
            pheronome_w,
            visibility_w,
        )
        self.exploitation_coef = exploitation_coef

    def create_ants(
        self,
        ant_amount,
        pheromone_w,
        visibility_w,
    ) -> list[Ant]:
        return [
            Ant(
                self.map,
                next_place_choice_fn=self.ant_colony_choice,
                pheromone_fn=Ant.ant_system_pheromone,
                pheromone_weight=pheromone_w,
                visibility_weight=visibility_w,
            )
            for _ in range(ant_amount)
        ]

    def ant_colony_choice(self, ant: Ant):
        """Uses the fixed threshold for decision whether prefer exploitation
        or exploration.
        """

        return Ant.ant_colony_choice(ant, self.exploitation_coef)


class ElitistStrategy(AntSystemCommon):
    """Contains Elitist Strategy of Ant System algorithm."""

    def __init__(
        self,
        places: list[Place],
        iterations: int,
        ant_amount: int,
        pheronome_w: float,
        visibility_w: float,
        vaporization: float,
    ):
        super().__init__(
            iterations,
            places,
            vaporization,
            ant_amount,
            pheronome_w,
            visibility_w,
        )

    # Intentionally hides the AntSystemCommon.make_skep
    def make_step(self) -> bool:
        if self.is_finished:
            return False

        self._current_iteration += 1

        self.init_ants()
        self.release_ants()

        pheronomone_update_matrix = numpy.zeros(
            [len(self.map.places), len(self.map.places)]
        )
        traversed_edge_count = numpy.zeros([len(self.map.places), len(self.map.places)])
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

        # Put more pheronome to the best path
        for i, p in enumerate(self._best_path):
            place_i = p
            next_place_i = path[i + 1 if i + 1 < len(path) else 0]
            e = traversed_edge_count[place_i][next_place_i]
            pheronomone_update_matrix[place_i][next_place_i] += e * (
                self.ants[0].pheronome_deposit / self._best_path_len
            )
            pheronomone_update_matrix[next_place_i][place_i] += e * (
                self.ants[0].pheronome_deposit / self._best_path_len
            )

        self.map.vaporize_pheromone()
        self.map.add_pheromone(pheronomone_update_matrix)

        return True


class MinMaxAntSystem(AntSystemCommon):
    """Min-Max ant algorithm. Minimum and maximum of pheromone on paths are
    specified to avoid the explosion of pheromone on some paths and the new
    pheromone is produced only with ant which found the best path.
    """

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
        super().__init__(
            iterations,
            places,
            vaporization,
            ant_amount,
            pheronome_w,
            visibility_w,
        )
        self.min_pheromone = min_pheromone
        self.max_pheromone = max_pheromone

    def make_step(self) -> bool:
        if self.is_finished:
            return False

        self._current_iteration += 1

        self.init_ants()
        self.release_ants()

        pheronomone_update_matrix = None
        for a in self.ants:
            path_len = a.get_path_len()
            if self._best_path_len is None or path_len < self._best_path_len:
                self._best_path_len = path_len
                self._best_path = a.get_path()
                logging.info(
                    f"improvement: it={self.current_iteration}, "
                    f"len={self._best_path_len:g}, path={self._best_path}"
                )

                # IMPORTANT: There is a difference with AntSystem
                # Only the ant which found the best path should produce the pheromone
                pheronomone_update_matrix = a.get_new_pheromone_matrix()
            a.reset()

        self.map.vaporize_pheromone()
        if pheronomone_update_matrix is not None:
            self.map.add_pheromone(pheronomone_update_matrix)

        # Clipping of pheromone on the edges
        self.map.pheromone_m = self.map.pheromone_m.clip(
            self.min_pheromone, self.max_pheromone
        )
        return True
