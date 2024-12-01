# ant.py
# Contains class represeting an artificial ant for ant algorithms
# Author: Vojtěch Dvořák (xdvora3o)

from __future__ import annotations

import numpy
import numpy.typing
from ant_map import Map


class Ant:
    """Represents one artifical ant. The way how its chooses the next place
    where to go and how it distributes pheromone is specified by arguments.
    """

    def __init__(
        self,
        map: Map,
        next_place_choice_fn,
        pheromone_fn,
        pheromone_weight: int = 1,
        visibility_weight: int = 1,
        pheronome_deposit: float = 1.0,
    ):
        self.tabu = []
        self.map: Map = map
        self.pheromone_w = pheromone_weight
        self.visibility_w = visibility_weight
        self.pheronome_deposit = pheronome_deposit
        self.next_place_choice_fn = next_place_choice_fn
        self.pheromone_fn = pheromone_fn

    @classmethod
    def ant_system_pheromone(cls, ant : Ant) -> numpy.typing.NDArray:
        if ant.position is None:
            raise Exception("Ant is not initialized!")

        new_pheromone = numpy.zeros([len(ant.map.places), len(ant.map.places)])
        path_len = ant.get_path_len()
        for i, place in enumerate(ant.get_path()):
            next_place = ant.tabu[i + 1] if i + 1 < len(ant.tabu) else ant.tabu[0]
            new_pheromone[place, next_place] = ant.pheronome_deposit / path_len
            # Pheromone matrix is symmetric, so update also the corresponding element
            new_pheromone[next_place, place] = ant.pheronome_deposit / path_len

        return new_pheromone

    @classmethod
    def ant_density_pheromone(cls, ant : Ant) -> numpy.typing.NDArray:
        if ant.position is None:
            raise Exception("Ant is not initialized!")

        new_pheromone = numpy.zeros([len(ant.map.places), len(ant.map.places)])
        for i, place in enumerate(ant.get_path()):
            next_place = ant.tabu[i + 1] if i + 1 < len(ant.tabu) else ant.tabu[0]
            new_pheromone[place, next_place] = ant.pheronome_deposit
            # Pheromone matrix is symmetric, so update also the corresponding element
            new_pheromone[next_place, place] = ant.pheronome_deposit

        return new_pheromone

    @classmethod
    def ant_quantity_pheromone(cls, ant : Ant) -> numpy.typing.NDArray:
        if ant.position is None:
            raise Exception("Ant is not initialized!")

        new_pheromone = numpy.zeros([len(ant.map.places), len(ant.map.places)])
        for i, place in enumerate(ant.get_path()):
            next_place = ant.tabu[i + 1] if i + 1 < len(ant.tabu) else ant.tabu[0]
            edge_len = ant.map.distance_m[place, next_place]
            new_pheromone[place, next_place] = ant.pheronome_deposit / (edge_len + 1e-30)
            # Pheromone matrix is symmetric, so update also the corresponding element
            new_pheromone[next_place, place] = ant.pheronome_deposit / (edge_len + 1e-30)

        return new_pheromone

    @classmethod
    def ant_system_choice(cls, ant : Ant) -> int:
        probabilities = cls.ant_system_transition_probabilities(ant)
        place_indeces = numpy.arange(len(ant.map.places))
        next_place_index = numpy.random.choice(place_indeces, p=probabilities)
        return next_place_index

    @classmethod
    def ant_system_transition_probabilities(cls, ant : Ant) -> numpy.typing.NDArray:
        current_place_idx = ant.position
        place_probs = numpy.zeros(len(ant.map.places))
        for i, _ in enumerate(place_probs):
            if i in ant.tabu:
                continue

            dist = ant.map.distance_m[current_place_idx][i]
            p = ant.map.pheromone_m[current_place_idx][i]
            v = 1 / (dist + 1e-30) # Add very small number to distance to avoid divison by zero
            place_probs[i] = (p**ant.pheromone_w) * (v**ant.visibility_w)

        # Normalize probabilities to make valid (categorial) probability distribution
        place_probs = place_probs / numpy.sum(place_probs)
        return place_probs

    @classmethod
    def ant_colony_choice(cls, ant : Ant, threshold : float) -> int:
        r = numpy.random.random()
        if r > threshold:
            # Exploration
            return cls.ant_system_choice(ant)

        # Exploitation (r <= threshold)
        next_place_index = None
        next_place_quality = None
        current_place_idx = ant.position
        for i, _ in enumerate(ant.map.places):
            if i in ant.tabu:
                continue

            dist = ant.map.distance_m[current_place_idx][i]
            p = ant.map.pheromone_m[current_place_idx][i]
            v = 1 / (dist + 1e-30) # Add very small number to distance to avoid divison by zero
            quality = p * (v**ant.visibility_w)
            if next_place_quality is None or quality > next_place_quality:
                next_place_quality = quality
                next_place_index = i

        return next_place_index

    @property
    def position(self) -> int | None:
        return self.tabu[len(self.tabu) - 1] if len(self.tabu) > 0 else None

    @property
    def is_finished(self):
        return len(self.tabu) == len(self.map.places)

    def get_path(self):
        return self.tabu

    def get_path_len(self) -> float | None:
        if self.position is None:
            raise Exception("Ant is not initialized!")

        path_len = 0.0
        for i, place in enumerate(self.get_path()):
            next_place = self.tabu[i + 1] if i + 1 < len(self.tabu) else self.tabu[0]
            path_len += self.map.distance_m[place][next_place]

        return path_len

    def get_new_pheromone_matrix(self):
        return self.pheromone_fn(self)

    def reset(self):
        self.tabu = []

    def reset_best(self):
        self.best_path = None
        self.best_path_len = None

    def move_to(self, place_index: int):
        self.tabu.append(place_index)

    def init(self):
        """Puts the ant to the random place on the map."""

        if self.position is not None:
            raise Exception("Ant is already placed somewhere!")

        initial_place = numpy.random.randint(0, len(self.map.places))
        self.move_to(int(initial_place))

    def make_step(self) -> bool:
        """Chooses the next place the goes there. If it returns True the path
        was finished and ant does nothing.
        """

        if self.position is None:
            raise Exception("Ant was not initialized!")

        if self.is_finished:
            return True

        next_place_index = self.next_place_choice_fn(self)
        self.move_to(int(next_place_index))

        return False
