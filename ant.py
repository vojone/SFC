from __future__ import annotations

import numpy
import numpy.typing
from ant_map import Map


class Ant:
    def __init__(
        self,
        map: Map,
        next_place_choice_fn,
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

    @property
    def position(self) -> int | None:
        return self.tabu[len(self.tabu) - 1] if len(self.tabu) > 0 else None

    @property
    def is_finished(self):
        return len(self.tabu) == len(self.map.places)

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

    def get_new_pheromone_matrix(self) -> numpy.typing.NDArray:
        if self.position is None:
            raise Exception("Ant is not initialized!")

        new_pheromone = numpy.zeros([len(self.map.places), len(self.map.places)])
        path_len = self.get_path_len()
        for i, place in enumerate(self.get_path()):
            next_place = self.tabu[i + 1] if i + 1 < len(self.tabu) else self.tabu[0]

            new_pheromone[place, next_place] = self.pheronome_deposit / path_len
            # Pheromone matrix is symmetric, so update also the corresponding element
            new_pheromone[next_place, place] = self.pheronome_deposit / path_len

        return new_pheromone

    def reset(self):
        self.tabu = []

    def reset_best(self):
        self.best_path = None
        self.best_path_len = None

    def move_to(self, place_index: int):
        self.tabu.append(place_index)

    def init(self):
        if self.position is not None:
            raise Exception("Ant is already placed somewhere!")

        initial_place = numpy.random.randint(0, len(self.map.places))
        self.move_to(initial_place)

    def make_step(self) -> bool:
        if self.position is None:
            raise Exception("Ant was not initialized!")

        if self.is_finished:
            return True

        next_place_index = self.next_place_choice_fn(self)
        self.move_to(next_place_index)

        return False
