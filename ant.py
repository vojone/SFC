import numpy
import numpy.typing

from map import Map


class Ant:
    def __init__(
        self,
        map: Map,
        pheromone_weight: int = 1,
        visibility_weight: int = 1,
        pheronome_deposit: float = 1.0,
    ):
        self.tabu = []
        self.map: Map = map
        self.pheromone_w = pheromone_weight
        self.visibility_w = visibility_weight
        self.pheronome_deposit = pheronome_deposit

    @property
    def position(self) -> int | None:
        return self.tabu[len(self.tabu) - 1] if len(self.tabu) > 0 else None

    @property
    def is_finished(self):
        return len(self.tabu) == len(self.map.places)

    def get_transition_probabilities(self) -> numpy.typing.NDArray:
        current_place_idx = self.position
        place_probs = numpy.zeros(len(self.map.places))
        for i, _ in enumerate(place_probs):
            if i in self.tabu:
                continue

            dist = self.map.distance_m[current_place_idx][i]
            p = self.map.pheromone_m[current_place_idx][i]
            v = 1 / dist
            place_probs[i] = (p**self.pheromone_w) * (v**self.visibility_w)

        # Normalize probabilities to make valid (categorial) probability distribution
        place_probs = place_probs / numpy.sum(place_probs)
        return place_probs

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

        probabilities = self.get_transition_probabilities()
        place_indeces = numpy.arange(len(self.map.places))
        next_place_index = numpy.random.choice(place_indeces, p=probabilities)
        self.move_to(next_place_index)

        return False
