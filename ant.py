import numpy
import numpy.typing

from map import Map

class Ant:
    def __init__(self, map : Map, pheromone_weight : int = 1, visibility_weight : int = 1):
        self.tabu = []
        self.best_path = None
        self.best_path_len = None
        self.map : Map = map
        self.pheromone_w = pheromone_weight
        self.visibility_w = visibility_weight

    @property
    def position(self) -> int | None:
        return self.tabu[len(self.tabu) - 1] if len(self.tabu) > 0 else None

    @property
    def is_finished(self):
        return len(self.tabu) == len(self.map.places)

    def get_transition_probabilities(self) -> numpy.typing.ArrayLike:
        current_place_idx = self.position
        place_probs = numpy.zeros(len(self.map.places))
        for i in range(len(place_probs)):
            if i in self.tabu:
                continue

            dist = self.map.distance_m[current_place_idx][i]
            pheromone = self.map.pheromone_m[current_place_idx][i]
            visiblity = 1 / dist
            place_probs[i] = (pheromone**self.pheromone_w) * (visiblity**self.visibility_w)

        # Normalize probabilities to make valid (categorial) probability distribution
        place_probs = place_probs / numpy.sum(place_probs)
        return place_probs

    def get_path(self):
        return self.tabu

    def get_path_len(self) -> float | None:
        if self.position is None:
           return None

        path_len = 0.0
        for i in range(len(self.get_path())):
            place_i = self.tabu[i]
            next_place_i = self.tabu[i + 1] if i + 1 < len(self.tabu) else self.tabu[0]
            path_len += self.map.distance_m[place_i][next_place_i]

        return path_len

    def reset(self):
        self.tabu = []

    def reset_best(self):
        self.best_path = None
        self.best_path_len = None

    def move_to(self, place_index : int):
        self.tabu.append(place_index)

    def init(self):
        if self.position is not None:
            raise Exception("Unable to initialize ant, because ant is already placed somewhere!")

        initial_place = numpy.random.randint(0, len(self.map.places))
        self.move_to(initial_place)

    def update_best_path(self):
        if not self.is_finished:
            raise Exception("Cannot update best path, when the current path was not finished!")

        current_path = self.get_path()
        current_path_len = self.get_path_len()
        if self.best_path_len is None or self.best_path_len >= current_path_len:
            self.best_path = current_path
            self.best_path_len = current_path_len

        return self.best_path_len

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

