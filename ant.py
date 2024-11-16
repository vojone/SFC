import numpy

from map import Map

class Ant:
    def __init__(self, map : Map):
        self.tabu = []
        self.best_path = None
        self.best_path_len = None
        self.map : Map = map

    def position(self) -> int | None:
        return self.tabu[len(self.tabu) - 1] if len(self.tabu) > 0 else None

    def reset(self):
        self.tabu = []

    def move_to(self, place_index : int):
        self.tabu.append(place_index)

    def init(self):
        if self.position is not None:
            raise Exception("Unable to initialize ant, because ant is already placed somewhere!")

        initial_place = numpy.random.randint(0, len(self.map.places))
        self.move(initial_place)
