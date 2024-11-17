import numpy
import numpy.typing
from scipy.spatial import distance_matrix

class Place:
    def __init__(self, coords: tuple[float], name: str | None = None):
        self.coords = coords
        self.name = name

class Map:
    def __init__(self, places: list[Place], initial_pheromone : float = 1.0, pheronomone_vaporization : float = 0.2):
        self.places : list[Place] = places

        places_coords = [ p.coords for p in self.places ]

        # With p=2 we are using euclidean distances of places
        self.distance_m = numpy.array(distance_matrix(places_coords, places_coords, p=2))

        # Initial pheromone on the each path
        self.pheromone_m = numpy.full([len(places), len(places)], initial_pheromone)
        self.pheronomone_vaporization = pheronomone_vaporization

    def vaporize_pheromone(self) -> numpy.typing.NDArray:
        self.pheromone_m *= (1 - self.pheronomone_vaporization)
        return self.pheromone_m

    def add_pheromone(self, new_pheromone : numpy.typing.NDArray) -> numpy.typing.NDArray:
        if self.pheromone_m.size != new_pheromone.size():
            raise Exception("Size of new pheromone matrix and ph. map do not match!")

        # Add new pheromone
        self.pheromone_m += new_pheromone

        return self.pheromone_m