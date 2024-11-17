import numpy
from scipy.spatial import distance_matrix

class Place:
    def __init__(self, coords: tuple[float], name: str | None = None):
        self.coords = coords
        self.name = name

class Map:
    def __init__(self, places: list[Place], initial_pheromone : float = 1.0):
        self.places : list[Place] = places

        places_coords = [ p.coords for p in self.places ]

        # With p=2 we are using euclidean distances of places
        self.distance_m = numpy.array(distance_matrix(places_coords, places_coords, p=2))

        # Initial pheromone on the each path
        self.pheromone_m = numpy.full([len(places), len(places)], initial_pheromone)
