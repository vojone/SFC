import numpy

from ant import Ant
from ant_map import Map, Place

class AntSystem:
    def __init__(self, places : list[tuple[str, tuple]], ant_amount : int, n_cycles : int, pheronome_w : float, visibility_w : float, vaporization : float):
        self.map = Map([ Place(p[1], p[0]) for p in places ], pheronomone_vaporization=vaporization)
        self.ants = [
            Ant(
                self.map,
                next_place_choice_fn=Ant.ant_system_choice,
                pheromone_weight=pheronome_w,
                visibility_weight=visibility_w
            ) for _ in range(ant_amount)
        ]
        self.n_cycles = n_cycles

    def run(self):
        best_path = None
        best_path_len = None
        for _ in range(self.n_cycles):
            for a in self.ants:
                a.init()

            for a in self.ants:
                while not a.is_finished:
                    a.make_step()

            pheronomone_update_matrix = numpy.zeros([len(self.map.places), len(self.map.places)])
            for a in self.ants:
                path_len = a.get_path_len()
                if best_path_len is None or path_len < best_path_len:
                    best_path_len = path_len
                    best_path = a.get_path()
                    print(f"Best path len: {best_path_len} Best path: {best_path}")

                pheronomone_update_matrix += a.get_new_pheromone_matrix()
                a.reset()

            self.map.vaporize_pheromone()
            self.map.add_pheromone(pheronomone_update_matrix)

        return best_path


