import tkinter

import sys
import os
import json
import csv
import threading
import logging
import numpy
import numpy.typing
import time

import ant_algorithm as alg
from gui import GUI


def get_seed():
    return numpy.random.randint(1, int(1e9))


class AlgorithmRunner:
    def __init__(self, algorithm: alg.AntAlgorithm, gui: GUI | None, done_callback):
        self.algorithm = algorithm
        self.gui = gui
        self.thread = threading.Thread(target=self.runner)
        self.sleep_event = threading.Event()
        self.finish_task_event = threading.Event()
        self.run_event = threading.Event()
        self.terminated_event = threading.Event()
        self.done_callback = done_callback
        self.total_time = 0.0

    @property
    def is_alive(self):
        return self.thread.is_alive()

    def start(self):
        self.thread.start()

    def stop(self):
        self.run_event.clear()

    def make_step(self):
        self.sleep_event.set()

    def run(self):
        self.run_event.set()
        self.sleep_event.set()

    def wait_for_result(self):
        self.finish_task_event.wait()
        self.finish_task_event.clear()

    def terminate(self):
        self.run_event.clear()
        self.terminated_event.set()
        self.sleep_event.set()
        while self.thread.is_alive():
            self.gui.root.update()

        self.thread.join()

    def runner(self):
        while not self.terminated_event.is_set():
            self.sleep_event.wait()
            self.sleep_event.clear()
            if self.terminated_event.is_set():
                break

            start_t = time.time()
            if self.run_event.is_set():
                while self.run_event.is_set() and self.algorithm.make_step():
                    self.total_time += time.time() - start_t
                    if self.gui is not None:
                        self.gui.update_speed(self.algorithm.current_iteration, self.total_time)
                        self.gui.update_best_path(self.algorithm.best_path_len)
                        self.gui.var_iterations.set(self.algorithm.current_iteration)
                    start_t = time.time()
            else:
                self.algorithm.make_step()
                self.total_time += time.time() - start_t
                if self.gui is not None:
                    self.gui.update_speed(self.algorithm.current_iteration, self.total_time)
                    self.gui.update_best_path(self.algorithm.best_path_len)
                    self.gui.var_iterations.set(self.algorithm.current_iteration)

            if not self.terminated_event.is_set():
                self.done_callback(not self.algorithm.is_finished)
            self.finish_task_event.set()


class App:
    ALGORITHM_CLASSES = {
        "Ant System": alg.AntSystem,  # Default
        "Ant Density": alg.AntDensity,
        "Ant Quantity": alg.AntQuantity,
        "Ant Colony": alg.AntColony,
        "Elitist Strategy": alg.ElitistStrategy,
        "Min-Max Ant System": alg.MinMaxAntSystem,
    }

    TEMRINAL_LOG_FORMAT_STR = "%(levelname)s: %(message)s"

    def __init__(
        self,
        data_filepath: str | None,
        seed: int | None = None,
        has_gui: bool = True,
        logging_enabled: str = True,
    ):
        self.data_filepath = data_filepath

        self.data: list | None = None
        self.algorithm: alg.AntAlgorithm | None = None
        self.algorithm_runner: AlgorithmRunner | None = None
        self.algorithm_class = None
        self.run_jobid = None

        self.best_solution = None
        self.to_draw = {}

        default_algorithm_name = list(self.ALGORITHM_CLASSES.keys())[0]
        self.algorithm_class = self.ALGORITHM_CLASSES[default_algorithm_name]
        self.current_params = {}
        self.total_iterations = 0
        self.seed = seed

        if has_gui:
            logger = None
            if logging_enabled:
                logging.basicConfig(
                    level=logging.INFO, format=self.TEMRINAL_LOG_FORMAT_STR
                )
                logger = logging.getLogger()

            self.gui = GUI(logger)
            self.gui.set_quit_fn(self._quit)
            self.gui.var_opened_file.set(
                os.path.basename(data_filepath)
                if data_filepath is not None
                else "No file opened"
            )
            self.gui.button_open_file.configure(command=self._open_file)
            self.gui.button_run.configure(command=self._run)
            self.gui.button_stop.configure(command=self._stop)
            self.gui.button_stop["state"] = "disabled"
            self.gui.button_save.configure(command=self._save)
            self.gui.button_restore.configure(command=self._restore)

            self.gui.button_step.bind("<ButtonPress-1>", self._step_press)
            self.gui.button_step.bind("<ButtonRelease-1>", self._step_release)
            self.gui.button_reset.configure(command=self._reset)

            self.gui.set_algorithm_options(list(self.ALGORITHM_CLASSES.keys()))
            self.gui.var_algorithm.trace_add("write", self._change_algorithm)

            default_algorithm_name = list(self.ALGORITHM_CLASSES.keys())[0]
            self.gui.update_params(self.gui.ALGORIHTM_PARAMS[default_algorithm_name])

            self.gui.checkbox_pheromone.configure(command=self._toggle_pheromone)
            self.gui.checkbox_best_path.configure(command=self._toggle_best_path)

            self.gui.on_show_place_names = self._toggle_place_names
            self.gui.on_show_distances = self._toggle_best_path
            self.gui.on_show_pheromone_amount = self._toggle_pheromone
            self.gui.on_use_custom_seed = self._use_seed

            self.gui.on_save_params = self._save_params_to_file
            self.gui.on_save_params_with_seed = self._save_params_with_seed_to_file
            self.gui.load_params_cb = self._load_params

            self._toggle_best_path()

    @property
    def has_gui(self):
        return self.gui is not None

    def _quit(self):
        if self.algorithm_runner is not None:
            self.algorithm_runner.terminate()

        if self.has_gui:
            self.gui.root.quit()
            self.gui.root.destroy()

    def _load_params(self):
        filename = self.gui.open_window_params_file()
        if not filename:
            return

        fp = open(filename, mode="r")
        params_dict = json.load(fp)
        self.load_params(params_dict)
        if self.save_params():
            self.reset(reseed=("seed" not in params_dict))

    def load_params(self, params_dict: dict):
        if "algorithm" not in params_dict:
            raise Exception("missing parameter 'algorithm'")
        self.set_algorithm(params_dict["algorithm"])

        if "seed" in params_dict:
            self.seed = params_dict["seed"]
            if self.has_gui:
                self.gui.var_seed.set(params_dict["seed"])

        for param_name in params_dict:
            if param_name in ["seed", "algorithm"]:
                continue

            if param_name not in self.current_params:
                raise Exception(f"unrecognized param '{param_name}'")

            self.current_params[param_name] = params_dict[param_name]
            if self.has_gui:
                self.gui.param_dict[param_name][0].set(params_dict[param_name])
                if param_name not in self.current_params:
                    raise Exception(f"unrecognized param '{param_name}'")

    def _use_seed(self):
        self.seed = self.gui.var_seed.get()
        self.reset(reseed=False)

    def _save(self):
        if not self.save_params():
            return

        logging.info("Changes of params succesfully saved")
        self.reset()
        self.gui.button_save["state"] = "disabled"
        self.gui.button_restore["state"] = "disabled"

    def _restore(self):
        self.restore_params()
        logging.info("Changes of params were restored")
        self.reset()
        self.gui.button_save["state"] = "disabled"
        self.gui.button_restore["state"] = "disabled"

    def _change_algorithm(self, *args, **kwargs):
        algorithm_name = self.gui.var_algorithm.get()
        self.set_algorithm(algorithm_name)
        self.save_params()
        self.reset()
        print(f"Algorithm changed to {self.gui.var_algorithm.get()}")

    def _algorithm_cb(self, continues: bool):
        self.best_solution = (self.algorithm.best_path, self.algorithm.best_path_len)

        if self.algorithm.is_finished:
            logging.info(
                f"Finished in it={self.algorithm.current_iteration}, best: "
                f"len={self.best_solution[1]:g}, path={self.best_solution[0]}"
            )

        if self.has_gui:
            self.gui.redraw_canvas(self.to_draw)
            self.gui.button_stop["state"] = "disabled"
            if not continues:
                self.gui.button_step["state"] = "disabled"
                self.gui.button_run["state"] = "disabled"
                self.gui.set_finished_status()
            else:
                self.gui.button_step["state"] = "normal"
                self.gui.button_run["state"] = "normal"
                self.gui.set_paused_status()

    def _stop(self):
        if self.algorithm_runner is not None:
            self.algorithm_runner.stop()

    def _reset(self):
        self.reset()

    def _step_release(self, *args, **kwargs):
        if self.run_jobid is not None:
            self.gui.root.after_cancel(self.run_jobid)
        self.run_jobid = None
        if self.algorithm_runner is None:
            return

        self.algorithm_runner.stop()

    def _step_press(self, *args, **kwargs):
        STEP_BUTTON_RUN_MS = 600

        if self.algorithm_runner is None:
            return

        self.gui.set_running_status()
        self.algorithm_runner.make_step()
        self.run_jobid = self.gui.root.after(STEP_BUTTON_RUN_MS, self._run)

    def _run(self):
        if self.algorithm_runner is None:
            return


        self.gui.set_running_status()
        self.algorithm_runner.run()
        self.gui.button_stop["state"] = "enabled"
        self.gui.button_run["state"] = "disabled"

    def _open_file(self):
        self.data_filepath = self.gui.open_window_data_file()
        if self.data_filepath is None or self.data_filepath == "":
            return

        data_filename = os.path.basename(self.data_filepath)
        self.gui.var_opened_file.set(data_filename)
        self.load_data()
        self._reset()

    def _toggle_pheromone(self):
        if self.gui.var_pheronomone.get() == 1:
            self.add_to_draw("pheromone", self.draw_pheromone)
        else:
            self.remove_to_draw("pheromone")

    def _toggle_best_path(self):
        if self.gui.var_best_path.get() == 1:
            self.add_to_draw("best_path", self.draw_best_path)
        else:
            self.remove_to_draw("best_path")

    def _toggle_place_names(self):
        if self.gui.var_show_place_names.get() == 1:
            self.add_to_draw("place_names", self.draw_data_names)
        else:
            self.remove_to_draw("place_names")

    def _save_params_to_file(self):
        filename = self.gui.open_window_save_params()
        if not filename:
            return

        fp = open(filename, mode="w")
        params = self.make_params_dict()
        fp.write(json.dumps(params, indent=4))
        fp.close()

    def _save_params_with_seed_to_file(self):
        filename = self.gui.open_window_save_params(custom_str="-seed")
        if not filename:
            return

        fp = open(filename, mode="w")
        params = self.make_params_dict()
        params["seed"] = self.seed
        fp.write(json.dumps(params, indent=4))
        fp.close()

    def add_to_draw(self, name: str, draw_fn):
        self.to_draw[name] = draw_fn
        self.gui.redraw_canvas(self.to_draw)

    def remove_to_draw(self, name: str):
        if name not in self.to_draw:
            return

        del self.to_draw[name]
        self.gui.redraw_canvas(self.to_draw)

    def load_data(self):
        def get_data(data_tuple : tuple):
            if len(data_tuple) == 2:
                return (float(data_tuple[0]), float(data_tuple[1]))
            elif len(data_tuple) == 3:
                return (float(data_tuple[1]), float(data_tuple[2]))
            else:
                raise ValueError("invalid format")

        def get_data_name(data_tuple : tuple):
            if len(data_tuple) == 2:
                return None
            else:
                return data_tuple[0]

        try:
            fp = open(self.data_filepath, "r")
        except FileNotFoundError as e:
            logging.error(f"Error while opening data file: {e}")
            return

        try:
            raw_data = json.load(fp)["data"]
        except json.decoder.JSONDecodeError:
            try:
                file_content = fp.read()
                raw_data = csv.reader(file_content.splitlines())
            except:
                logging.error(
                    f"Unable to parse the input file '{self.data_filepath}'!"
                    "Expected proper JSON or CSV format."
                )
        try:
            self.data = [ get_data(d) for d in raw_data ]
            self.data_names = [ get_data_name(d) for d in raw_data ]
        except ValueError:
            logging.error("Invalid format of the input file!")
            return

        self.to_draw["data"] = self.draw_data
        self.reset()

    def set_algorithm(self, algorithm_name: str):
        self.algorithm_class = self.ALGORITHM_CLASSES[algorithm_name]
        if self.has_gui:
            self.gui.update_params(self.gui.ALGORIHTM_PARAMS[algorithm_name])

    def start(self):
        self.save_params()
        if self.data_filepath is not None:
            self.load_data()

    def reset(self, reseed: bool = True):
        if not self.data:
            return

        if self.algorithm_runner is not None and self.algorithm_runner.is_alive:
            self.algorithm_runner.terminate()
        if self.has_gui:
            self.gui.clear_log()
            self.gui.disable_speed_label()
            if reseed and self.gui.var_fixed_seed.get() == 0:
                self.seed = get_seed()
                self.gui.var_seed.set(self.seed)

        numpy.random.seed(self.seed)
        self.best_solution = None
        self.algorithm_init()
        if self.has_gui:
            self.gui.set_paused_status()
            self.gui.reset_best_path()
            self.gui.var_iterations.set(self.algorithm.current_iteration)
            self.gui.button_stop["state"] = "disabled"
            self.gui.button_step["state"] = "normal"
            self.gui.button_run["state"] = "normal"
            self.gui.redraw_canvas(self.to_draw)

    def save_params(self) -> bool:
        if not self.has_gui:
            return True

        if not self.gui.param_validate():
            return False

        self.current_params.clear()
        for p in self.gui.param_dict:
            self.current_params[p] = self.gui.param_dict[p][0].get()

        self.total_iterations = self.gui.param_dict["iterations"][0].get()
        self.gui.var_total_iterations.set(self.total_iterations)
        self.gui.param_stored()

        return True

    def restore_params(self):
        if not self.has_gui:
            return

        for p in self.gui.param_dict:
            self.gui.param_dict[p][0].set(self.current_params[p])
        self.gui.param_stored()

    def make_params_dict(self) -> dict:
        result = {}
        result["algorithm"] = self.gui.var_algorithm.get()
        for p in self.current_params:
            result[p] = self.current_params[p]

        return result

    def algorithm_init(self):
        if self.has_gui:
            stored_params_str = f"initializing with seed={self.seed}"
            for p in self.current_params:
                stored_params_str += f", {p}={self.current_params[p]}"
            logging.info(stored_params_str)

        self.algorithm = self.algorithm_class(
            alg.AntAlgorithm.tuples_to_places(self.data, self.data_names),
            **self.current_params,
        )
        self.algorithm.start()

        if self.has_gui:
            logging.info(f"'{self.gui.var_algorithm.get()}' initialized")

        self.algorithm_runner = AlgorithmRunner(
            self.algorithm, self.gui, self._algorithm_cb
        )
        self.algorithm_runner.start()

    def draw_best_path(self):
        if not self.has_gui or self.best_solution is None:
            return

        best_path = self.best_solution[0]
        self.gui.draw_path(
            best_path,
            self.algorithm.map.places,
            bool(self.gui.var_show_distances.get()),
            self.algorithm.map.distance_m,
        )

    def draw_pheromone(self):
        if not self.has_gui:
            return

        self.gui.draw_matrix_data(
            self.algorithm.map.pheromone_m,
            self.algorithm.map.places,
            bool(self.gui.var_show_pheromone_amount.get()),
        )

    def draw_data(self):
        if self.data is not None:
            self.gui.draw_data(self.data)

    def draw_data_names(self):
        if self.data is not None:
            self.gui.draw_data_names(self.data, self.data_names)


if __name__ == "__main__":
    app = App(sys.argv[1] if len(sys.argv) > 1 else None)
    app.start()
    tkinter.mainloop()
