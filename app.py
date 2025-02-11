# app.py
# Implementation of Ant Algorithms application
# Author: Vojtěch Dvořák (xdvora3o)

import tkinter

import sys
import os
import json
import csv
import logging
import numpy
import numpy.typing
import time
import threading

import ant_algorithm as alg
from gui import GUI
from enum import Enum
from algorithm_stats import AlgorithmStats


def get_seed():
    """Generates random number of reasonable amount of digits which is
    used as a random seed for the execution. It does not matter that is is not
    properly random.
    """

    return numpy.random.randint(1, int(1e9))


class RunMode(Enum):
    """Run modes of algorithm runner."""

    STEP = 0 # Make steps
    RUN = 1 # Run until end (or it is paused interactively)


class AlgorithmRunner:
    """Class that wrapps thread which is responsible for execution of the
    algorithm."""

    def __init__(
        self,
        algorithm: alg.AntAlgorithm,
        gui: GUI | None,
        algorithm_stats: AlgorithmStats,
        on_algorithm_done,
        iteration_done_update
    ):
        # Algorithm instance
        self.algorithm = algorithm
        self.gui = gui
        # Running threads
        self.thread = threading.Thread(target=self.runner)
        # Sync events
        self.sleep_event = threading.Event()
        self.finish_task_event = threading.Event()
        self.run_event = threading.Event()
        self.terminated_event = threading.Event()
        # Callbacks
        self.on_algorithm_done = on_algorithm_done
        self.iteration_done_update = iteration_done_update
        self.algorithm_stats = algorithm_stats

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
        """Gracefully terminates the running thread. Running thread can be
        terminated only on the end of algorithm iteration.
        """

        self.run_event.clear()
        self.terminated_event.set()
        self.sleep_event.set()
        while self.thread.is_alive():
            self.gui.root.update()

        self.thread.join()

    def runner(self):
        """Main loop of the running thread."""

        while not self.terminated_event.is_set():
            self.sleep_event.wait()
            self.sleep_event.clear()
            if self.terminated_event.is_set():
                break

            start_t = time.time()
            if self.run_event.is_set():
                while self.run_event.is_set() and self.algorithm.make_step():
                    self.algorithm_stats.run.total_time += time.time() - start_t
                    self.iteration_done_update()
                    start_t = time.time()
            else:
                self.algorithm.make_step()
                self.algorithm_stats.run.total_time += time.time() - start_t
                self.iteration_done_update()

            if not self.terminated_event.is_set():
                self.on_algorithm_done(not self.algorithm.is_finished)
            self.finish_task_event.set()


class App:
    """Class wrapping the whole Ant Algorithms app. Connects GUI, with
    implementation of ant algorithms and AlgorithmRunner."""

    # Dictionary mapping algorithm names to corresponding classes
    ALGORITHM_CLASSES = {
        "Ant System": alg.AntSystem,  # Default
        "Ant Density": alg.AntDensity,
        "Ant Quantity": alg.AntQuantity,
        "Ant Colony": alg.AntColony,
        "Elitist Strategy": alg.ElitistStrategy,
        "Min-Max Ant System": alg.MinMaxAntSystem,
    }

    # Format of log messages in the terminal
    TERMINAL_LOG_FORMAT_STR = "%(levelname)s: %(message)s"

    # Are continues updates used by default?
    CONTINUOS_UPDATES_DEFAULT = False

    def __init__(
        self,
        data_filepath: str | None,
        seed: int | None = None,
        has_gui: bool = True,
        logging_enabled: str = True,
    ):
        self.data_filepath = data_filepath
        self.algorithm_stats = AlgorithmStats()

        self.data: list | None = None
        self.algorithm: alg.AntAlgorithm | None = None
        self.algorithm_runner: AlgorithmRunner | None = None
        self.algorithm_class: alg.AntAlgorithm | None = None
        self.run_jobid: str | None = None

        self.to_draw: dict = {}

        self.algorithm_name = list(self.ALGORITHM_CLASSES.keys())[0]
        self.algorithm_class = self.ALGORITHM_CLASSES[self.algorithm_name]
        self.run_mode: RunMode | None = None
        self.current_params: dict = {}
        self.total_iterations = 0
        self.seed = seed
        self.remaining_runs = 0

        # Configurate GUI if the app has gui
        if has_gui:
            logger = None
            if logging_enabled:
                logging.basicConfig(
                    level=logging.INFO, format=self.TERMINAL_LOG_FORMAT_STR
                )
                logger = logging.getLogger()

            self.gui = GUI(self.algorithm_stats, logger)

            # Set callback and default values
            self.gui.set_quit_fn(self._quit)
            self.gui.var_opened_file.set(
                os.path.basename(data_filepath)
                if data_filepath is not None
                else "No file opened"
            )
            self.gui.button_open_file.configure(command=self._open_file)
            self.gui.button_run.configure(command=self._user_run)
            self.gui.button_stop.configure(command=self._stop)
            self.gui.button_stop["state"] = "disabled"
            self.gui.button_save.configure(command=self._save)
            self.gui.button_restore.configure(command=self._restore)

            self.gui.button_step.bind("<ButtonPress-1>", self._step_press)
            self.gui.button_step.bind("<ButtonRelease-1>", self._step_release)
            self.gui.button_reset.configure(command=self._reset)

            self.gui.set_algorithm_options(list(self.ALGORITHM_CLASSES.keys()))
            self.gui.var_algorithm.trace_add("write", self._change_algorithm)

            self.gui.update_params(self.gui.ALGORIHTM_PARAMS[self.algorithm_name])

            self.gui.checkbox_pheromone.configure(command=self._toggle_pheromone)
            self.gui.checkbox_best_path.configure(command=self._toggle_best_path)

            self.gui.on_show_place_names = self._toggle_place_names
            self.gui.on_show_distances = self._toggle_best_path
            self.gui.on_show_pheromone_amount = self._toggle_pheromone
            self.gui.on_use_custom_seed = self._use_seed

            self.gui.on_save_params = self._save_params_to_file
            self.gui.on_save_params_with_seed = self._save_params_with_seed_to_file
            self.gui.load_params_cb = self._load_params

            # Disable all buttons by default until application is started
            self.gui.button_stop["state"] = "disabled"
            self.gui.button_step["state"] = "disabled"
            self.gui.button_run["state"] = "disabled"

            self.gui.var_continuous_updates.set(self.CONTINUOS_UPDATES_DEFAULT)
            # Display the best path by default
            self._toggle_best_path()


    @property
    def has_gui(self):
        return self.gui is not None


    def _quit(self):
        """Quits the app gracefully."""

        if self.algorithm_runner is not None:
            self.algorithm_runner.terminate()

        if self.has_gui:
            self.gui.root.quit()
            self.gui.root.destroy()


    def _load_params(self):
        """Load params callback."""

        # Open dialog
        filename = self.gui.open_window_params_file()
        if not filename:
            return

        # Load params to dicts
        fp = open(filename, mode="r")
        params_dict = json.load(fp)
        self.load_params(params_dict)
        if self.save_params():
            self.reset(reseed=("seed" not in params_dict))


    def _use_seed(self):
        """Use custom seed callback."""

        self.seed = self.gui.var_seed.get()
        self.reset(reseed=False)


    def _save(self):
        """Callback for saving changes of params. It also resets the execution.
        """

        if not self.save_params():
            return

        logging.info("Changes of params succesfully saved")
        self.reset()
        self.gui.button_save["state"] = "disabled"
        self.gui.button_restore["state"] = "disabled"


    def _restore(self):
        """Callback for restoring the params. It also resets the execution."""

        self.restore_params()
        logging.info("Changes of params were restored")
        self.reset()
        self.gui.button_save["state"] = "disabled"
        self.gui.button_restore["state"] = "disabled"


    def _change_algorithm(self, *args, **kwargs):
        """Callback for changing the type of algorithm. It also resets the
        execution and saves the params (which are set to the defalt values).
        """

        algorithm_name = self.gui.var_algorithm.get()
        self.set_algorithm(algorithm_name)
        logging.info(f"Algorithm changed to {self.gui.var_algorithm.get()}")
        self.save_params()
        self.reset()


    def _on_algorithm_iteration_done(self):
        """Function that is called when every iteration of algorithm is done.
        """

        solution_update = False
        if (len(self.algorithm_stats.run.best_len_history) == 0 or
            self.algorithm.best_path_len != self.algorithm_stats.run.best_len_history[-1]):
            solution_update = True

        self.algorithm_stats.add_best_solution(float(self.algorithm.best_path_len))
        if self.has_gui:
            self.gui.update_speed(
                self.algorithm.current_iteration,
                self.algorithm_stats.run.total_time
            )
            self.gui.update_best_path(self.algorithm.best_path_len)
            self.gui.var_iterations.set(self.algorithm.current_iteration)

        # Redraw canvas if there was any update and continuos updates are activated
        if solution_update:
            self.algorithm_stats.set_best(self.algorithm.best_path, self.algorithm.best_path_len)
            if self.has_gui and self.gui.var_continuous_updates.get():
                self.gui.redraw_canvas(self.to_draw)
                self.gui.update_history(self.algorithm_stats.run.best_len_history)


    def _on_algorithm_done(self, continues: bool):
        """Called when is portion of iterations executed - depends on which
        button was used for the execution."""

        def auto_rerun():
            """When there are multiple runs specified the next execution is
            start automatically.
            """
            self.reset()
            self._run()

        # Update statistics
        self.algorithm_stats.set_best(self.algorithm.best_path, self.algorithm.best_path_len)
        if self.algorithm.is_finished:
            self.algorithm_stats.set_finished()
            self.algorithm_stats.store()
            logging.info(
                f"Finished in it={self.algorithm.current_iteration}, best: "
                f"len={self.algorithm_stats.run.best_solution[1]:g}, "
                f"path={self.algorithm_stats.run.best_solution[0]}"
            )

        # Update GUI
        if self.has_gui:
            self.gui.redraw_canvas(self.to_draw)
            self.gui.button_stop["state"] = "disabled"
            self.gui.update_convergence(self.algorithm_stats.run.best_len_history)
            self.gui.update_history()

            if continues:
                self.gui.button_step["state"] = "normal"
                self.gui.button_run["state"] = "normal"
                self.gui.set_paused_status()
            else:
                self.remaining_runs -= 1
                self.gui.button_step["state"] = "disabled"
                self.gui.button_run["state"] = "disabled"
                self.gui.set_finished_status()
                if self.run_mode == RunMode.RUN and self.remaining_runs:
                    self.gui.root.after(0, auto_rerun)


    def _stop(self):
        if self.algorithm_runner is not None:
            self.algorithm_runner.stop()


    def _reset(self):
        self.reset()


    def _step_release(self, *args, **kwargs):
        if self.run_jobid is not None:
            # Button was released before run job was executed, so cancel it
            self.gui.root.after_cancel(self.run_jobid)
        self.run_jobid = None
        if self.algorithm_runner is None:
            return

        self.algorithm_runner.stop()


    def _step_press(self, *args, **kwargs):
        """Callback for step button. Step button can be used in two modes. If
        there is just single click only one iteration of lagoirthm is made. If
        If button is pressed for a whil
        """

        STEP_BUTTON_RUN_MS = 600

        if self.algorithm_runner is None:
            return
        if self.algorithm.is_finished:
            return

        self.run_mode = RunMode.STEP
        self.gui.set_running_status()
        self.algorithm_runner.make_step()
        # Schedule run event if step button is not released for a while
        self.run_jobid = self.gui.root.after(STEP_BUTTON_RUN_MS, self._run)


    def _user_run(self):
        if self.algorithm_runner is None:
            return

        self.run_mode = RunMode.RUN
        self.remaining_runs = self.gui.var_number_of_runs.get()
        self._run()


    def _run(self):
        self.gui.set_running_status()
        self.algorithm_runner.run()
        self.gui.button_stop["state"] = "enabled"
        self.gui.button_run["state"] = "disabled"


    def _open_file(self):
        """Callback to open the file with the data (places)."""

        self.data_filepath = self.gui.open_window_data_file()
        if self.data_filepath is None or self.data_filepath == "":
            return

        data_filename = os.path.basename(self.data_filepath)
        self.gui.var_opened_file.set(data_filename)
        self.load_data()


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
        """Callback for saving parameters (without seed)."""

        filename = self.gui.open_window_save_params()
        if not filename:
            return

        fp = open(filename, mode="w")
        params = self.make_params_dict()
        fp.write(json.dumps(params, indent=4))
        fp.close()


    def _save_params_with_seed_to_file(self):
        """Callback for saving parameters with seed."""

        filename = self.gui.open_window_save_params(custom_str="-seed")
        if not filename:
            return

        fp = open(filename, mode="w")
        params = self.make_params_dict()
        params["seed"] = self.seed
        fp.write(json.dumps(params, indent=4))
        fp.close()


    def load_params(self, params_dict: dict):
        """Loads parameters from dict to the app."""

        # Algorithm and seed are handled separately
        if "algorithm" not in params_dict:
            raise Exception("missing parameter 'algorithm'")
        self.set_algorithm(params_dict["algorithm"])
        self.save_params()

        if "seed" in params_dict:
            self.seed = params_dict["seed"]
            if self.has_gui:
                self.gui.var_seed.set(params_dict["seed"])

        # Load the rest of parameters
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


    def add_to_draw(self, name: str, draw_fn):
        """Adds draw function to the draw dictionary. This function will be
        called everytime the canvas is redrawn."""

        self.to_draw[name] = draw_fn
        self.gui.redraw_canvas(self.to_draw)


    def remove_to_draw(self, name: str):
        """Removes draw function from the to_draw dictionary."""

        if name not in self.to_draw:
            return

        del self.to_draw[name]
        self.gui.redraw_canvas(self.to_draw)


    def load_data(self):
        """Opens file with filepath stored in self.data_filepath property,
        tries to parse it and load to the app. Currently it support CSV format
        and JSON format.
        """

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

        # Try to parse the data
        try:
            raw_data = json.load(fp)["data"]
        except (json.decoder.JSONDecodeError, TypeError, KeyError) as _:
            try:
                fp.seek(0)
                raw_data = [ row for row in csv.reader(fp, delimiter=" ") if row ]
            except:
                logging.error(
                    f"Unable to parse the input file '{self.data_filepath}'!"
                    "Expected proper JSON or CSV format."
                )
                return
        try:
            self.data = [ get_data(d) for d in raw_data ]
            self.data_names = [ get_data_name(d) for d in raw_data ]
        except ValueError:
            logging.error("Invalid format of the input file!")
            return

        self.to_draw["data"] = self.draw_data
        self.reset()


    def set_algorithm(self, algorithm_name: str):
        """Sets used algorithm for the solution."""

        self.algorithm_name = algorithm_name
        self.algorithm_class = self.ALGORITHM_CLASSES[algorithm_name]
        if self.has_gui:
            self.gui.update_params(self.gui.ALGORIHTM_PARAMS[algorithm_name])


    def start(self):
        """Starts the app."""

        self.save_params()
        if self.data_filepath is not None:
            self.load_data()


    def reset(self, reseed: bool = True):
        """Restarts executions inside the app."""

        if not self.data:
            return

        if self.algorithm_runner is not None and self.algorithm_runner.is_alive:
            self.algorithm_runner.terminate()

        not_finished = self.algorithm and not self.algorithm.is_finished
        has_any_data = self.algorithm and self.algorithm.current_iteration > 0
        if self.algorithm_stats.run is not None and not_finished and has_any_data:
            self.algorithm_stats.store()
            if self.has_gui:
                self.gui.update_history()

        if self.has_gui:
            self.gui.clear_log()
            self.gui.disable_speed_label()
            if reseed and self.gui.var_fixed_seed.get() == 0:
                self.seed = get_seed()
                self.gui.var_seed.set(self.seed)

        numpy.random.seed(self.seed)
        self.algorithm_init()
        if self.has_gui:
            self.gui.clear_convergence()
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
        """Creates dictionary from the current parameters of algorithm."""
        result = {}
        result["algorithm"] = self.gui.var_algorithm.get()
        for p in self.current_params:
            result[p] = self.current_params[p]

        return result


    def algorithm_init(self):
        """Instantiates the algorithm give it to the new runner and starts it.
        """

        if self.has_gui:
            stored_params_str = f"initializing with seed={self.seed}"
            for p in self.current_params:
                stored_params_str += f", {p}={self.current_params[p]}"
            logging.info(stored_params_str)

        self.algorithm_stats.run_init(self.seed, self.algorithm_name, self.current_params)
        self.algorithm = self.algorithm_class(
            alg.AntAlgorithm.tuples_to_places(self.data, self.data_names),
            **self.current_params,
        )
        self.algorithm.start()

        if self.has_gui:
            logging.info(f"'{self.gui.var_algorithm.get()}' initialized")

        # Prepare AlgorithmRunner with the algorithm
        self.algorithm_runner = AlgorithmRunner(
            self.algorithm,
            self.gui,
            self.algorithm_stats,
            self._on_algorithm_done,
            self._on_algorithm_iteration_done,
        )
        self.algorithm_runner.start()


    def draw_best_path(self):
        if (not self.has_gui or
            self.algorithm_stats.run is None or
            self.algorithm_stats.run.best_solution is None):
            return

        best_path = self.algorithm_stats.run.best_solution[0]
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
