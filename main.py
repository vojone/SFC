import tkinter

import sys
import os
import json
import tkinter.filedialog
import threading
import tkinter.ttk
import matplotlib.pyplot as plt
import numpy
import numpy.typing

from ant_system import AntAlgorithm, AntSystem, AntColony
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk, FigureCanvasTkAgg


class GUI:
    ALGORIHTM_PARAMS = {
        "Ant System": {
            "ant_amount": (20, "Number Of Ants", tkinter.IntVar),
            "pheronome_w": (1.0, "Pheromone weight", tkinter.DoubleVar),
            "visibility_w": (1.0, "Visibility weight", tkinter.DoubleVar),
            "vaporization": (0.2, "Vaporization", tkinter.DoubleVar),
        },
        "Ant Colony": {
            "ant_amount": (20, "Number Of Ants", tkinter.IntVar),
            "pheronome_w": (1.0, "Pheromone weight", tkinter.DoubleVar),
            "visibility_w": (1.0, "Visibility weight", tkinter.DoubleVar),
            "vaporization": (0.2, "Vaporization", tkinter.DoubleVar),
            "exploitation_coef": (0.3, "Exploitation threshold ", tkinter.DoubleVar),
        },
    }

    def __init__(self):
        self.root = tkinter.Tk()
        self.root.wm_title("ACO")

        self.var_opened_file = tkinter.StringVar(master=self.root, value="")
        self.opened_file_label = tkinter.Label(
            master=self.root, textvariable=self.var_opened_file
        )
        self.opened_file_label.pack(side=tkinter.TOP)

        self.button_open_file = tkinter.Button(master=self.root, text="Open file")
        self.button_open_file.pack(side=tkinter.TOP)

        self.var_algorithm = tkinter.StringVar()
        self.combobox_algorithm = tkinter.ttk.Combobox(
            master=self.root, state="readonly", textvariable=self.var_algorithm
        )
        self.combobox_algorithm.pack(side=tkinter.TOP)

        fig = plt.figure(figsize=(5, 5), dpi=100)
        self.graph_axis = fig.add_subplot()

        self.var_iterations = tkinter.IntVar(master=self.root, value=0)
        self.label_iterations = tkinter.Label(
            master=self.root, textvariable=self.var_iterations
        )
        self.label_iterations.pack(side=tkinter.TOP)

        self.var_total_iterations = tkinter.IntVar(master=self.root, value=100)
        self.label_total_iterations = tkinter.Label(
            master=self.root, text="Total iterations"
        )
        self.label_total_iterations.pack(side=tkinter.TOP)
        self.entry_total_iterations = tkinter.Entry(
            master=self.root, textvariable=self.var_total_iterations
        )
        self.entry_total_iterations.pack(side=tkinter.TOP)

        self.var_seed = tkinter.IntVar(master=self.root, value=0)
        label_seed = tkinter.Label(master=self.root, text="Seed")
        label_seed.pack(side=tkinter.TOP)
        self.entry_seed = tkinter.Entry(master=self.root, textvariable=self.var_seed)
        self.entry_seed.pack(side=tkinter.TOP)

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(
            side=tkinter.TOP, fill=tkinter.BOTH, expand=True
        )

        self.button_run = tkinter.Button(master=self.root, text="Run")
        self.button_run.pack(side=tkinter.BOTTOM)

        self.button_stop = tkinter.Button(master=self.root, text="Stop")
        self.button_stop.pack(side=tkinter.BOTTOM)

        self.button_step = tkinter.Button(master=self.root, text="Step")
        self.button_step.pack(side=tkinter.BOTTOM)

        self.button_reset = tkinter.Button(master=self.root, text="Reset")
        self.button_reset.pack(side=tkinter.BOTTOM)

        self.button_quit = tkinter.Button(master=self.root, text="Quit")
        self.button_quit.pack(side=tkinter.BOTTOM)

        self.param_frame = tkinter.Frame(master=self.root)
        self.param_frame.pack(side=tkinter.BOTTOM)
        self.param_dict = {}

        self.var_pheronomone = tkinter.IntVar(master=self.root, value=0)
        self.checkbox_pheromone = tkinter.Checkbutton(
            master=self.root,
            text="Show pheromone",
            variable=self.var_pheronomone,
            onvalue=1,
            offvalue=0,
        )
        self.checkbox_pheromone.pack(side=tkinter.BOTTOM)

        self.var_best_path = tkinter.IntVar(master=self.root, value=1)
        self.checkbox_best_path = tkinter.Checkbutton(
            master=self.root,
            text="Show best path",
            variable=self.var_best_path,
            onvalue=1,
            offvalue=0,
        )
        self.checkbox_best_path.pack(side=tkinter.BOTTOM)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)

    def set_quit_fn(self, quit_fn):
        self.button_quit.configure(command=quit_fn)
        self.root.protocol("WM_DELETE_WINDOW", quit_fn)

    def draw_path(self, path: list, data: list):
        for i, p in enumerate(path):
            place_i = p
            next_place_i = path[i + 1 if i + 1 < len(path) else 0]
            place = data[place_i]
            next_place = data[next_place_i]
            self.graph_axis.plot(
                [place.coords[0], next_place.coords[0]],
                [place.coords[1], next_place.coords[1]],
                color="r",
            )

    def draw_matrix_data(
        self,
        matrix: numpy.typing.NDArray,
        data: list,
        color: str = "g",
        threshold: float = 0.1,
        min_width: float = 0.1,
        max_width: float = 5.0,
    ):
        # Iterate through elements in the upper triangular matrix, because
        # distance matrix/pheromone matrix should be symetric
        max_value = matrix.max()
        min_value = matrix.min()
        val_diff = max_value - min_value
        for ri, r in enumerate(matrix):
            for ci in range(len(r) - ri):
                # Compute width
                place_i = ri
                next_place_i = ri + ci
                value = matrix[place_i][next_place_i]

                # Normalize interval to <0.0, 1.0> (min-max normalization)
                normalized_value = (value - min_value) / (val_diff + 1e-30)
                if normalized_value < threshold:
                    continue

                # Renormalize to <0.0, 1.0> because the interval was clipped by threshold
                renormalized_value = (normalized_value - threshold) / (1 - threshold + 1e-30)
                width = renormalized_value * (max_width - min_width) + min_width
                place = data[place_i]
                next_place = data[next_place_i]
                self.graph_axis.plot(
                    [place.coords[0], next_place.coords[0]],
                    [place.coords[1], next_place.coords[1]],
                    linewidth=width,
                    color=color,
                )

    def draw_map(self, data: list):
        self.graph_axis.scatter(x=[p[1][0] for p in data], y=[p[1][1] for p in data])

    def open_data_file(self):
        return tkinter.filedialog.askopenfilename(
            master=self.root,
            title="Select input file",
            filetypes=(("JSON files", "*.json*"), ("All files", "*.*")),
        )

    def update_params(self, new_params: dict):
        for c in self.param_frame.winfo_children():
            c.destroy()

        self.param_dict.clear()
        for param_name in new_params:
            default, label_text, var_type = new_params[param_name]
            var_param_entry = var_type(master=self.param_frame, value=default)

            label_param_entry = tkinter.Label(master=self.param_frame, text=label_text)
            label_param_entry.pack(side=tkinter.TOP)

            param_entry = tkinter.Entry(
                name=param_name, master=self.param_frame, textvariable=var_param_entry
            )
            param_entry.pack(side=tkinter.TOP)

            self.param_dict[param_name] = var_param_entry

    def set_algorithm_options(self, algorithm_options: list[str]):
        self.combobox_algorithm.configure(values=algorithm_options)
        self.var_algorithm.set(algorithm_options[0])


class AlgorithmRunner:
    def __init__(self, algorithm: AntAlgorithm, gui: GUI | None, done_callback):
        self.algorithm = algorithm
        self.gui = gui
        self.thread = threading.Thread(target=self.runner)
        self.sleep_event = threading.Event()
        self.finish_task_event = threading.Event()
        self.run_event = threading.Event()
        self.terminated_event = threading.Event()
        self.done_callback = done_callback

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
        self.terminated_event.set()
        self.sleep_event.set()
        self.thread.join()

    def runner(self):
        while not self.terminated_event.is_set():
            self.sleep_event.wait()
            self.sleep_event.clear()
            if self.terminated_event.is_set():
                break

            if self.run_event.is_set():
                while self.run_event.is_set() and self.algorithm.make_step():
                    if self.gui is not None:
                        self.gui.var_iterations.set(self.algorithm.current_iteration)
                        self.gui.root.update_idletasks()
            else:
                self.algorithm.make_step()
                self.gui.var_iterations.set(self.algorithm.current_iteration)

            self.done_callback(not self.algorithm.is_finished)
            self.finish_task_event.set()


class App:
    ALGORITHM_CLASSES = {
        "Ant System": AntSystem,  # Default
        "Ant Colony": AntColony,
    }

    def __init__(self, data_filepath: str | None, has_gui: bool = True):
        self.data_filepath = data_filepath

        self.data: list | None = None
        self.algorithm: AntAlgorithm | None = None
        self.algorithm_runner: AlgorithmRunner | None = None
        self.algorithm_class = None

        self.best_solution = None
        self.to_draw = {}

        default_algorithm_name = list(self.ALGORITHM_CLASSES.keys())[0]
        self.algorithm_class = self.ALGORITHM_CLASSES[default_algorithm_name]

        if has_gui:
            self.gui = GUI()
            self.gui.set_quit_fn(self._quit)
            self.gui.var_opened_file.set(
                os.path.basename(data_filepath)
                if data_filepath is not None
                else "No file opened"
            )
            self.gui.button_open_file.configure(command=self._open_file)
            self.gui.button_run.configure(command=self._run)
            self.gui.button_stop.configure(command=self._stop)
            self.gui.button_step.configure(command=self._step)
            self.gui.button_reset.configure(command=self._reset)

            self.gui.set_algorithm_options(list(self.ALGORITHM_CLASSES.keys()))
            self.gui.var_algorithm.trace_add("write", self._change_algorithm)

            default_algorithm_name = list(self.ALGORITHM_CLASSES.keys())[0]
            self.gui.update_params(self.gui.ALGORIHTM_PARAMS[default_algorithm_name])

            self.gui.checkbox_pheromone.configure(command=self._toggle_pheromone)
            self.gui.checkbox_best_path.configure(command=self._toggle_best_path)

    @property
    def has_gui(self):
        return self.gui is not None

    def _quit(self):
        if self.algorithm_runner is not None:
            self.algorithm_runner.terminate()

        if self.has_gui:
            self.gui.root.quit()
            self.gui.root.destroy()

    def _change_algorithm(self, *args, **kwargs):
        algorithm_name = self.gui.var_algorithm.get()
        self.set_algorithm(algorithm_name)
        self.reset()
        print(f"Algorithm changed to {self.gui.var_algorithm.get()}")

    def _algorithm_cb(self, continues: bool):
        self.best_solution = (self.algorithm.best_path, self.algorithm.best_path_len)

        if self.has_gui:
            self.gui.graph_axis.cla()
            if self.gui.var_best_path.get() == 1:
                self.draw_best_path()
            if self.gui.var_pheronomone.get() == 1:
                self.draw_best_path()

            self.gui.draw_map(self.data)
            self.gui.canvas.draw()
            if not continues:
                self.gui.button_step["state"] = "disabled"
                self.gui.button_run["state"] = "disabled"
            else:
                self.gui.button_step["state"] = "normal"
                self.gui.button_run["state"] = " normal"

    def _stop(self):
        self.algorithm_runner.stop()

    def _reset(self):
        self.reset()

    def _step(self):
        if self.algorithm_runner is None:
            return

        self.algorithm_runner.make_step()

    def _run(self):
        if self.algorithm_runner is None:
            return

        self.algorithm_runner.run()
        self.gui.button_run["state"] = "disabled"

    def _open_file(self):
        self.data_filepath = self.gui.open_data_file()

        data_filename = os.path.basename(self.data_filepath)
        self.gui.var_opened_file.set(data_filename)
        self.load_data()
        self._reset()

    def _toggle_pheromone(self):
        if self.gui.var_pheronomone.get() == 1:
            self.to_draw["pheromone"] = self.draw_pheromone
        else:
            del self.to_draw["pheromone"]

    def _toggle_best_path(self):
        if self.gui.var_pheronomone.get() == 1:
            self.to_draw["best_path"] = self.draw_best_path
        else:
            del self.to_draw["best_path"]

    def load_data(self):
        fp = open(self.data_filepath, "r")
        self.data = json.load(fp)

    def set_algorithm(self, algorithm_name: str):
        self.algorithm_class = self.ALGORITHM_CLASSES[algorithm_name]
        if self.has_gui:
            self.gui.update_params(self.gui.ALGORIHTM_PARAMS[algorithm_name])

    def start(self):
        if self.data_filepath is not None:
            self.load_data()

        self.reset()

    def reset(self):
        if self.has_gui:
            self.gui.button_step["state"] = "normal"
            self.gui.button_run["state"] = "normal"
            self.gui.graph_axis.cla()
            if self.data is not None:
                self.gui.draw_map(self.data)
            self.gui.canvas.draw()

        if self.algorithm_runner is not None and self.algorithm_runner.is_alive:
            self.algorithm_runner.terminate()

        self.algorithm_init()
        if self.has_gui:
            self.gui.var_iterations.set(self.algorithm.current_iteration)

    def algorithm_init(self):
        current_params = {}
        total_iterations = 0

        if self.has_gui:
            total_iterations = self.gui.var_total_iterations.get()
            for p in self.gui.param_dict:
                current_params[p] = self.gui.param_dict[p].get()

        self.algorithm = self.algorithm_class(
            AntAlgorithm.tuples_to_places(self.data),
            iterations=total_iterations,
            **current_params,
        )
        self.algorithm.start()
        self.algorithm_runner = AlgorithmRunner(
            self.algorithm, self.gui, self._algorithm_cb
        )
        self.algorithm_runner.start()

    def draw_best_path(self):
        if not self.has_gui:
            return

        best_path = self.best_solution[0]
        self.gui.draw_path(best_path, self.algorithm.map.places)

    def draw_pheromone(self):
        if not self.has_gui:
            return

        self.gui.draw_matrix_data(
            self.algorithm.map.pheromone_m, self.algorithm.map.places
        )

    def redraw_canvas(self):
        if not self.has_gui:
            return

        self.gui.graph_axis.cla()
        for _, draw_fn in self.to_draw:
            draw_fn()

        if self.data is not None:
            self.gui.draw_map(self.data)

        self.gui.canvas.draw()


if __name__ == "__main__":
    app = App(sys.argv[1])
    app.start()
    tkinter.mainloop()
