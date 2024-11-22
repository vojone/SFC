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

from ant_algorithm import AntAlgorithm, AntSystem, AntColony
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk, FigureCanvasTkAgg

def get_seed():
    return numpy.random.randint(1, int(1e9))

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

    ANNOTATION_FONT_DICT = {
        "backgroundcolor": "#ffffffd0",
        "color": "gray",
        "size": 7,
        "horizontalalignment" : "center",
        "verticalalignment" : "center"
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

        self.var_show_place_names = tkinter.IntVar(master=self.root, value=0)
        self.checkbox_place_names = tkinter.Checkbutton(
            master=self.root,
            text="Show place names",
            variable=self.var_show_place_names,
            onvalue=1,
            offvalue=0,
        )
        self.checkbox_place_names.pack(side=tkinter.BOTTOM)

        self.var_distances = tkinter.IntVar(master=self.root, value=0)
        self.checkbox_distances = tkinter.Checkbutton(
            master=self.root,
            text="Show path distances",
            variable=self.var_distances,
            onvalue=1,
            offvalue=0,
        )
        self.checkbox_distances.pack(side=tkinter.BOTTOM)

        self.var_pheromone_amount = tkinter.IntVar(master=self.root, value=0)
        self.checkbox_pheromone_amount = tkinter.Checkbutton(
            master=self.root,
            text="Show pheromone amount",
            variable=self.var_pheromone_amount,
            onvalue=1,
            offvalue=0,
        )
        self.checkbox_pheromone_amount.pack(side=tkinter.BOTTOM)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)

    def set_quit_fn(self, quit_fn):
        self.button_quit.configure(command=quit_fn)
        self.root.protocol("WM_DELETE_WINDOW", quit_fn)

    def draw_path(self, path: list, data: list, draw_path_len : bool = False, distance_m : numpy.typing.NDArray = None, zorder: int = 10):
        for i, p in enumerate(path):
            place_i = p
            next_place_i = path[i + 1 if i + 1 < len(path) else 0]
            place = data[place_i]
            next_place = data[next_place_i]
            self.graph_axis.plot(
                [place.coords[0], next_place.coords[0]],
                [place.coords[1], next_place.coords[1]],
                color="r",
                zorder=zorder,
            )

            if draw_path_len and distance_m is not None:
                path_length = distance_m[place_i][next_place_i]
                self.graph_axis.text(
                    x=(place.coords[0] + next_place.coords[0]) / 2,
                    y=(place.coords[1] + next_place.coords[1]) / 2,
                    s=f"{path_length:g}",
                    zorder=zorder,
                    fontdict=self.ANNOTATION_FONT_DICT,
                )

    def draw_matrix_data(
        self,
        matrix: numpy.typing.NDArray,
        data: list,
        draw_values: bool = False,
        color: str = "g",
        threshold: float = 0.1,
        min_width: float = 0.1,
        max_width: float = 5.0,
        zorder: int = 10,
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
                renormalized_value = (normalized_value - threshold) / (
                    1 - threshold + 1e-30
                )
                width = renormalized_value * (max_width - min_width) + min_width
                place = data[place_i]
                next_place = data[next_place_i]
                self.graph_axis.plot(
                    [place.coords[0], next_place.coords[0]],
                    [place.coords[1], next_place.coords[1]],
                    linewidth=width,
                    color=color,
                    zorder=zorder,
                )
                if draw_values:
                    # Draw text value in the middle of path
                    self.graph_axis.text(
                        x=(place.coords[0] + next_place.coords[0]) / 2,
                        y=(place.coords[1] + next_place.coords[1]) / 2,
                        s=f"{value:g}",
                        zorder=zorder,
                        fontdict=self.ANNOTATION_FONT_DICT,
                    )

    def draw_data(self, data: list, zorder=90):
        self.graph_axis.scatter(
            x=[p[1][0] for p in data], y=[p[1][1] for p in data], zorder=zorder
        )

    def draw_data_names(self, data: list, zorder=90):
        for p in data:
            self.graph_axis.text(
                x=p[1][0],
                y=p[1][1],
                s=f" {p[0]}",
                zorder=zorder,
                fontdict={"size": 9},
            )

    def redraw_canvas(self, to_draw):
        self.graph_axis.cla()

        for name in to_draw:
            draw_fn = to_draw[name]
            draw_fn()

        self.canvas.draw()

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
            # label_param_entry.pack(side=tkinter.TOP)

            param_entry = tkinter.Entry(
                name=param_name, master=self.param_frame, textvariable=var_param_entry
            )
            # param_entry.pack(side=tkinter.TOP)

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
        self.run_jobid = None

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

            self.gui.button_step.bind("<ButtonPress-1>", self._step_press)
            self.gui.button_step.bind("<ButtonRelease-1>", self._step_release)
            self.gui.button_reset.configure(command=self._reset)

            self.gui.set_algorithm_options(list(self.ALGORITHM_CLASSES.keys()))
            self.gui.var_algorithm.trace_add("write", self._change_algorithm)

            default_algorithm_name = list(self.ALGORITHM_CLASSES.keys())[0]
            self.gui.update_params(self.gui.ALGORIHTM_PARAMS[default_algorithm_name])

            self.gui.checkbox_pheromone.configure(command=self._toggle_pheromone)
            self.gui.checkbox_best_path.configure(command=self._toggle_best_path)

            self.gui.checkbox_place_names.configure(command=self._toggle_place_names)
            self.gui.checkbox_distances.configure(command=self._toggle_best_path)
            self.gui.checkbox_pheromone_amount.configure(command=self._toggle_pheromone)

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

    def _change_algorithm(self, *args, **kwargs):
        algorithm_name = self.gui.var_algorithm.get()
        self.set_algorithm(algorithm_name)
        self.reset()
        print(f"Algorithm changed to {self.gui.var_algorithm.get()}")

    def _algorithm_cb(self, continues: bool):
        self.best_solution = (self.algorithm.best_path, self.algorithm.best_path_len)

        if self.has_gui:
            self.gui.redraw_canvas(self.to_draw)
            if not continues:
                self.gui.button_step["state"] = "disabled"
                self.gui.button_run["state"] = "disabled"
            else:
                self.gui.button_step["state"] = "normal"
                self.gui.button_run["state"] = "normal"

    def _stop(self):
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

        self.algorithm_runner.make_step()
        self.run_jobid = self.gui.root.after(STEP_BUTTON_RUN_MS, self._run)

    def _run(self):
        if self.algorithm_runner is None:
            return

        self.algorithm_runner.run()
        self.gui.button_run["state"] = "disabled"

    def _open_file(self):
        self.data_filepath = self.gui.open_data_file()
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

    def add_to_draw(self, name : str, draw_fn):
        self.to_draw[name] = draw_fn
        self.gui.redraw_canvas(self.to_draw)

    def remove_to_draw(self, name : str):
        if name not in self.to_draw:
            return

        del self.to_draw[name]
        self.gui.redraw_canvas(self.to_draw)

    def load_data(self):
        fp = open(self.data_filepath, "r")
        self.data = json.load(fp)
        self.to_draw["data"] = self.draw_data
        if self.has_gui:
            self.gui.redraw_canvas(self.to_draw)

        self.reset()

    def set_algorithm(self, algorithm_name: str):
        self.algorithm_class = self.ALGORITHM_CLASSES[algorithm_name]
        if self.has_gui:
            self.gui.update_params(self.gui.ALGORIHTM_PARAMS[algorithm_name])

    def start(self):
        if self.data_filepath is not None:
            self.load_data()

    def reset(self):
        if self.algorithm_runner is not None and self.algorithm_runner.is_alive:
            self.algorithm_runner.terminate()
        if self.has_gui:
            self.gui.var_seed.set(get_seed())

        self.best_solution = None
        self.algorithm_init()
        if self.has_gui:
            self.gui.var_iterations.set(self.algorithm.current_iteration)
            self.gui.button_step["state"] = "normal"
            self.gui.button_run["state"] = "normal"
            self.gui.redraw_canvas(self.to_draw)

    def algorithm_init(self):
        current_params = {}
        total_iterations = 0

        if self.has_gui:
            numpy.random.seed(self.gui.var_seed.get())
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
        if not self.has_gui or self.best_solution is None:
            return

        best_path = self.best_solution[0]
        self.gui.draw_path(best_path, self.algorithm.map.places, bool(self.gui.var_distances.get()), self.algorithm.map.distance_m)

    def draw_pheromone(self):
        if not self.has_gui:
            return

        self.gui.draw_matrix_data(
            self.algorithm.map.pheromone_m, self.algorithm.map.places, bool(self.gui.var_pheromone_amount.get())
        )

    def draw_data(self):
        if self.data is not None:
            self.gui.draw_data(self.data)

    def draw_data_names(self):
        if self.data is not None:
            self.gui.draw_data_names(self.data)


if __name__ == "__main__":
    app = App(sys.argv[1] if len(sys.argv) > 1 else None)
    app.start()
    tkinter.mainloop()
