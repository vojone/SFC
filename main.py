import tkinter

import sys
import os
import json
import tkinter.filedialog
import threading
import tkinter.ttk
import matplotlib.pyplot as plt

from ant_system import AntAlgorithm, AntSystem, AntColony
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk, FigureCanvasTkAgg

class GUI:
    ALGORIHTM_PARAMS = {
        "Ant System": {
            "ant_amount" :  (20,    "Number Of Ants",       tkinter.IntVar),
            "pheronome_w":  (1.0,   "Pheromone weight",     tkinter.DoubleVar),
            "visibility_w": (1.0,   "Visibility weight",    tkinter.DoubleVar),
            "vaporization": (0.2,   "Vaporization",         tkinter.DoubleVar),
        },
        "Ant Colony": {
            "ant_amount" :  (20,    "Number Of Ants",       tkinter.IntVar),
            "pheronome_w":  (1.0,   "Pheromone weight",     tkinter.DoubleVar),
            "visibility_w": (1.0,   "Visibility weight",    tkinter.DoubleVar),
            "vaporization": (0.2,   "Vaporization",         tkinter.DoubleVar),
            "exploitation_coef": (0.3, "Exploitation threshold ", tkinter.DoubleVar),
        },
    }

    def __init__(self):
        self.root = tkinter.Tk()
        self.root.wm_title("ACO")

        self.var_opened_file = tkinter.StringVar(master=self.root, value="")
        self.opened_file_label = tkinter.Label(master=self.root, textvariable=self.var_opened_file)
        self.opened_file_label.pack(side=tkinter.TOP)

        self.button_open_file = tkinter.Button(master=self.root, text="Open file")
        self.button_open_file.pack(side=tkinter.TOP)

        self.var_algorithm = tkinter.StringVar()
        self.combobox_algorithm = tkinter.ttk.Combobox(master=self.root, state="readonly", textvariable=self.var_algorithm)
        self.combobox_algorithm.pack(side=tkinter.TOP)

        fig = plt.figure(figsize=(5, 5), dpi=100)
        self.graph_axis = fig.add_subplot()

        self.var_iterations = tkinter.IntVar(master=self.root, value=0)
        self.label_iterations = tkinter.Label(master=self.root, textvariable=self.var_iterations)
        self.label_iterations.pack(side=tkinter.TOP)

        self.var_seed = tkinter.IntVar(master=self.root, value=0)
        label_seed = tkinter.Label(master=self.root, text="Seed")
        label_seed.pack(side=tkinter.TOP)
        self.entry_seed = tkinter.Entry(master=self.root, textvariable=self.var_seed)
        self.entry_seed.pack(side=tkinter.TOP)

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

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

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)

    def set_quit_fn(self, quit_fn):
        self.button_quit.configure(command=quit_fn)
        self.root.protocol('WM_DELETE_WINDOW', quit_fn)

    def draw_path(self, path : list, data : list):
        for i, p in enumerate(path):
            place_i = p
            next_place_i = path[i + 1 if i + 1 < len(path) else 0]
            place = data[place_i]
            next_place = data[next_place_i]
            self.graph_axis.plot(
                [place.coords[0], next_place.coords[0]],
                [place.coords[1], next_place.coords[1]],
                color='r'
            )

    def draw_map(self, data : list):
        self.graph_axis.scatter(x=[ p[1][0] for p in data], y=[ p[1][1] for p in data])

    def open_data_file(self):
        return tkinter.filedialog.askopenfilename(
            master=self.root,
            title="Select input file",
            filetypes = (("JSON files", "*.json*"), ("All files","*.*"))
        )

    def update_params(self, new_params : dict):
        for c in self.param_frame.winfo_children():
            c.destroy()

        self.param_dict.clear()
        for param_name in new_params:
            default, label_text, var_type = new_params[param_name]
            var_param_entry = var_type(master=self.param_frame, value=default)

            label_param_entry = tkinter.Label(master=self.param_frame, text=label_text)
            label_param_entry.pack(side=tkinter.TOP)

            param_entry = tkinter.Entry(name=param_name, master=self.param_frame, textvariable=var_param_entry)
            param_entry.pack(side=tkinter.TOP)

            self.param_dict[param_name] = var_param_entry

    def set_algorithm_options(self, algorithm_options : list[str]):
        self.combobox_algorithm.configure(values=algorithm_options)
        self.var_algorithm.set(algorithm_options[0])



class App:
    ALGORITHM_CLASSES = {
        "Ant System": AntSystem, # Default
        "Ant Colony": AntColony,
    }

    def __init__(self, data_filepath : str):
        self.data_filepath = data_filepath
        self.algorithm = None
        self.algorithm_class = None
        self.algorithm_is_running = False
        self.best_solution = None

        self.gui = GUI()
        self.gui.set_quit_fn(self._quit)
        self.gui.var_opened_file.set(os.path.basename(data_filepath))
        self.gui.button_open_file.configure(command=self._open_file)
        self.gui.button_run.configure(command=self._run)
        self.gui.button_stop.configure(command=self._stop)
        self.gui.button_step.configure(command=self._step)
        self.gui.button_reset.configure(command=self._reset)

        self.gui.set_algorithm_options(list(self.ALGORITHM_CLASSES.keys()))
        self.gui.var_algorithm.trace_add('write', self._change_algorithm)

        default_algorithm_name = list(self.ALGORITHM_CLASSES.keys())[0]
        self.gui.update_params(self.gui.ALGORIHTM_PARAMS[default_algorithm_name])
        self.algorithm_class = self.ALGORITHM_CLASSES[default_algorithm_name]

    def _quit(self):
        self.gui.root.quit()
        self.gui.root.destroy()

    def _change_algorithm(self, *args, **kwargs):
        algorithm_name = self.gui.var_algorithm.get()
        self.gui.update_params(self.gui.ALGORIHTM_PARAMS[algorithm_name])
        self.algorithm_class = self.ALGORITHM_CLASSES[algorithm_name]

        self._reset()

        print(f"Algorithm changed to {self.gui.var_algorithm.get()}")

    def _step(self):
        continues = self.algorithm_step()
        self.gui.var_iterations.set(self.algorithm.current_iteration)

        self.gui.graph_axis.cla()
        self.draw_best_path()
        self.gui.draw_map(self.data)
        self.gui.canvas.draw()
        if not continues:
            self.button_step["state"] = "disabled"

    def _run(self):
        self.algorithm_run_until_end()
        self.gui.var_iterations.set(self.algorithm.current_iteration)
        self.gui.graph_axis.cla()
        self.draw_best_path()
        self.gui.draw_map(self.data)
        self.gui.canvas.draw()
        self.gui.button_step["state"] = "disabled"

    def _stop(self):
        self.algorithm_is_running = False

    def _reset(self):
        self.gui.button_step["state"] = "normal"
        self.gui.graph_axis.cla()
        self.gui.draw_map(self.data)
        self.gui.canvas.draw()

        self.algorithm_init()
        self.gui.var_iterations.set(self.algorithm.current_iteration)

    def _open_file(self):
        self.data_filepath = self.gui.open_data_file()

        self.gui.var_opened_file.set(os.path.basename(self.data_filepath))
        self.load_data()
        self._reset()

    def load_data(self):
        fp = open(self.data_filepath, "r")
        self.data = json.load(fp)

    def start(self):
        self.gui.graph_axis.cla()
        if self.data_filepath is not None:
            self.load_data()
            self.gui.draw_map(self.data)

        self.algorithm_init()
        self.gui.var_iterations.set(self.algorithm.current_iteration)

    def end(self):
        pass

    def draw_best_path(self):
        best_path = self.best_solution[0]
        self.gui.draw_path(best_path, self.algorithm.map.places)

    def algorithm_init(self):
        current_params = {}
        for p in self.gui.param_dict:
            current_params[p] = self.gui.param_dict[p].get()

        self.algorithm = self.algorithm_class(
            AntAlgorithm.tuples_to_places(self.data),
            iterations=100,
            **current_params,
            # ant_amount=20,
            # pheronome_w=1,
            # visibility_w=1,
            # vaporization=0.2
        )

        self.algorithm.start()

    def algorithm_step(self) -> bool:
        result = self.algorithm.make_step()
        self.best_solution = (
            self.algorithm.best_path,
            self.algorithm.best_path_len
        )

        return result

    def algorithm_run_until_end(self):
        self.algorithm_is_running = True
        while self.algorithm_is_running and self.algorithm.make_step():
            self.gui.var_iterations.set(self.algorithm.current_iteration)
            self.gui.root.update_idletasks()

        self.best_solution = (
            self.algorithm.best_path,
            self.algorithm.best_path_len
        )

if __name__ == "__main__":
    app = App(sys.argv[1])
    app.start()
    tkinter.mainloop()


