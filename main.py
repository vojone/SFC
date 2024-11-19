import tkinter

import sys
import os
import json
import tkinter.filedialog
import matplotlib.pyplot as plt

from ant_system import AntSystem
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk, FigureCanvasTkAgg

class GUI:
    def __init__(self):
        self.root = tkinter.Tk()
        self.root.wm_title("ACO")

        self.var_opened_file = tkinter.StringVar(master=self.root, value="")
        self.opened_file_label = tkinter.Label(master=self.root, textvariable=self.var_opened_file)
        self.opened_file_label.pack(side=tkinter.TOP)

        self.button_open_file = tkinter.Button(master=self.root, text="Open file")
        self.button_open_file.pack(side=tkinter.TOP)

        fig = plt.figure(figsize=(5, 5), dpi=100)
        self.graph_axis = fig.add_subplot()

        self.var_iterations = tkinter.IntVar(master=self.root, value=0)
        self.label_iterations = tkinter.Label(master=self.root, textvariable=self.var_iterations)
        self.label_iterations.pack(side=tkinter.TOP)

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

        self.button_run = tkinter.Button(master=self.root, text="Run")
        self.button_run.pack(side=tkinter.BOTTOM)

        self.button_step = tkinter.Button(master=self.root, text="Step")
        self.button_step.pack(side=tkinter.BOTTOM)

        self.button_reset = tkinter.Button(master=self.root, text="Reset")
        self.button_reset.pack(side=tkinter.BOTTOM)

        self.button_quit = tkinter.Button(master=self.root, text="Quit")
        self.button_quit.pack(side=tkinter.BOTTOM)

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

class App:
    def __init__(self, data_filepath : str):
        self.data_filepath = data_filepath
        self.algorithm = None
        self.best_solution = None

        self.gui = GUI()
        self.gui.set_quit_fn(self._quit)
        self.gui.var_opened_file.set(os.path.basename(data_filepath))
        self.gui.button_open_file.configure(command=self._open_file)
        self.gui.button_run.configure(command=self._run)
        self.gui.button_step.configure(command=self._step)
        self.gui.button_reset.configure(command=self._reset)

    def _quit(self):
        self.gui.root.quit()
        self.gui.root.destroy()

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

    def _reset(self):
        self.gui.button_step["state"] = "normal"
        self.gui.graph_axis.cla()
        self.gui.draw_map(self.data)
        self.gui.canvas.draw()

        self.algorithm_init(AntSystem)
        self.gui.var_iterations.set(self.algorithm.current_iteration)

    def _open_file(self):
        self.data_filepath = self.gui.open_data_file()

        self.gui.var_opened_file.set(text=os.path.basename(self.data_filepath))
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

        self.algorithm_init(AntSystem)
        self.gui.var_iterations.set(self.algorithm.current_iteration)

    def end(self):
        pass

    def draw_best_path(self):
        best_path = self.best_solution[0]
        self.gui.draw_path(best_path, self.algorithm.map.places)

    def algorithm_init(self, algorithm_class):
        self.algorithm = algorithm_class(
            self.data,
            ant_amount=20,
            iterations=100,
            pheronome_w=1,
            visibility_w=1,
            vaporization=0.2
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
        while self.algorithm.make_step():
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


