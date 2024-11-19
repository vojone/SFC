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
        pass


class App:
    def __init__(self, data_filepath : str):
        self.data_filepath = data_filepath
        self.algorithm = None
        self.best_solution = None

        self.root = tkinter.Tk()
        self.root.wm_title("ACO")
        self.opened_file_label = tkinter.Label(master=self.root, text=os.path.basename(self.data_filepath))
        self.opened_file_label.pack(side=tkinter.TOP)

        self.button_open_file = tkinter.Button(master=self.root, text="Open file", command=self._open_file)
        self.button_open_file.pack(side=tkinter.TOP)

        fig = plt.figure(figsize=(5, 5), dpi=100)
        self.graph_axis = fig.add_subplot()

        self.var_iterations = tkinter.IntVar(master=self.root, value=0)
        self.label_iterations = tkinter.Label(master=self.root, textvariable=self.var_iterations)
        self.label_iterations.pack(side=tkinter.TOP)

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)

        self.button_run = tkinter.Button(master=self.root, text="Run", command=self._run)
        self.button_run.pack(side=tkinter.BOTTOM)

        self.button_step = tkinter.Button(master=self.root, text="Step", command=self._step)
        self.button_step.pack(side=tkinter.BOTTOM)

        self.button_reset = tkinter.Button(master=self.root, text="Reset", command=self._reset)
        self.button_reset.pack(side=tkinter.BOTTOM)

        self.button_quit = tkinter.Button(master=self.root, text="Quit", command=self._quit)
        self.button_quit.pack(side=tkinter.BOTTOM)

        self.root.protocol('WM_DELETE_WINDOW', self._quit)

    def _quit(self):
        self.root.quit()
        self.root.destroy()

    def _step(self):
        continues = self.algorithm_step()
        self.var_iterations.set(self.algorithm.current_iteration)

        self.graph_axis.cla()
        self.draw_best_path()
        self.draw_map()
        self.canvas.draw()
        if not continues:
            self.button_step["state"] = "disabled"


    def _run(self):
        self.algorithm_run_until_end()
        self.var_iterations.set(self.algorithm.current_iteration)
        self.graph_axis.cla()
        self.draw_best_path()
        self.draw_map()
        self.canvas.draw()
        self.button_step["state"] = "disabled"

    def _reset(self):
        self.button_step["state"] = "normal"
        self.graph_axis.cla()
        self.draw_map()
        self.canvas.draw()

        self.algorithm_init(AntSystem)
        self.var_iterations.set(self.algorithm.current_iteration)

    def _open_file(self):
        self.data_filepath = tkinter.filedialog.askopenfilename(
            master=self.root,
            title="Select input file",
            filetypes = (("JSON files", "*.json*"), ("All files","*.*"))
        )

        self.opened_file_label.configure(text=os.path.basename(self.data_filepath))
        self.load_data()
        self._reset()

    def load_data(self):
        fp = open(self.data_filepath, "r")
        self.data = json.load(fp)

    def start(self):
        self.graph_axis.cla()
        if self.data_filepath is not None:
            self.load_data()
            self.draw_map()

        self.algorithm_init(AntSystem)
        self.var_iterations.set(self.algorithm.current_iteration)

    def end(self):
        pass

    def draw_map(self):
        self.graph_axis.scatter(x=[ p[1][0] for p in self.data], y=[ p[1][1] for p in self.data])

    def draw_best_path(self):
        best_path = self.best_solution[0]
        for i, p in enumerate(best_path):
            place_i = p
            next_place_i = best_path[i + 1 if i + 1 < len(best_path) else 0]
            place = self.algorithm.map.places[place_i]
            next_place = self.algorithm.map.places[next_place_i]
            self.graph_axis.plot(
                [place.coords[0], next_place.coords[0]],
                [place.coords[1], next_place.coords[1]],
                color='r'
            )


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
            self.var_iterations.set(self.algorithm.current_iteration)
            self.root.update_idletasks()

        self.best_solution = (
            self.algorithm.best_path,
            self.algorithm.best_path_len
        )

if __name__ == "__main__":
    app = App(sys.argv[1])
    app.start()
    tkinter.mainloop()


