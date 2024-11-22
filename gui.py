import tkinter
import tkinter.ttk
import matplotlib.pyplot as plt
import numpy

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

    ANNOTATION_FONT_DICT = {
        "backgroundcolor": "#ffffffd0",
        "color": "gray",
        "size": 7,
        "horizontalalignment": "center",
        "verticalalignment": "center",
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

        self.button_save = tkinter.Button(master=self.root, text="Save changes")
        self.button_save.pack(side=tkinter.BOTTOM)

        self.button_restore = tkinter.Button(master=self.root, text="Restore")
        self.button_restore.pack(side=tkinter.BOTTOM)

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

    def draw_path(
        self,
        path: list,
        data: list,
        draw_path_len: bool = False,
        distance_m: numpy.typing.NDArray = None,
        zorder: int = 10,
    ):
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
            label_param_entry.pack(side=tkinter.TOP)

            param_entry = tkinter.Entry(
                name=param_name, master=self.param_frame, textvariable=var_param_entry
            )
            param_entry.pack(side=tkinter.TOP)

            self.param_dict[param_name] = var_param_entry

    def set_algorithm_options(self, algorithm_options: list[str]):
        self.combobox_algorithm.configure(values=algorithm_options)
        self.var_algorithm.set(algorithm_options[0])
