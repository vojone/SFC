import tkinter
import tkinter.filedialog
import tkinter.scrolledtext
import tkinter.ttk
import matplotlib.pyplot as plt
import numpy
import logging

from datetime import datetime
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk, FigureCanvasTkAgg


def positive_integer(x : str):
    try:
        if int(x) <= 0:
            raise ValueError("lower than zero")
    except ValueError:
        raise Exception(f"expected integer greater than zero, got {x}")

def valid_float(x : str):
    try:
        float(x)
    except ValueError:
        raise Exception(f"expected float, got {x}")

def mfloat_between_one_and_zero(x):
    try:
        if float(x) < 0 or float(x) > 1.0:
            raise ValueError("outside the interval")
    except ValueError:
        raise Exception(f"expected float between 0.0 and 1.0, got {x}")

class GUI:
    ALGORIHTM_PARAMS = {
        "Ant System": {
            "ant_amount": (20, "Number Of Ants", tkinter.IntVar, positive_integer),
            "pheronome_w": (1.0, "Pheromone weight", tkinter.DoubleVar, valid_float),
            "visibility_w": (1.0, "Visibility weight", tkinter.DoubleVar, valid_float),
            "vaporization": (0.2, "Vaporization", tkinter.DoubleVar, mfloat_between_one_and_zero),
        },
        "Ant Colony": {
            "ant_amount": (20, "Number Of Ants", tkinter.DoubleVar, positive_integer),
            "pheronome_w": (1.0, "Pheromone weight", tkinter.DoubleVar, valid_float),
            "visibility_w": (1.0, "Visibility weight", tkinter.DoubleVar, valid_float),
            "vaporization": (0.2, "Vaporization", tkinter.DoubleVar, mfloat_between_one_and_zero),
            "exploitation_coef": (0.3, "Exploitation threshold ", tkinter.DoubleVar, mfloat_between_one_and_zero),
        },
    }

    ANNOTATION_FONT_DICT = {
        "backgroundcolor": "#ffffffd0",
        "color": "gray",
        "size": 7,
        "horizontalalignment": "center",
        "verticalalignment": "center",
    }

    def __init__(self, logger=None):
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

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(
            side=tkinter.TOP, fill=tkinter.BOTH, expand=True
        )

        self.button_save = tkinter.Button(master=self.root, text="Save changes")
        self.button_save.pack(side=tkinter.BOTTOM)

        self.button_restore = tkinter.Button(master=self.root, text="Restore")
        self.button_restore.pack(side=tkinter.BOTTOM)

        self.button_run = tkinter.ttk.Button(master=self.root, text="Run")
        self.button_run.pack(side=tkinter.BOTTOM)

        self.button_stop = tkinter.Button(master=self.root, text="Stop")
        self.button_stop.pack(side=tkinter.BOTTOM)

        self.button_step = tkinter.Button(master=self.root, text="Step")
        self.button_step.pack(side=tkinter.BOTTOM)

        self.button_reset = tkinter.Button(master=self.root, text="Reset")
        self.button_reset.pack(side=tkinter.BOTTOM)

        self.param_frame = tkinter.Frame(master=self.root)
        self.param_frame.pack(side=tkinter.BOTTOM)
        self.param_dict = {}

        self.var_pheronomone = tkinter.IntVar(master=self.root, value=0)
        self.checkbox_pheromone = self.create_checkbox(
            self.root, "Show pheromone", self.var_pheronomone
        )
        self.checkbox_pheromone.pack(side=tkinter.BOTTOM)

        self.var_best_path = tkinter.IntVar(master=self.root, value=1)
        self.checkbox_best_path = self.create_checkbox(
            self.root, "Show best path", self.var_best_path
        )
        self.checkbox_best_path.pack(side=tkinter.BOTTOM)

        self.var_seed = tkinter.IntVar(master=self.root, value=0)
        self.entry_seed = None

        self.var_fixed_seed = tkinter.IntVar(master=self.root, value=0)
        self.checkbox_fixed_seed = None
        self.use_custom_seed = None

        self.var_show_place_names = tkinter.IntVar(master=self.root, value=0)
        self.checkbox_place_names_on_change = None

        self.var_distances = tkinter.IntVar(master=self.root, value=0)
        self.checkbox_distances_on_change = None

        self.var_pheromone_amount = tkinter.IntVar(master=self.root, value=0)
        self.checkbox_pheromone_amount_on_change = None

        self.toolbar = tkinter.Menu(self.root)
        self.root.config(menu=self.toolbar)
        self.file_menu = tkinter.Menu(self.toolbar, tearoff="off")
        self.file_menu.add_command(label="Save log", command=self.save_log)
        self.file_menu.add_separator()

        self.on_quit = None
        self.save_params_cb = None
        self.save_params_with_seed_cb = None
        self.load_params_cb = None
        self.file_menu.add_command(label="Save params", command=self.save_params)
        self.file_menu.add_command(label="Save params with seed", command=self.save_params_with_seed)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Load params", command=self.load_params)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Quit", command=self.quit)

        self.toolbar.add_cascade(label="File", menu=self.file_menu)

        self.toolbar.add_command(label="Settings", command=self.open_setings_menu)
        self.toolbar.add_command(label="Log", command=self.open_log_window)

        self.canvas_toolbar = NavigationToolbar2Tk(
            self.canvas, self.root, pack_toolbar=False
        )
        self.canvas_toolbar.update()
        self.canvas_toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)

        self.logging_widget = None
        self.log = []
        if logger is not None:
            text_handler = GUILogHandler(self, self.log)
            logger = logging.getLogger()
            logger.addHandler(text_handler)

    def quit(self):
        if self.on_quit:
            self.on_quit()

    def set_quit_fn(self, quit_fn):
        self.on_quit = quit_fn
        self.root.protocol("WM_DELETE_WINDOW", quit_fn)

    def load_params(self):
        if self.load_params_cb:
            self.load_params_cb()

    def open_log_window(self):
        def on_close():
            self.logging_widget = None
            log_window.destroy()

        log_window = tkinter.Toplevel(self.root)
        log_window.transient(self.root)
        log_window.title("ACO - Log")
        log_window.protocol("WM_DELETE_WINDOW", on_close)
        self.logging_widget = tkinter.scrolledtext.ScrolledText(master=log_window)
        self.logging_widget.config(spacing3=10)
        self.logging_widget.pack(expand=True, fill="both")
        self.logging_widget.configure(state="normal")
        trailing_newline = "\n" if self.log else ""
        self.logging_widget.insert(tkinter.END, "\n".join(self.log) + trailing_newline)
        self.logging_widget.configure(state="disabled")

    def save_log(self):
        timestamp = datetime.now().strftime("%m-%d-%H%M%S")
        alg_name = self.var_algorithm.get().replace(" ", "")
        ifilename = f"{alg_name}-{timestamp}.log"
        filename = tkinter.filedialog.asksaveasfilename(
            confirmoverwrite=True,
            title="Save Log As",
            initialfile=ifilename
        )

        if not filename:
            return

        fp = open(filename, mode="w")
        fp.write("\n".join(self.log) + "\n")
        fp.close()

    def save_params(self):
        if self.save_params_cb is not None:
            self.save_params_cb()

    def save_params_with_seed(self):
        if self.save_params_with_seed_cb is not None:
            self.save_params_with_seed_cb()

    def open_save_params(self, custom_str : str = ""):
        timestamp = datetime.now().strftime("%H%M%S")
        alg_name = self.var_algorithm.get().replace(" ", "")
        filename = f"{alg_name}-params{custom_str}-{timestamp}.json"
        return tkinter.filedialog.asksaveasfilename(
            confirmoverwrite=True,
            title="Save Params As",
            initialfile=filename
        )

    def open_setings_menu(self):
        settings_window = tkinter.Toplevel(self.root)
        settings_window.transient(self.root)
        settings_window.title("ACO - Advanced settings")

        label_seed = tkinter.Label(master=settings_window, text="Seed")
        label_seed.pack(side=tkinter.TOP)
        self.entry_seed = tkinter.Entry(
            master=settings_window, textvariable=self.var_seed
        )
        self.entry_seed.pack(side=tkinter.TOP)

        button_use_seed = tkinter.Button(settings_window, text="Use")
        if self.use_custom_seed:
            button_use_seed.configure(command=self.use_custom_seed)
        button_use_seed.pack(side=tkinter.TOP)

        checkbox_fix_seed = self.create_checkbox(
            settings_window, "Fix seed", self.var_fixed_seed
        )
        if checkbox_fix_seed:
            checkbox_fix_seed.configure(command=self.checkbox_fixed_seed)
        checkbox_fix_seed.pack(side=tkinter.TOP)

        checkbox_place_names = self.create_checkbox(
            settings_window, "Show place names", self.var_show_place_names
        )
        if self.checkbox_place_names_on_change:
            checkbox_place_names.configure(command=self.checkbox_place_names_on_change)
        checkbox_place_names.pack(side=tkinter.BOTTOM)

        checkbox_distances = self.create_checkbox(
            settings_window, "Show path distances", self.var_distances
        )
        if self.checkbox_distances_on_change:
            checkbox_distances.configure(command=self.checkbox_distances_on_change)
        checkbox_distances.pack(side=tkinter.BOTTOM)

        checkbox_pheromone_amount = self.create_checkbox(
            settings_window, "Show pheromone amount", self.var_pheromone_amount
        )
        if self.checkbox_pheromone_amount_on_change:
            checkbox_pheromone_amount.configure(
                command=self.checkbox_pheromone_amount_on_change
            )
        checkbox_pheromone_amount.pack(side=tkinter.BOTTOM)

    def create_checkbox(self, root, label, variable) -> tkinter.Checkbutton:
        return tkinter.Checkbutton(
            master=root,
            text=label,
            variable=variable,
            onvalue=1,
            offvalue=0,
        )

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
            title="Select input file with data",
            filetypes=(("JSON files", "*.json*"), ("All files", "*.*")),
        )

    def open_params_file(self):
        return tkinter.filedialog.askopenfilename(
            master=self.root,
            title="Select file with params",
            filetypes=(("JSON files", "*.json*"), ("All files", "*.*")),
        )

    def update_params(self, new_params: dict):
        for c in self.param_frame.winfo_children():
            c.destroy()

        self.param_dict.clear()
        for param_name in new_params:
            default, label_text, var_type, validator = new_params[param_name]
            var_param_entry = var_type(master=self.param_frame, value=default)

            label_param_entry = tkinter.Label(master=self.param_frame, text=label_text)
            label_param_entry.pack(side=tkinter.TOP)

            param_entry = tkinter.Entry(
                name=param_name, master=self.param_frame, textvariable=var_param_entry
            )
            param_entry.pack(side=tkinter.TOP)

            var_param_entry.trace_add(
                "write",
                lambda *args, param_name=param_name, **kwargs: self.param_changed(
                    param_name
                ),
            )

            self.param_dict[param_name] = (var_param_entry, param_entry, validator)

    def param_changed(self, param_name: str):
        try:
            new_val = self.param_dict[param_name][1].get()
            self.param_dict[param_name][2](new_val)
        except Exception:
            self.param_dict[param_name][1].configure(background="#ffdddd")
        else:
            self.param_dict[param_name][1].configure(background="#dddddd")
        self.button_save["state"] = "normal"
        self.button_restore["state"] = "normal"

    def param_validate(self):
        for name in self.param_dict:
            try:
                self.param_dict[name][2](self.param_dict[name][1].get())
            except Exception as e:
                logging.error(f"{e}")
                return False

        return True

    def param_stored(self):
        for param_name in self.param_dict:
            self.param_dict[param_name][1].configure(background="#ffffff")

        self.button_save["state"] = "disabled"
        self.button_restore["state"] = "disabled"

    def set_algorithm_options(self, algorithm_options: list[str]):
        self.combobox_algorithm.configure(values=algorithm_options)
        self.var_algorithm.set(algorithm_options[0])

    def clear_log(self):
        self.log.clear()
        if self.logging_widget is not None:
            self.logging_widget.configure(state="normal")
            self.logging_widget.delete(1.0, tkinter.END)
            self.logging_widget.configure(state="disabled")


class GUILogHandler(logging.Handler):
    """Custom logging handler for saving logs to the variable and eventually
    for printing log to the window.
    """

    FORMAT_STR = "%(asctime)s.%(msecs)03d: %(levelname)s: %(message)s"

    def __init__(self, gui: GUI, log_list: list[str]):
        super().__init__()
        self.gui = gui
        self.log_list = log_list
        self.formatter = logging.Formatter(self.FORMAT_STR, datefmt="%H:%M:%S")

    def emit(self, record: logging.LogRecord):
        log_msg = self.format(record)
        self.log_list.append(log_msg)
        if self.gui.logging_widget is not None:
            self.gui.logging_widget.configure(state="normal")
            self.gui.logging_widget.insert(tkinter.END, f"{log_msg}\n")
            self.gui.logging_widget.configure(state="disabled")
            self.gui.logging_widget.see(tkinter.END)
