import tkinter
import tkinter.filedialog
import tkinter.scrolledtext
import tkinter.ttk
import matplotlib.pyplot as plt
import numpy
import logging

from datetime import datetime
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk, FigureCanvasTkAgg

from algorithm_stats import AlgorithmStats


def positive_integer(x: str):
    try:
        if int(x) <= 0:
            raise ValueError("lower than zero")
    except ValueError:
        raise Exception(f"expected integer greater than zero, got {x}")


def valid_float(x: str):
    try:
        float(x)
    except ValueError:
        raise Exception(f"expected float, got {x}")


def float_one_to_zero(x):
    try:
        if float(x) < 0 or float(x) > 1.0:
            raise ValueError("outside the interval")
    except ValueError:
        raise Exception(f"expected float between 0.0 and 1.0, got {x}")


def create_checkbox(root, label, variable) -> tkinter.Checkbutton:
    return tkinter.ttk.Checkbutton(
        master=root,
        text=label,
        variable=variable,
        onvalue=1,
        offvalue=0,
    )

class LogWindow(tkinter.Toplevel):
    def __init__(
        self,
        master,
        on_window_close,
        on_clear_log,
        on_save_log,
        log_content : str = "",
        **kwargs
    ):
        super().__init__(master=master, **kwargs)

        self.title("Ant Algorithms - Log")
        self.transient(master)
        self.minsize(400, 200)

        if on_window_close is not None:
            self.protocol("WM_DELETE_WINDOW", on_window_close)

        self.logging_widget = tkinter.scrolledtext.ScrolledText(master=self)
        self.logging_widget.config(spacing3=10)

        self.logging_widget.configure(state="normal")
        trailing_newline = "\n" if log_content else ""
        self.logging_widget.insert(tkinter.END, "\n".join(log_content) + trailing_newline)
        self.logging_widget.configure(state="disabled")

        frame_controls = tkinter.Frame(master=self)
        frame_controls.columnconfigure(2, weight=1)

        clear_button = tkinter.ttk.Button(master=frame_controls, text="Clear")
        if on_clear_log is not None:
            clear_button.configure(command=on_clear_log)
        clear_button.grid(row=0, column=0)

        save_button = tkinter.ttk.Button(master=frame_controls, text="Save")
        if on_save_log is not None:
            save_button.configure(command=on_save_log)
        save_button.grid(row=0, column=1)

        frame_controls.pack(side=tkinter.BOTTOM, fill=tkinter.X)
        self.logging_widget.pack(expand=True, side=tkinter.TOP, fill=tkinter.BOTH)


class HistoryWindow(tkinter.Toplevel):
    def __init__(
        self,
        master,
        algorithm_stats : AlgorithmStats,
        **kwargs
    ):
        super().__init__(master=master, **kwargs)

        self.displayed_objects = {}

        self.title("Ant Algorithms - History Of Runs")
        self.transient(master)
        self.minsize(400, 200)
        self.algorithm_stats = algorithm_stats


        self.frame_controls = tkinter.Frame(master=self)
        self.frame_controls.pack(side=tkinter.TOP, fill=tkinter.X, pady=(10, 10))
        self.frame_controls.columnconfigure(4, weight=1)
        initial_label = tkinter.ttk.Label(master=self.frame_controls, text="Nothing selected...")
        initial_label.pack(side=tkinter.TOP, fill=tkinter.X, padx=(10, 10), pady=(10, 10))

        self.frame_selection_details = tkinter.Frame(master=self)
        self.frame_selection_details.pack(side=tkinter.TOP, fill=tkinter.X)
        self.initial_label = tkinter.ttk.Label(master=self.frame_selection_details, text="")
        self.initial_label.pack(side=tkinter.TOP, fill=tkinter.X, padx=(10, 10))

        self.group_name_var = tkinter.StringVar(master=self.frame_controls, value="")
        self.new_name_var = tkinter.StringVar(master=self.frame_controls, value="")

        frame_overview = tkinter.Frame(master=self)
        frame_overview.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

        frame_list = tkinter.Frame(master=frame_overview)
        frame_list.pack(side=tkinter.LEFT, fill=tkinter.Y, anchor="w")

        frame_charts = tkinter.Frame(master=frame_overview)
        frame_charts.pack(side=tkinter.RIGHT, fill=tkinter.BOTH, expand=True)


        fig = plt.figure(figsize=(5, 4), dpi=100)
        fig.subplots_adjust(left=0.18)
        self.graph_axis = fig.add_subplot()

        self.canvas = FigureCanvasTkAgg(fig, master=frame_charts)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(
            side=tkinter.TOP, fill=tkinter.BOTH, expand=True
        )

        self.canvas_toolbar = NavigationToolbar2Tk(
            self.canvas, frame_charts, pack_toolbar=False
        )
        self.canvas_toolbar.update()
        self.canvas_toolbar.pack(side=tkinter.TOP, fill=tkinter.X, padx=(10, 0))

        frame_canvas_options = tkinter.Frame(frame_charts)
        frame_canvas_options.pack(side=tkinter.TOP, fill=tkinter.X, padx=(10, 0))

        frame_canvas_options.columnconfigure(1, weight=1)

        # combobox = tkinter.ttk.Combobox(master=frame_canvas_options)
        # combobox.grid(row=0, column=0)

        button_clear = tkinter.ttk.Button(master=frame_canvas_options, text="Clear", command=self.clear_graph)
        button_clear.grid(row=0, column=2)


        self.tree_view_runs = tkinter.ttk.Treeview(master=frame_list, columns=["algorithm", "type", "id"], show="tree headings")
        self.tree_view_runs.heading("algorithm", text="Algorithm")
        self.tree_view_runs["displaycolumns"] = ("algorithm")
        self.tree_view_runs.pack(fill=tkinter.BOTH, expand=True)
        self.tree_view_runs.bind("<<TreeviewSelect>>", self.on_selection_change)

        self.update_tree_view()

    def update_tree_view(self):
        if len(self.tree_view_runs.get_children()) > 0:
            self.tree_view_runs.delete(*self.tree_view_runs.get_children())

        group_id_to_parent_id = {}
        for group_id in self.algorithm_stats.run_groups:
            group_name = self.algorithm_stats.run_groups[group_id].name
            parent_id = self.tree_view_runs.insert(
                "", tkinter.END, text=group_name, values=["", 1, group_id], open=False
            )
            group_id_to_parent_id[group_id] = parent_id

        for run_id in self.algorithm_stats.run_history:
            run = self.algorithm_stats.run_history[run_id]
            parent = "" if run.group is None else group_id_to_parent_id[run.group]
            self.tree_view_runs.insert(parent, tkinter.END, text=run.name, values=[run.algorithm, 0, run_id])

    def configure_canvas(self):
        self.graph_axis.set_xlabel("Iteration")
        self.graph_axis.set_ylabel("Best path")

    def clear_graph(self):
        self.displayed_objects.clear()
        self.graph_axis.cla()
        self.configure_canvas()
        self.canvas.draw()

    def create_group(self):
        group_name = self.group_name_var.get().strip()
        if not self.tree_view_runs.selection() or not group_name:
            return

        self.group_name_var.set("")

        ids = []
        for item in self.tree_view_runs.selection():
            ids.append(self.tree_view_runs.item(item)["values"][2])

        group_id = self.algorithm_stats.make_group(ids, group_name)
        parent_id = self.tree_view_runs.insert("", 0, text=group_name, values=["", 1, group_id], open=False)
        for item in self.tree_view_runs.selection():
            self.tree_view_runs.move(item, parent_id, tkinter.END)


    def rename_object(self):
        name = self.new_name_var.get().strip()
        if not self.tree_view_runs.selection() or not name:
            return

        self.new_name_var.set(name)
        self.tree_view_runs.item(self.tree_view_runs.selection()[0], text=name)

        item = self.tree_view_runs.item(self.tree_view_runs.selection()[0])
        is_group = bool(item["values"][1])
        if is_group:
            self.algorithm_stats.rename_group(item["values"][2], name)
        else:
            self.algorithm_stats.rename_run(item["values"][2], name)

    def ungroup(self):
        if not self.tree_view_runs.selection():
            return

        selected_group = self.tree_view_runs.selection()[0]
        group = self.tree_view_runs.item(selected_group)
        self.algorithm_stats.delete_group(group["values"][2])
        self.update_tree_view()

    def display_object(self):
        if not self.tree_view_runs.selection():
            return

        item = self.tree_view_runs.item(self.tree_view_runs.selection()[0])
        is_group = bool(item["values"][1])
        if is_group:
            total_history = []
            group = self.algorithm_stats.run_groups[item["values"][2]]
            for r in group.runs:
                total_history.append(self.algorithm_stats.run_history[r].best_len_history)

            shortest_history = min(len(h) for h in total_history)
            clipped_histories = [h[:shortest_history] for h in total_history]

            histories = numpy.array(clipped_histories)

            median = numpy.median(histories, axis=0)
            perc25 = numpy.percentile(histories, 25, axis=0)
            perc75 = numpy.percentile(histories, 75, axis=0)

            lines = self.graph_axis.plot(median, label=group.name)
            color = lines[0].get_color()
            self.graph_axis.fill_between(range(histories.shape[1]), perc25, perc75, color=color, alpha=0.3)
        else:
            run = self.algorithm_stats.run_history[item["values"][2]]
            best_path_history = run.best_len_history
            if len(best_path_history) == 1:
                self.graph_axis.scatter([0], best_path_history[0], label=run.name)
            elif len(best_path_history) > 0:
                self.graph_axis.plot(range(len(best_path_history)), best_path_history, label=run.name)

        self.graph_axis.legend()
        self.canvas.draw()


    def delete_objects(self):
        if not self.tree_view_runs.selection():
            return

        for i in self.tree_view_runs.selection():
            item = self.tree_view_runs.item(i)
            is_group = bool(item["values"][1])
            if is_group:
                self.algorithm_stats.delete_group(item["values"][2])
            else:
                self.algorithm_stats.delete_run(item["values"][2])

        self.update_tree_view()

    def clear_controls(self):
        for c in self.frame_controls.winfo_children():
            c.destroy()

    def multiple_runs_controls(self):
        label_group_name = tkinter.Label(master=self.frame_controls, text="New group name")
        label_group_name.grid(row=0, column=0, padx=(10, 10))

        entry_group_name = tkinter.ttk.Entry(master=self.frame_controls, textvariable=self.group_name_var)
        entry_group_name.grid(row=0, column=1, padx=(0, 10))

        create_group_button = tkinter.ttk.Button(master=self.frame_controls, text="Create", command=self.create_group)
        create_group_button.grid(row=0, column=2, padx=(0, 10))

        button_delete = tkinter.ttk.Button(master=self.frame_controls, text="Delete", command=self.delete_objects, style="delete_button.TButton")
        button_delete.grid(row=0, column=6, padx=(0, 10))

    def multiple_object_controls(self):
        button_delete = tkinter.ttk.Button(master=self.frame_controls, text="Delete", command=self.delete_objects, style="delete_button.TButton")
        button_delete.grid(row=0, column=6, padx=(0, 10))

    def object_controls(self, is_group : bool = False):
        label_name = tkinter.Label(master=self.frame_controls, text="Name")
        label_name.grid(row=0, column=0, padx=(10, 10))

        entry_name = tkinter.ttk.Entry(master=self.frame_controls, textvariable=self.new_name_var)
        entry_name.grid(row=0, column=1, padx=(0, 10))

        create_button = tkinter.ttk.Button(master=self.frame_controls, text="Rename", command=self.rename_object)
        create_button.grid(row=0, column=2, padx=(0, 10))

        button_display = tkinter.ttk.Button(master=self.frame_controls, text="Display", command=self.display_object)
        button_display.grid(row=0, column=3, padx=(0, 10))

        if is_group:
            button_split = tkinter.ttk.Button(master=self.frame_controls, text="Split", command=self.ungroup)
            button_split.grid(row=0, column=5, padx=(0, 10))

        button_delete = tkinter.ttk.Button(master=self.frame_controls, text="Delete", command=self.delete_objects, style="delete_button.TButton")
        button_delete.grid(row=0, column=6, padx=(0, 10))


    def on_selection_change(self, *args, **kwargs):
        selected_items_cnt = len(self.tree_view_runs.selection())
        self.clear_controls()
        if selected_items_cnt > 1:
            for item in self.tree_view_runs.selection():
                if self.tree_view_runs.parent(item) or self.tree_view_runs.item(item)["values"][1]:
                    return

            self.multiple_runs_controls()
        elif selected_items_cnt == 1:
            item = self.tree_view_runs.item(self.tree_view_runs.selection()[0])
            is_group = bool(item["values"][1])
            self.new_name_var.set(item["text"])
            self.object_controls(is_group)





class SettingsWindow(tkinter.Toplevel):
    def __init__(
        self,
        master,
        var_seed : tkinter.StringVar,
        var_fixed_seed : tkinter.IntVar,
        var_show_place_names : tkinter.IntVar,
        var_show_distances : tkinter.IntVar,
        var_show_pheromone_amount : tkinter.IntVar,
        var_continuous_updates : tkinter.IntVar,
        **kwargs
    ):
        super().__init__(master=master, **kwargs)

        self.title("Ant Algorithms - Advanced settings")
        self.transient(master)
        self.minsize(250, 250)

        label_seed = tkinter.ttk.Label(master=self, text="Seed")
        label_frame_seed_settings = tkinter.ttk.Labelframe(master=self, labelwidget=label_seed)
        label_frame_seed_settings.pack(side=tkinter.TOP, fill=tkinter.X, padx=(10, 10), pady=(10, 10))

        frame_custom_seed_settings = tkinter.Frame(master=label_frame_seed_settings)
        frame_custom_seed_settings.pack(side=tkinter.TOP, fill=tkinter.X)
        self.entry_seed = tkinter.ttk.Entry(
            master=frame_custom_seed_settings, textvariable=var_seed
        )
        self.entry_seed.grid(row=0, column=1, padx=(10, 10))
        self.button_use_seed = tkinter.ttk.Button(master=frame_custom_seed_settings, text="Use")
        self.button_use_seed.grid(row=0, column=2, padx=(0, 5))

        frame_fix_seed = tkinter.Frame(master=label_frame_seed_settings)
        frame_fix_seed.pack(side=tkinter.TOP, fill=tkinter.X)

        self.checkbox_fix_seed = create_checkbox(
            frame_fix_seed, "Fix seed", var_fixed_seed
        )
        self.checkbox_fix_seed.pack(side=tkinter.TOP, padx=(10, 10), pady=(10, 10), anchor="w")


        label_display_settings = tkinter.Label(master=self, text="View")
        label_frame_display_settings = tkinter.ttk.LabelFrame(master=self, labelwidget=label_display_settings)
        label_frame_display_settings.pack(side=tkinter.TOP, fill=tkinter.X, padx=(10, 10), pady=(0, 10))
        self.checkbox_show_place_names = create_checkbox(
            label_frame_display_settings, "Show place names", var_show_place_names
        )
        self.checkbox_show_place_names.pack(side=tkinter.TOP, padx=(10, 10), anchor="w")

        self.checkbox_show_distances = create_checkbox(
            label_frame_display_settings, "Show path distances", var_show_distances
        )
        self.checkbox_show_distances.pack(side=tkinter.TOP, padx=(10, 10), anchor="w")

        self.checkbox_show_pheromone_amount = create_checkbox(
            label_frame_display_settings, "Show pheromone amount", var_show_pheromone_amount
        )
        self.checkbox_show_pheromone_amount.pack(side=tkinter.TOP, padx=(10, 10), anchor="w")

        checkbox_continuous_updates = create_checkbox(
            label_frame_display_settings, "Continuous updates of charts", var_continuous_updates
        )
        checkbox_continuous_updates.pack(side=tkinter.TOP, padx=(10, 10), pady=(0, 10), anchor="w")

class ConvergenceWindow(tkinter.Toplevel):
    def __init__(
        self,
        master,
        best_path_history,
        **kwargs
    ):
        super().__init__(master=master, **kwargs)
        self.best_path_history = best_path_history

        self.title("Ant Algorithms - Convergence")
        self.transient(master)
        self.minsize(300, 300)

        frame_chart = tkinter.Frame(self)

        fig = plt.figure(figsize=(5, 4), dpi=100)
        fig.subplots_adjust(left=0.18)
        self.graph_axis = fig.add_subplot()

        self.canvas = FigureCanvasTkAgg(fig, master=frame_chart)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(
            side=tkinter.TOP, fill=tkinter.BOTH, expand=True
        )

        self.canvas_toolbar = NavigationToolbar2Tk(
            self.canvas, frame_chart, pack_toolbar=False
        )
        self.canvas_toolbar.update()
        self.canvas_toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X, padx=(10, 0))

        frame_controls = tkinter.Frame(master=self)
        frame_controls.columnconfigure(1, weight=1)

        frame_controls.pack(side=tkinter.BOTTOM, fill=tkinter.X)
        frame_chart.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

        self.draw(best_path_history)

    def configure_canvas(self):
        self.graph_axis.set_xlabel("Iteration")
        self.graph_axis.set_ylabel("Best path length")
        self.graph_axis.xaxis.get_major_locator().set_params(integer=True)

    def draw(self, best_path_history : list[float]):
        self.remove_graphs()
        self.configure_canvas()
        if len(best_path_history) == 1:
            self.graph_axis.scatter([0], best_path_history[0])
        elif len(best_path_history) > 0:
            self.graph_axis.plot(range(len(best_path_history)), best_path_history)
        self.canvas.draw()

    def clear_canvas(self):
        self.remove_graphs()
        self.canvas.draw()

    def remove_graphs(self):
        all_artists = self.graph_axis.lines + self.graph_axis.collections
        for artist in all_artists:
            artist.remove()
        self.graph_axis.set_prop_cycle(None)

class GUI:
    ANT_ALGORITHM_COMMON_PARAMS = {
        "iterations": (100, "Iterations", tkinter.IntVar, positive_integer),
        "ant_amount": (20, "Number Of Ants", tkinter.IntVar, positive_integer),
        "pheronome_w": (1.0, "Pheromone weight", tkinter.DoubleVar, valid_float),
        "visibility_w": (1.0, "Visibility weight", tkinter.DoubleVar, valid_float),
        "vaporization": (0.2, "Vaporization", tkinter.DoubleVar, float_one_to_zero),
    }

    ALGORIHTM_PARAMS = {
        "Ant System": {
            **ANT_ALGORITHM_COMMON_PARAMS,
        },
        "Ant Density": {
            **ANT_ALGORITHM_COMMON_PARAMS,
        },
        "Ant Quantity": {
            **ANT_ALGORITHM_COMMON_PARAMS,
        },
        "Ant Colony": {
            **ANT_ALGORITHM_COMMON_PARAMS,
            "exploitation_coef": (0.3, "Exploitation threshold ", tkinter.DoubleVar, float_one_to_zero),
        },
        "Elitist Strategy": {
            **ANT_ALGORITHM_COMMON_PARAMS,
        },
        "Min-Max Ant System": {
            **ANT_ALGORITHM_COMMON_PARAMS,
            "vaporization": (5e-5, "Vaporization", tkinter.DoubleVar, float_one_to_zero),
            "min_pheromone": (0.7, "Min pheromone", tkinter.DoubleVar, valid_float),
            "max_pheromone": (1.0, "Max pheromone", tkinter.DoubleVar, valid_float),
        },
    }

    ANNOTATION_FONT_DICT = {
        "backgroundcolor": "#ffffffd0",
        "color": "gray",
        "size": 7,
        "horizontalalignment": "center",
        "verticalalignment": "center",
    }

    SPEED_PRECISION = 4
    ITERATION_PER_SPEED_UPDATE = 10

    MODIFIED_PARAM_COLOR = "#00dd11"
    ERROR_PARAM_COLOR = "#ff0000"
    NORMAL_PARAM_COLOR = "#000000"

    PAUSED_STATUS_COLOR = "#000000"
    RUNNING_STATUS_COLOR = "#dd7700"
    FINISHED_STATUS_COLOR = "#00aa00"

    MAX_PARAM_IN_COLUMN = 5

    def __init__(self, algorithm_stats, logger : logging.Logger = None):
        self.root = tkinter.Tk()
        self.root.wm_title("Ant Algorithms")
        self.root.minsize(400, 700)

        self.build_toolbar()
        self.build_top_main_window_frame()
        self.build_bottom_main_window_frame()
        self.build_central_main_window_frame()

        self.var_seed = tkinter.IntVar(master=self.root, value=0)
        self.var_show_place_names = tkinter.IntVar(master=self.root, value=0)
        self.var_fixed_seed = tkinter.IntVar(master=self.root, value=0)
        self.var_show_distances = tkinter.IntVar(master=self.root, value=0)
        self.var_show_pheromone_amount = tkinter.IntVar(master=self.root, value=0)
        self.var_continuous_updates = tkinter.IntVar(master=self.root, value=0)

        self.on_use_custom_seed = None
        self.on_show_place_names = None
        self.on_show_distances = None
        self.on_show_pheromone_amount = None

        self.convergence_window = None

        self.logging_widget = None
        self.log = []
        self.algorithm_stats = algorithm_stats
        if logger is not None:
            text_handler = GUILogHandler(self, self.log)
            logger = logging.getLogger()
            logger.addHandler(text_handler)

    def build_top_main_window_frame(self):
        frame_top_main_window = tkinter.Frame(self.root)
        frame_top_main_window.pack(side=tkinter.TOP, fill=tkinter.X)

        frame_algorithm_data_file = tkinter.Frame(frame_top_main_window)
        frame_algorithm_data_file.pack(side=tkinter.TOP, fill=tkinter.X)
        frame_algorithm_data_file.columnconfigure(2, weight=1)

        self.var_algorithm = tkinter.StringVar()
        algorithm_label = tkinter.ttk.Label(
            master=frame_algorithm_data_file, text="Algorithm:"
        )
        algorithm_label.grid(row=0, column=0, padx=(10, 5), pady=(10, 0))
        self.combobox_algorithm = tkinter.ttk.Combobox(
            master=frame_algorithm_data_file,
            state="readonly",
            textvariable=self.var_algorithm,
        )
        self.combobox_algorithm.grid(row=0, column=1, pady=(10, 0))

        self.var_opened_file = tkinter.StringVar(
            master=frame_algorithm_data_file, value=""
        )
        self.opened_file_label = tkinter.ttk.Label(
            master=frame_algorithm_data_file, textvariable=self.var_opened_file
        )
        self.opened_file_label.grid(row=0, column=4, padx=(0, 5), pady=(10, 0))

        self.button_open_file = tkinter.ttk.Button(
            master=frame_algorithm_data_file, text="Open file"
        )
        self.button_open_file.grid(row=0, column=5, padx=(0, 10), pady=(10, 0))

        frame_display_options = tkinter.Frame(self.root)
        frame_display_options.pack(side=tkinter.TOP, fill=tkinter.X)
        frame_display_options.columnconfigure(2, weight=1)
        self.var_pheronomone = tkinter.IntVar(master=frame_display_options, value=0)

        self.var_best_path = tkinter.IntVar(master=frame_display_options, value=1)
        self.checkbox_best_path = create_checkbox(
            frame_display_options, "Show best path", self.var_best_path
        )
        self.checkbox_best_path.grid(row=0, column=0, padx=(10, 10), pady=(10, 0))

        self.var_pheronomone = tkinter.IntVar(master=frame_display_options, value=0)
        self.checkbox_pheromone = create_checkbox(
            frame_display_options, "Show pheromone", self.var_pheronomone
        )
        self.checkbox_pheromone.grid(row=0, column=1, padx=(0, 10), pady=(10, 0))

    def build_central_main_window_frame(self):
        frame_central_main_window = tkinter.Frame(self.root)
        frame_central_main_window.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

        frame_chart = tkinter.Frame(frame_central_main_window)
        frame_chart.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

        fig = plt.figure(figsize=(5, 4), dpi=100)
        self.graph_axis = fig.add_subplot()

        self.canvas = FigureCanvasTkAgg(fig, master=frame_chart)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(
            side=tkinter.TOP, fill=tkinter.BOTH, expand=True
        )

        self.canvas_toolbar = NavigationToolbar2Tk(
            self.canvas, frame_chart, pack_toolbar=False
        )
        self.canvas_toolbar.update()
        self.canvas_toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X, padx=(10, 0))

        frame_runinfo = tkinter.Frame(self.root)
        frame_runinfo.columnconfigure(4, weight=1)
        frame_runinfo.pack(side=tkinter.TOP, fill=tkinter.X)

        best_len_annotation = tkinter.ttk.Label(master=frame_runinfo, text="Best len:")
        best_len_annotation.grid(row=0, column=0, padx=(10, 5))

        self.var_best_len = tkinter.StringVar(master=frame_runinfo, value="--")
        self.label_best_len = tkinter.ttk.Label(
            master=frame_runinfo, textvariable=self.var_best_len
        )
        self.label_best_len.grid(row=0, column=1)

        self.status = tkinter.StringVar(master=frame_runinfo, value="")
        self.label_status = tkinter.ttk.Label(
            master=frame_runinfo, textvariable=self.status
        )
        self.label_status.grid(row=0, column=5, padx=(10, 10))

        frame_runstats = tkinter.Frame(self.root)
        frame_runstats.columnconfigure(4, weight=1)
        frame_runstats.pack(side=tkinter.TOP, fill=tkinter.X)

        self.var_iterations = tkinter.IntVar(master=frame_runstats, value=0)
        self.var_total_iterations = tkinter.IntVar(master=frame_runstats, value=0)
        label_iterations_annotation = tkinter.ttk.Label(
            master=frame_runstats, text="It.:"
        )
        label_iterations_annotation.grid(row=0, column=0, padx=(10, 5))
        self.label_iterations = tkinter.ttk.Label(
            master=frame_runstats, textvariable=self.var_iterations
        )
        self.label_iterations.grid(row=0, column=1)
        label_iterations_annotation_sep = tkinter.ttk.Label(
            master=frame_runstats, text="/"
        )
        label_iterations_annotation_sep.grid(row=0, column=2)

        label_iterations_annotation_total = tkinter.ttk.Label(
            master=frame_runstats, textvariable=self.var_total_iterations
        )
        label_iterations_annotation_total.grid(row=0, column=3)

        self.speed = tkinter.DoubleVar(master=frame_runstats, value=0)
        self.speed_label = tkinter.ttk.Label(master=frame_runstats, text="--")
        self.speed_label.grid(row=0, column=5, padx=(10, 0))

        speed_label_annot = tkinter.ttk.Label(master=frame_runstats, text="s/it")
        speed_label_annot.grid(row=0, column=6, padx=(0, 10))

    def build_bottom_main_window_frame(self):
        frame_bottom_main_window = tkinter.Frame(self.root)
        frame_bottom_main_window.pack(side=tkinter.BOTTOM, fill=tkinter.X)

        frame_controls = tkinter.Frame(frame_bottom_main_window)
        frame_controls.columnconfigure(3, weight=1)
        frame_controls.pack(side=tkinter.TOP, fill=tkinter.X)

        self.button_stop = tkinter.ttk.Button(master=frame_controls, text="Pause")
        self.button_stop.grid(row=0, column=0, padx=(10, 10), pady=(5, 10))

        self.button_step = tkinter.ttk.Button(master=frame_controls, text="Step")
        self.button_step.grid(row=0, column=1, padx=(0, 10), pady=(5, 10))

        self.button_run = tkinter.ttk.Button(master=frame_controls, text="Run")
        self.button_run.grid(row=0, column=2, padx=(0, 10), pady=(5, 10))

        self.button_reset = tkinter.ttk.Button(master=frame_controls, text="Reset")
        self.button_reset.grid(row=0, column=4, padx=(10, 10), pady=(5, 10))

        param_frame_label = tkinter.ttk.Label(
            master=frame_bottom_main_window, text="Parameters", foreground="gray"
        )

        label_frame_params = tkinter.ttk.Labelframe(
            frame_bottom_main_window, labelwidget=param_frame_label
        )
        label_frame_params.pack(
            side=tkinter.TOP, fill=tkinter.X, padx=(10, 10), pady=(10, 0)
        )

        self.param_frame = tkinter.Frame(master=label_frame_params)
        self.param_frame.grid(row=1, column=0, columnspan=2)
        self.param_dict = {}

        frame_param_controls = tkinter.Frame(frame_bottom_main_window)
        frame_param_controls.columnconfigure(2, weight=1)
        frame_param_controls.pack(side=tkinter.TOP, fill=tkinter.X)

        self.button_save = tkinter.ttk.Button(
            master=frame_param_controls, text="Save Params"
        )
        self.button_save.grid(row=0, column=0, padx=(10, 10), pady=(10, 10))

        self.button_restore = tkinter.ttk.Button(
            master=frame_param_controls, text="Restore Params"
        )
        self.button_restore.grid(row=0, column=1, padx=(0, 10), pady=(10, 10))

        modified_style = tkinter.ttk.Style(master=label_frame_params)
        modified_style.configure("modified_style.TEntry", foreground=self.MODIFIED_PARAM_COLOR)
        error_style = tkinter.ttk.Style(master=label_frame_params)
        error_style.configure("error_style.TEntry", foreground=self.ERROR_PARAM_COLOR)
        normal_style = tkinter.ttk.Style(master=label_frame_params)
        normal_style.configure("normal_style.TEntry", foreground=self.NORMAL_PARAM_COLOR)

        delete_button_style = tkinter.ttk.Style(master=label_frame_params)
        delete_button_style.configure(
            "delete_button.TButton",
            foreground="red",
        )

    def build_toolbar(self):
        self.toolbar = tkinter.Menu(self.root)
        self.root.config(menu=self.toolbar)
        self.file_menu = tkinter.Menu(self.toolbar, tearoff="off")
        self.file_menu.add_command(label="Save log", command=self.open_window_save_log)
        self.file_menu.add_separator()

        self.on_quit = None
        self.on_save_params = None
        self.on_save_params_with_seed = None
        self.load_params_cb = None
        self.file_menu.add_command(label="Save params", command=self.save_params)
        self.file_menu.add_command(
            label="Save params with seed", command=self.save_params_with_seed
        )
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Load params", command=self.load_params)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Quit", command=self.quit)

        self.toolbar.add_cascade(label="File", menu=self.file_menu)

        self.toolbar.add_command(
            label="Settings", command=self.open_window_advanced_settings
        )
        self.toolbar.add_command(label="Log", command=self.open_window_log)
        self.toolbar.add_command(label="Convergence", command=self.open_window_convergence)
        self.toolbar.add_command(label="History", command=self.open_window_history)

    def quit(self):
        if self.on_quit:
            self.on_quit()

    def clear_convergence(self):
        if self.convergence_window:
            self.convergence_window.clear()

    def set_quit_fn(self, quit_fn):
        self.on_quit = quit_fn
        self.root.protocol("WM_DELETE_WINDOW", quit_fn)

    def set_paused_status(self):
        self.status.set("Paused")
        self.label_status.configure(foreground=self.NORMAL_PARAM_COLOR)

    def set_finished_status(self):
        self.status.set("Finished")
        self.label_status.configure(foreground=self.FINISHED_STATUS_COLOR)

    def set_running_status(self):
        self.status.set("Running...")
        self.label_status.configure(foreground=self.RUNNING_STATUS_COLOR)

    def disable_speed_label(self):
        self.speed_label.configure(textvariable="")
        self.speed_label.configure(text="--")

    def enable_speed_label(self):
        self.speed_label.configure(textvariable=self.speed)

    def update_speed(self, iteration: int, cumulative_time: float):
        if iteration == 1:
            self.enable_speed_label()
            self.speed.set(round(cumulative_time, self.SPEED_PRECISION))
        elif iteration > 0 and iteration % self.ITERATION_PER_SPEED_UPDATE == 0:
            new_speed = cumulative_time / iteration  # Compute average speed
            self.speed.set(round(new_speed, self.SPEED_PRECISION))

    def update_best_path(self, new_best_path_len: float):
        self.var_best_len.set(f"{new_best_path_len:g}")

    def reset_best_path(self):
        self.var_best_len.set("--")

    def load_params(self):
        if self.load_params_cb:
            self.load_params_cb()

    def open_window_log(self):
        def on_window_close():
            self.logging_widget = None
            log_window.destroy()

        log_window = LogWindow(
            master=self.root,
            on_window_close=on_window_close,
            on_save_log=self.open_window_save_log,
            on_clear_log=self.clear_log,
            log_content=self.log
        )
        self.logging_widget = log_window.logging_widget

    def open_window_convergence(self):
        self.convergence_window = ConvergenceWindow(
            self.root,
            self.algorithm_stats.best_len_history
        )

    def open_window_history(self):
        self.history_window = HistoryWindow(
            self.root,
            self.algorithm_stats,
        )

    def update_history(self):
        self.history_window.update_tree_view()

    def open_window_save_log(self):
        timestamp = datetime.now().strftime("%m-%d-%H%M%S")
        alg_name = self.var_algorithm.get().replace(" ", "")
        ifilename = f"{alg_name}-{timestamp}.log"
        filename = tkinter.filedialog.asksaveasfilename(
            confirmoverwrite=True, title="Save Log As", initialfile=ifilename
        )

        if not filename:
            return

        fp = open(filename, mode="w")
        fp.write("\n".join(self.log) + "\n")
        fp.close()

    def save_params(self):
        if self.on_save_params is not None:
            self.on_save_params()

    def save_params_with_seed(self):
        if self.on_save_params_with_seed is not None:
            self.on_save_params_with_seed()

    def open_window_save_params(self, custom_str: str = ""):
        timestamp = datetime.now().strftime("%H%M%S")
        alg_name = self.var_algorithm.get().replace(" ", "")
        filename = f"{alg_name}-params{custom_str}-{timestamp}.json"
        return tkinter.filedialog.asksaveasfilename(
            confirmoverwrite=True, title="Save Params As", initialfile=filename
        )

    def open_window_advanced_settings(self):
        settings_window = SettingsWindow(
            master=self.root,
            var_seed=self.var_seed,
            var_fixed_seed=self.var_fixed_seed,
            var_show_place_names=self.var_show_place_names,
            var_show_distances=self.var_show_distances,
            var_show_pheromone_amount=self.var_show_pheromone_amount,
            var_continuous_updates=self.var_continuous_updates,
        )

        if self.on_use_custom_seed is not None:
            settings_window.button_use_seed.configure(command=self.on_use_custom_seed)
        if self.on_show_distances is not None:
            settings_window.checkbox_show_distances.configure(command=self.on_show_distances)
        if self.on_show_pheromone_amount is not None:
            settings_window.checkbox_show_pheromone_amount.configure(command=self.on_show_pheromone_amount)
        if self.on_show_place_names is not None:
            settings_window.checkbox_show_place_names.configure(command=self.on_show_place_names)

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
            x=[p[0] for p in data], y=[p[1] for p in data], zorder=zorder
        )

    def draw_data_names(self, data: list, data_names: list, zorder=90):
        for i, p in enumerate(data):
            if data_names[i] is None:
                continue

            self.graph_axis.text(
                x=p[0],
                y=p[1],
                s=f" {data_names[i]}",
                zorder=zorder,
                fontdict={"size": 9},
            )

    def redraw_canvas(self, to_draw):
        self.graph_axis.cla()

        for name in to_draw:
            draw_fn = to_draw[name]
            draw_fn()

        self.canvas.draw()

    def open_window_data_file(self):
        return tkinter.filedialog.askopenfilename(
            master=self.root,
            title="Select input file with data",
            filetypes=(
                ("JSON files", "*.json*"),
                ("CSV files", "*.csv*"),
                ("All files", "*.*"),
            )
        )

    def open_window_params_file(self):
        return tkinter.filedialog.askopenfilename(
            master=self.root,
            title="Select file with params",
            filetypes=(
                ("JSON files", "*.json*"),
                ("All files", "*.*")
            ),
        )

    def update_params(self, new_params: dict):
        for c in self.param_frame.winfo_children():
            c.destroy()

        self.param_dict.clear()
        for i, param_name in enumerate(new_params):
            default, label_text, var_type, validator = new_params[param_name]
            var_param_entry = var_type(master=self.param_frame, value=default)

            c = (i // self.MAX_PARAM_IN_COLUMN) * 2
            r = i % self.MAX_PARAM_IN_COLUMN

            label_param_entry = tkinter.ttk.Label(
                master=self.param_frame, text=label_text, anchor="e"
            )
            label_param_entry.grid(
                row=r, column=c, padx=(10, 10), pady=(0, 10), sticky="W"
            )

            param_entry = tkinter.ttk.Entry(
                name=param_name, master=self.param_frame, textvariable=var_param_entry
            )
            param_entry.grid(row=r, column=c + 1, padx=(0, 10), pady=(0, 10))

            var_param_entry.trace_add(
                "write",
                lambda *args, param_name=param_name: self.param_changed(param_name),
            )

            self.param_dict[param_name] = (var_param_entry, param_entry, validator)

    def param_changed(self, param_name: str):
        try:
            new_val = self.param_dict[param_name][1].get()
            self.param_dict[param_name][2](new_val)
        except Exception:
            self.param_dict[param_name][1].configure(style="error_style.TEntry")
        else:
            self.param_dict[param_name][1].configure(style="modified_style.TEntry")
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
            self.param_dict[param_name][1].configure(style="normal_style.TEntry")

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
        def _task():
            log_msg = self.format(record)
            self.log_list.append(log_msg)
            if self.gui.logging_widget is not None:
                self.gui.logging_widget.configure(state="normal")
                self.gui.logging_widget.insert(tkinter.END, f"{log_msg}\n")
                self.gui.logging_widget.configure(state="disabled")
                self.gui.logging_widget.see(tkinter.END)

        # Plan it as a tkinter task to avoid the delay of the algorithm
        self.gui.root.after(0, _task)
