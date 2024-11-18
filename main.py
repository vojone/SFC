import tkinter

import sys
import json
import matplotlib.pyplot as plt

from ant_system import AntSystem
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk, FigureCanvasTkAgg

fp = open(sys.argv[1], "r")
places = json.load(fp)


ant_system = AntSystem(
    places,
    ant_amount=20,
    iterations=100,
    pheronome_w=1,
    visibility_w=1,
    vaporization=0.2
)

ant_system.start()
while ant_system.make_step():
    pass

best_path = ant_system.best_path
best_path_len = ant_system.best_path_len


fig = plt.figure(figsize=(5, 5), dpi=100)
ax = fig.add_subplot()
ax.scatter(x=[ p[1][0] for p in places], y=[ p[1][1] for p in places])

for i, p in enumerate(best_path):
    place_i = p
    next_place_i = best_path[i + 1 if i + 1 < len(best_path) else 0]
    place = ant_system.map.places[place_i]
    next_place = ant_system.map.places[next_place_i]
    ax.plot(
        [place.coords[0], next_place.coords[0]],
        [place.coords[1], next_place.coords[1]],
        color='r'
    )


root = tkinter.Tk()
root.wm_title("ACO")

def _quit():
    root.quit()
    root.destroy()

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

button_quit = tkinter.Button(master=root, text="Quit", command=_quit)
button_quit.pack(side=tkinter.BOTTOM)

toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
toolbar.update()
toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)

root.protocol('WM_DELETE_WINDOW', _quit)

tkinter.mainloop()


