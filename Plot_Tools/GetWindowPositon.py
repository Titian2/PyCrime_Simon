import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

def on_close():
    # Get the position and size of the window
    window_geometry = tk_window.geometry()
    print("Window geometry:", window_geometry)

    # Close the window
    root.destroy()

# Create a root Tkinter window
root = tk.Tk()
root.title("Move and Resize Plot Window")

# Create a figure and plot
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])

# Create a canvas to display the plot in the Tkinter window
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()

# Get the Tkinter window
tk_window = canvas.get_tk_widget().master

# Bind the close event to the on_close function
tk_window.protocol("WM_DELETE_WINDOW", on_close)

# Pack the canvas and start the Tkinter event loop
canvas.get_tk_widget().pack()
root.mainloop()
