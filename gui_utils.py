import tkinter as tk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure

class Window:
    def __init__(self):
        self.window_times = None
        self.entry_updated = False
        self.master = tk.Tk()
        self.entries_window_label = []
        self.entries_window_widget = []
        self.json_path = "_window_times.json"
        self.button_update_entry = None
        self._create_entries_window()
        self._create_entry_update_button()

    def init_figure_canvas(self, master):
        fig = Figure()
        canvas = FigureCanvasTkAgg(fig, master=master)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, master, pack_toolbar=False)
        toolbar.update()
        ax = fig.add_subplot(1, 1, 1, sharex=None)
        return canvas, toolbar, fig, ax

    def _update_entry(self):
        self.window_times = []
        for i in range(len(self.entries_window_widget)):
            entry_tmp = self.entries_window_widget[i].get()
            entry_tmp = float(entry_tmp)
            self.window_times.append(entry_tmp)
        self.entries_window_widget[i].delete(0, "end")
        self.entries_window_widget[i].insert(0, self.window_times[i])
        # save in json
        import json
        with open(self.json_path, 'w') as f:
            json.dump(self.window_times, f, indent=4)

        self.entry_updated = True

    def _create_entries_window(self):
        entry_names = ['Baseline start', 'Baseline end', 'EMG start', 'EMG end', 'Stim time']
        for i, name in enumerate(entry_names):
            self.entries_window_label.append(tk.Label(master=self.master, text=name))
            self.entries_window_widget.append(tk.Entry(master=self.master, width=8))
            if self.window_times is not None:
                self.entries_window_widget[i].insert(0, self.window_times[i])

    def _create_entry_update_button(self):
        self.button_update_entry = tk.Button(master=self.master, text="Validate", command=self._update_entry)

    def get_signal_idxs(self, emg_enveloppe, baseline_idx=None, signal_idx=None, json_path=None, original_signal=None):
        self.json_path = json_path if json_path is not None else self.json_path
        if baseline_idx is not None and signal_idx is not None:
            return baseline_idx, signal_idx
        tk.messagebox.showwarning("Warning", "You need to select the baseline and signal regions."
                                             "Please do it in the next window.")
        canvas, toolbar, fig, ax = self.init_figure_canvas(self.master)
        for i in range(len(self.entries_window_widget)):
            self.entries_window_label[i].pack(side=tk.TOP)
            self.entries_window_widget[i].pack()
        self.button_update_entry.pack(side=tk.BOTTOM)
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # while baseline_idx is None or signal_idx is None
        ax.plot(original_signal)
        ax.plot(emg_enveloppe)
        canvas.draw()
        tk.mainloop()


if __name__ == '__main__':
    import numpy as np
    import json
    Window().get_signal_idxs(np.random.rand(100))
