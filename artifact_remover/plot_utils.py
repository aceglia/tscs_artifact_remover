import numpy as np

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt


class PlotSolution:
    def __init__(self, signals=True, fft=False, singular_values=False, channel_names=()):
        self.next_button = None
        self.previous_button = None
        self.root = tk.Tk()
        self._create_prev_next_buttons()
        self.plot_signals = signals
        self.plot_fft = fft
        self.plot_singular_values = singular_values
        self.nb_suplot = sum([self.plot_signals, self.plot_fft, self.plot_singular_values])
        fct_to_plot = [self._plot_curves, self._plot_fft, self._plot_singular_values]
        self.function_to_plot = [
            fct_to_plot[i]
            for i, to_plot in enumerate([self.plot_signals, self.plot_fft, self.plot_singular_values])
            if to_plot
        ]
        self.batch_idx = 0
        self.channel_idx = 0
        self.total_batch = 0
        self.channel_names = channel_names
        self.total_channel = len(channel_names)
        self.batch_idx = 0
        self.channel_idx = 0
        self.canvas, self.toolbar, self.fig, self.axes = self.init_figure_canvas(self.nb_suplot)

    def initialize(self, dict):
        for key, value in dict.items():
            setattr(self, key, value)

    def _create_prev_next_buttons(self):
        self.previous_button = tk.Button(master=self.root, text="Previous", command=self._previous_button_clicked)
        self.next_button = tk.Button(master=self.root, text="Next", command=self._next_button_clicked)
        self.previous_button.pack(side=tk.LEFT)
        self.next_button.pack(side=tk.LEFT)

    def _create_combo_boxes(self):
        self.channels_select = ttk.Combobox(self.root, values=self.channel_names, state="readonly")
        self.channels_select.set(self.channel_names[0])
        self.channels_select.bind("<<ComboboxSelected>>", self._on_combobox_select)
        self.channels_select.pack(side=tk.LEFT)

    def _on_combobox_select(self, event):
        selected_channel = self.channels_select.get()
        self.channel_idx = self.channel_names.index(selected_channel)
        self._plot_graphs()

    def _previous_button_clicked(self):
        self.batch_idx = np.min([self.batch_idx - 1, 0])
        self._plot_graphs()

    def _next_button_clicked(self):
        self.batch_idx = np.min([self.batch_idx + 1, self.total_batch - 1])
        self._plot_graphs()

    def init_figure_canvas(self, nb_subplot):
        fig, axes = plt.subplots(nb_subplot, 1)
        canvas = FigureCanvasTkAgg(fig, master=self.root)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, self.root, pack_toolbar=False)
        toolbar.update()
        axes = axes.flatten() if nb_subplot > 1 else [axes]
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        return canvas, toolbar, fig, axes

    def _plot_curves(self, ax):
        ax.plot(self.data["init_data"][self.batch_idx, self.channel_idx, :], label="Initial Signal", alpha=0.2)
        ax.plot(self.data["output"][self.batch_idx, self.channel_idx, :], label="Processed Signal")
        if "groundtruth_signals" in self.data:
            ax.plot(
                self.data["groundtruth_signals"][self.batch_idx, self.channel_idx, :],
                label="Groundtruth Signal",
                linestyle="--",
            )
        if self.analysis is not None:
            text = (
                "Signals"
                if self.analysis is None
                else "Signals - Analysis:"
                + f" RMSE = {float(self.analysis['rmse'][self.batch_idx, self.channel_idx]):.4f}"
                + f", Correlation = {float(self.analysis['correlation'][self.batch_idx, self.channel_idx]):.4f}"
                + f", Lag = {float(self.analysis['lag'][self.batch_idx, self.channel_idx]):.4f}"
            )
            ax.set_title(text)

    def _plot_fft(self, ax):
        init_fft = self.data["init_data"][self.batch_idx, self.channel_idx, :]
        processed_fft = self.data["output"][self.batch_idx, self.channel_idx, :]
        gt = (
            None
            if "groundtruth_signals" not in self.data
            else self.data["groundtruth_signals"][self.batch_idx, self.channel_idx, :]
        )
        mdfs = None
        if self.analysis is not None:
            mdfs = [
                float(self.analysis["mdf_init"][self.batch_idx, self.channel_idx]),
                float(self.analysis["mdf_processed"][self.batch_idx, self.channel_idx]),
            ]
        for data in [init_fft, processed_fft, gt]:
            if data is None:
                continue
            fft_data = np.fft.fft(data)
            freq = np.fft.fftfreq(data.shape[-1], 1 / 1925.928779153747)
            ax.plot(freq[freq > 0], np.abs(fft_data[freq > 0]))
        text = "FFT" if self.analysis is None else "FFT - Analysis:" + f" MDFs = {mdfs}"
        ax.set_title(text)
        ax.legend(["Initial Signal", "Processed Signal", "Groundtruth Signal"])

    def _plot_singular_values(self, ax):
        ax.plot(self.data["s"][self.batch_idx, self.channel_idx, :], label="Initial Signal")
        ax.plot(self.data["s_reduced"][self.batch_idx, self.channel_idx, :], label="Processed Signal")

    def _plot_graphs(self):
        count_axis = 0
        for fct in self.function_to_plot:
            if fct is None:
                continue
            ax = self.axes[count_axis]
            count_axis += 1
            ax.clear()
            fct(ax)
        self.canvas.draw()

    def plot(self, data: dict, stack_batch=False, analysis=None):
        self.analysis = analysis
        self.channel_names = (
            [f"Channel {i}" for i in range(data["init_data"].shape[1])]
            if len(self.channel_names) == 0
            else self.channel_names
        )
        self.total_channel = data["init_data"].shape[1]
        self.total_batch = data["init_data"].shape[0]
        self.data = {}
        for key, items in data.items():
            if items is None:
                continue
            if key == "u" or key == "v":
                continue
            if stack_batch:
                items = np.concatenate(items, axis=0)
                self.total_batch = 1
            self.data[key] = items
        self._create_combo_boxes()
        self._plot_graphs()
        tk.mainloop()
