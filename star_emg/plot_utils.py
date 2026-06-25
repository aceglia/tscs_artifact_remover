import numpy as np

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq


class PlotSolution:
    """
    Class to plot the solution of the EMG analysis.
    """

    def __init__(self, signals=True, fft=False, singular_values=False, channel_names=(), data_rate=2000):
        """
        Initialize the PlotSolution class.
        Parameters:
        ------------
        signals: bool
            Whether to plot the signals.
        fft: bool
            Whether to plot the FFT.
        singular_values: bool
            Whether to plot the singular values.
        channel_names: list of str
            The names of the channels to plot.
        data_rate: int
            The data rate of the signals, used for FFT plotting.

        Returns:
        --------
        None

        """
        self.next_button = None
        self.previous_button = None
        self.root = tk.Tk()
        self._create_prev_next_buttons()
        self.plot_signals = signals
        self.data_rate = data_rate
        self.plot_fft = fft
        self.plot_singular_values = singular_values
        self.nb_suplot = sum([self.plot_signals, self.plot_fft, self.plot_singular_values])
        fct_to_plot = [self._plot_curves, self._plot_fft, self._plot_singular_values]
        self.function_to_plot = [
            fct_to_plot[i]
            for i, to_plot in enumerate([self.plot_signals, self.plot_fft, self.plot_singular_values])
            if to_plot
        ]
        self.epochs_idx = 0
        self.channel_idx = 0
        self.total_epochs = 0
        self.channel_names = channel_names
        self.total_channel = len(channel_names)
        self.epochs_idx = 0
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
        self.epochs_idx = np.min([self.epochs_idx - 1, 0])
        self._plot_graphs()

    def _next_button_clicked(self):
        self.epochs_idx = np.min([self.epochs_idx + 1, self.total_epochs - 1])
        self._plot_graphs()

    def init_figure_canvas(self, nb_subplot):
        """
        Initialize the figure and canvas for plotting.
        Parameters:
        ------------
        nb_subplot: int
            The number of subplots to create.

        Returns:
        --------
        None
        """
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
        ax.plot(self.data["init_data"][self.epochs_idx, self.channel_idx, :], label="Initial Signal", alpha=0.2)
        ax.plot(self.data["output"][self.epochs_idx, self.channel_idx, :], label="Processed Signal")
        if "groundtruth_signals" in self.data:
            ax.plot(
                self.data["groundtruth_signals"][self.epochs_idx, self.channel_idx, :],
                label="Groundtruth Signal",
                linestyle="--",
            )
        text = "Signals" if self.analysis is None else "Analysis:\n"
        if self.analysis is not None:
            for n, name in enumerate(["raw", "process"]):
                text += (
                    name
                    + ": "
                    + f" Kurtosis = {float(self.analysis['kurtosis'][n][self.epochs_idx, self.channel_idx]):.4f}"
                    + f", Line Length = {float(self.analysis['Line Length'][n][self.epochs_idx, self.channel_idx]):.4f}"
                    + "\n"
                )
            # ax.set_title(text)
            ax.text(x=0, y=ax.get_ylim()[1], s=text, ha="left", va="top", fontsize=10)

    def _plot_fft(self, ax):
        init_fft = self.data["init_data"][self.epochs_idx, self.channel_idx, :]
        processed_fft = self.data["output"][self.epochs_idx, self.channel_idx, :]
        gt = (
            None
            if "groundtruth_signals" not in self.data
            else self.data["groundtruth_signals"][self.epochs_idx, self.channel_idx, :]
        )
        # get tab10 colors from matplotlib
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        alpha = [0.5, 1]
        for d, data in enumerate([init_fft, processed_fft, gt]):
            if data is None:
                continue
            fft_data = np.abs(rfft(data))
            freq = rfftfreq(data.shape[-1], 1 / self.data_rate)
            ax.plot(freq, fft_data, color=colors[d], alpha=alpha[d])

        if self.analysis is not None:
            for n, name in enumerate(["raw", "process"]):
                plt.vlines(
                    self.analysis["Median frequency"][n][self.epochs_idx, self.channel_idx],
                    0,
                    ax.get_ylim()[1],
                    color=colors[n],
                    linestyles="--",
                )
                plt.hlines(
                    self.analysis["FFT Amplitude"][n][self.epochs_idx, self.channel_idx],
                    0,
                    ax.get_xlim()[1],
                    color=colors[n],
                    linestyles="--",
                )

                # text = (name + ': ' +
                #     + f" Kurtosis = {float(self.analysis['kurtosis'][n][self.epochs_idx, self.channel_idx]):.4f}"
                #     + f", Line Length = {float(self.analysis['line_length'][n][self.epochs_idx, self.channel_idx]):.4f}"
                # )
            # ax.set_title(text)
            # ax.text(x=0, y=ax.get_ylim()[1], s=text, ha="left", va="top", fontsize=10)

        ax.set_title("FFT")
        ax.legend(["Initial Signal", "Processed Signal"])  # , "Groundtruth Signal"])

    def _plot_singular_values(self, ax):
        ax.plot(self.data["s"][self.epochs_idx, self.channel_idx, :], label="Initial Signal")
        ax.plot(self.data["s_reduced"][self.epochs_idx, self.channel_idx, :], label="Processed Signal")

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

    def plot(self, data: dict, stack_epochs=False, analysis=None):
        """
        Plot the data using a tkinter interface.
        Parameters:
        ------------
        data: dict
            The data to plot, which should contain the following keys:
            - init_data: np.ndarray
                The initial data to plot.
            - output: np.ndarray
                The processed data to plot.
            - groundtruth_signals: np.ndarray
                The groundtruth signals to plot.
            - s: np.ndarray
                The singular values to plot.
            - s_reduced: np.ndarray
                The reduced singular values to plot.
        stack_epochs: bool
            Whether to stack the epochs dimension of the data.
        analysis: dict, optional
            The analysis results to plot (from the Analysis class), which should contain the following keys:
            - kurtosis: np.ndarray
                The kurtosis values to plot.
            - Line Length: np.ndarray
                The line length values to plot.
            - Median frequency: np.ndarray
                The median frequency values to plot.
            - FFT Amplitude: np.ndarray
                The FFT amplitude values to plot.
        Returns:
        --------
        None
        """
        self.analysis = analysis
        self.channel_names = (
            [f"Channel {i}" for i in range(data["init_data"].shape[1])]
            if len(self.channel_names) == 0
            else self.channel_names
        )
        if data["init_data"].ndim != 3:
            data["init_data"] = data["init_data"][None]
        if data["output"].ndim != 3:
            data["output"] = data["output"][None]
        if "groundtruth_signals" in data and data["groundtruth_signals"].ndim != 3:
            data["groundtruth_signals"] = data["groundtruth_signals"][None]

        self.total_channel = data["init_data"].shape[1]
        self.total_epochs = data["init_data"].shape[0]
        self.data = {}
        for key, items in data.items():
            if items is None:
                continue
            if key == "u" or key == "v":
                continue
            if stack_epochs:
                items = np.concatenate(items, axis=0)
                self.total_epochs = 1
            self.data[key] = items
        self._create_combo_boxes()
        self._plot_graphs()
        tk.mainloop()
