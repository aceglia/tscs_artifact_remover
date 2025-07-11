import tkinter as tk

import scipy
from scipy import signal
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

import csv
from tkinter import filedialog
import numpy as np
from biosiglive import load


class ImageFrame:
    def __init__(self):
        self.is_txt_file = None
        self.data_for_svd = None
        self.is_data_loaded = False
        self.entries_window_label = []
        self.entries_window_widget = []
        self.toolbar = None
        self.file_path = ""
        self.init_data = None
        self.baseline = None
        self.time_vector = None
        self.chanels_names = None
        self.mep_data = None
        self.master = tk.Tk()
        self.canvas = None
        self.image_path = ""
        self.image = None
        self.axs = []
        self.fig = None
        self.hankel_matrix = None
        self.frame_has_changed = True
        self.muscle_has_changed = True
        self.window_size_has_changed = True
        self.fft_has_changed = True
        self.current_frame = 0
        self.current_muscle = 0
        self.window_times = [0.4, 0.5, 1, 1.2]
        self.end_fft = 15
        self.start_fft = 0
        self.init_figure_canvas()
        self._create_quit_button()
        self._create_next_button()
        self._create_previous_muscle_button()
        self._create_next_muscle_button()
        self._create_previous_button()
        self._create_open_file_button()
        self._create_slider()
        self._create_frame_box()


    def init_figure_canvas(self):
        self.fig = Figure()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.master, pack_toolbar=False)
        self.toolbar.update()

    def init_figure_fft(self):
        self.fig_fft = Figure()
        self.canvas_fft = FigureCanvasTkAgg(self.fig_fft, master=self.fft_drawing)
        self.canvas_fft.draw()
        self.toolbar_fft = NavigationToolbar2Tk(self.canvas_fft, self.fft_drawing, pack_toolbar=False)
        self.toolbar_fft.update()

    def _add_subplot(self, rows, cols, index):
        self.axs.append(self.fig.add_subplot(rows, cols, index, sharex=None))
        return self.axs[-1]

    def _update_figure(self):
        if not self.is_data_loaded:
            return
        for ax in self.axs:
            ax.clear()
        window_size = int(self.window_size)
        n_to_remove = int(self.n_signal_to_remove)
        # for k in range(self.init_data.shape[-1]):
        k = self.current_muscle
        data_tmp = self.data_for_svd[self.current_frame, :, k]
        # traject_matrix = np.column_stack([data_tmp[i: i + window_size] for i in range(len(data_tmp) - window_size)])
        if self.frame_has_changed or self.muscle_has_changed or self.window_size_has_changed:
            # down_sample_factor = data_tmp.shape[0] // window_size
            # self.hankel_matrix = scipy.linalg.hankel(data_tmp[:window_size], data_tmp[window_size-1:])
            self.hankel_matrix = np.column_stack([data_tmp[i: i + window_size] for i in range(len(data_tmp) - window_size)])
            self.frame_has_changed = False
            self.muscle_has_changed = False
            self.window_size_has_changed = False
            self.fft_has_changed = True
            traject_matrix = self.hankel_matrix.copy()
            U, S, Vh = scipy.linalg.svd(traject_matrix, full_matrices=False, check_finite=False, overwrite_a=True)
            self.U, self.S, self.Vh = U, S, Vh
        # if self.window_size_has_changed or self.frame_has_changed or self.muscle_has_changed:
        #     traject_matrix = self.hankel_matrix[:window_size, :-window_size].copy()
        #     self.window_size_has_changed = False
        # else:
        U, S, Vh = self.U.copy(), self.S.copy(), self.Vh.copy()
        count = 2
        for i in range(2, (len(self.signal_to_remove_idx) - 1) * 2):
            if i % 2 != 0:
                continue
            if self.signal_to_remove_idx[count].get() == 1:
                # U[:, i-2:i] = 0
                S[i-2:i] = 0
                # Vh[i-2:i, :] = 0
            count += 1
        signal_svd = U @ np.diag(S) @ Vh
        signal_svd = signal_svd[0, :]
        ref_to_plot = self.init_data[self.current_frame, :, k] if self.is_txt_file else self.init_data[self.current_frame, :, 0]
        self.axs[0].plot(ref_to_plot, alpha=0.3, color='r')
        self.axs[0].plot(signal_svd, alpha=0.8, color='g')
        self._show_error(signal_svd)
        T = 1 / 2000
        xf = np.fft.fftfreq(self.init_data[self.current_frame, :, k].shape[0], T)
        yf = np.fft.fft(self.init_data[self.current_frame, :, k])
        self.axs[1].plot(xf[xf > 0], np.abs(yf[xf > 0]), alpha=0.5, color='r')
        xf = np.fft.fftfreq(signal_svd.shape[0], T)
        yf = np.fft.fft(signal_svd)
        self.axs[1].plot(xf[xf > 0], np.abs(yf[xf > 0]), alpha=0.5, color='g')

        # pair S
        # s_pair = S[:-1:2]
        # s_odd = S[1::2]
        # s_mean = s_pair + s_odd / 2
        # s_mean = s_mean[S.shape[0]//4:]
        diff = S[:-1] - S[1:]
        # import matplotlib.pyplot as plt
        # plt.figure("first derivative")
        # plt.plot(diff)
        # diff_2 = diff[:-1] - diff[1:]
        # plt.figure("second derivative")
        # plt.plot(diff_2)
        # plt.show()
        idxs = np.argwhere(diff[:50] > diff[:50].mean() + 2*diff[:50].std())
        # self.axs[1].plot(S)
        # if idxs.shape[0] > 0:
        #     max_idx = int(idxs.max()) + 1
        #     self.axs[1].scatter(max_idx, S[max_idx], color='r')
        self.axs[0].set_title(self.chanels_names[k])
        self.canvas.draw()
        self.update_fft_plot()
        self.fft_has_changed = False

    def _show_error(self, signal_svd):
        from biosiglive import OfflineProcessing
        proc_emg = OfflineProcessing(2000)
        emg_enveloppe = proc_emg.process_emg(signal_svd[None, :], band_pass_filter=True,
        low_pass_filter=True,
        moving_average=False,
        centering=True,
        absolute_value=True)
        ratio = np.mean(emg_enveloppe[0, 5300:5600]) / np.mean(emg_enveloppe[0, 4500:5000])

        # signal_to_noise_ratio = np.mean(signal_svd) / np.std(signal_svd, ddof=1)
        if self.is_txt_file:
            y_min, y_max = self.axs[0].get_ylim()
            x_min, x_max = self.axs[0].get_xlim()
            self.axs[0].text(x_min, y_max - 0.05, f'emg/baseline: {ratio:.2f}')
        else:
            correlation = signal.correlate(signal_svd, self.init_data[self.current_frame, :signal_svd.shape[0], 0])
            lag = signal.correlation_lags(signal_svd.shape[0], self.init_data[self.current_frame, :signal_svd.shape[0], 0].shape[0])
            correlation /= np.max(correlation)
            final_lag = lag[np.argmax(correlation)]
            pearson = scipy.stats.pearsonr(signal_svd[20:], self.init_data[self.current_frame, 20:signal_svd.shape[0], 0])[0]        # signal_svd = self._butter_lowpass_filter(signal_svd, 450.0, 2000, order=4)
            n_frame_stim = 3 * 2000
            peak_to_peak_data = scipy.signal.find_peaks(np.abs(self.init_data[self.current_frame,
                                                        int(n_frame_stim + 0.1 * 2000):int(n_frame_stim + 0.4 * 2000), 0]), height=0.08)
            amplitude_ref = np.sum(peak_to_peak_data[1]["peak_heights"])
            peak_to_peak_data = scipy.signal.find_peaks(np.abs(signal_svd[int(n_frame_stim + 0.1 * 2000):int(n_frame_stim + 0.4 * 2000)]), height=0.08)
            peaks = [p for p in peak_to_peak_data[1]["peak_heights"] if p < 0.4]
            amplitude_proc = np.sum(peak_to_peak_data[1]["peak_heights"])
            peaks_error = amplitude_ref - amplitude_proc
            y_min, y_max = self.axs[0].get_ylim()
            x_min, x_max = self.axs[0].get_xlim()
            self.axs[0].text(x_min, y_max - 0.05,
                             f'pearson: {pearson:.4f}; lag: {int(final_lag)}; peaks diff: {peaks_error:.4f}; emg/baseline: {ratio:.2f}')

    @staticmethod
    def _load_txt_file(path, delimiter):
        frames = []
        with open(path, 'r') as file:
            reader = csv.reader(file, delimiter=delimiter)
            rows = []
            len_row = -1
            for row in reader:
                len_row = len(row) if len_row == -1 else len_row
                if len(row) != len_row:
                    frames.append(rows)
                    rows = []
                    continue
                rows.append(row)
        all_len = [len(row) for row in frames]
        frames = [row[:min(all_len)] for row in frames]
        array = np.array(frames)
        return array, frames

    def load_data(self, path, delimiter='\t'):
        array, frames = None, None
        if path.endswith('.txt'):
            array, frames = self._load_txt_file(path, delimiter)
        elif path.endswith('.bio'):
            data = load(path)
            frames = list(data.keys())
            for key in data.keys():
                if isinstance(data[key], np.ndarray):
                    array = data[key] if array is None else np.vstack((array, data[key]))
            array = array.T[None, ...]
        else:
            raise ValueError("File format not supported")
        self.is_txt_file = path.endswith('.txt')
        self.chanels_names = frames[0][0][1:] if path.endswith('.txt') else frames
        self.time_vector = array[0, 1:, 1:].astype(float) if path.endswith('.txt') else np.linspace(0, array.shape[1]/2000, array.shape[1])
        self.init_data = array[:, 1:, 1:].astype(float) if path.endswith('.txt') else array
        self.init_data -= np.mean(self.init_data, axis=1)[:, np.newaxis, :]
        self.init_data = self.filter_data(self.init_data, cutoff=450, order=2, filter_type='low')
        self.data_for_svd = self.init_data.copy()
        # self.data_for_svd = self.filter_data(self.init_data.copy(), cutoff=[30, 450], order=4, filter_type='band')
        # self.data_no_svd = self.filter_data(self.init_data.copy(), cutoff=30, order=4, filter_type='low')
        self.data_no_svd = np.zeros_like(self.data_for_svd)
        self.is_data_loaded = True
        n_rows = 2
        n_cols = 1  # int(self.init_data.shape[-1])
        for i in range(2):
            self._add_subplot(n_rows, n_cols, i+1)
        self.fft_drawing = tk.Toplevel(self.master)
        self.fft_window = tk.Toplevel(self.master)
        self.fft_drawing.title("FFT")
        self.init_figure_fft()
        self._create_next_fft_button()
        self._create_prev_fft_button()
        self._create_signal_check_buttons()
        n_rows = 15
        n_cols = 1  # int(self.init_data.shape[-1])
        self.axs_fft = []
        for i in range(n_rows):
            self.axs_fft.append(self.fig_fft.add_subplot(n_rows, n_cols, i+1, sharex=None))
        limit_per_column = 10
        count_col = 0
        count_row = 0
        self._create_update_signal_button()
        for i in range(len(self.signal_to_keep_buttons)):
            self.signal_to_keep_buttons[i].grid(row=count_row+1, column=count_col+1)
            count_row += 1
            if i % limit_per_column == 0:
                count_col += 1
                count_row = 0
        self.button_update_signal.grid(row=len(self.signal_to_keep_buttons) + 1, column=0)
        self.button_next_fft.pack(side=tk.BOTTOM, fill=tk.X)
        self.button_prev_fft.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas_fft.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar_fft.pack(side=tk.BOTTOM, fill=tk.X)


    @staticmethod
    def _butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def _butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self._butter_lowpass(cutoff, fs, order=order)
        y = signal.filtfilt(b, a, data)
        return y



    @staticmethod
    def _butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def _butter_highpass_filter(self, data, cutoff, fs, order=5):
        b, a = self._butter_highpass(cutoff, fs, order=order)
        y = signal.filtfilt(b, a, data)
        return y

    def filter_data(self, data, cutoff=450.0, order=5, filter_type='low'):
        if filter_type == 'low':
            filter_function = self._butter_lowpass_filter
        elif filter_type == 'band':
            filter_function = self._bandpass_filter
        elif filter_type == 'high':
            filter_function = self._butter_highpass_filter
        fs = 2000.0
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            for k in range(data.shape[-1]):
                filtered_data[i, :, k] = filter_function(data[i, :, k], cutoff, fs, order=order)
        return filtered_data

    def _create_quit_button(self):
        self.button_quit = tk.Button(master=self.master, text="Quit", command=self.master.destroy)

    def _create_slider(self):
        self.slider_signal = tk.Scale(self.master, from_=0, to=50, orient=tk.VERTICAL,
                                      command=self._update_signal, label="N signal")
        self.slider_signal.set(0)
        self.n_signal_to_remove = 0
        self.slider_window = tk.Scale(self.master, from_=30, to=1000, orient=tk.VERTICAL,
                                      command=self._update_window, label="Hankel window")
        self.slider_window.set(500)
        self.window_size = 500

    def _update_signal(self, value):
        self.n_signal_to_remove = int(value)
        self._update_figure()

    def _update_window(self, value):
        self.window_size = int(value)
        self.window_size_has_changed = True
        self._update_figure()

    def _create_frame_box(self):
        self.frame_box = tk.Text(self.master, width=2, height=1)
        self.frame_box.config(state=tk.NORMAL)
        self.frame_box.insert("end", str(self.current_frame))

    def _create_next_button(self):
        self.button_next = tk.Button(master=self.master, text="Next\nframe", command=self.next_frame)

    def _create_next_fft_button(self):
        self.button_next_fft = tk.Button(master=self.fft_drawing, text="Next\nFFTs", command=self.next_fft)

    def next_fft(self):
        if not self.is_data_loaded:
            return
        if self.start_fft >= self.Vh.shape[0]:
            return

        self.end_fft += 15
        self.start_fft = self.end_fft - 15
        self.end_fft = max(self.end_fft, 0)
        self.start_fft = max(self.start_fft, 0)
        self.fft_has_changed = True
        self.update_fft_plot()

    def _create_prev_fft_button(self):
        self.button_prev_fft = tk.Button(master=self.fft_drawing, text="Previous\nFFTs", command=self.prev_fft)

    def prev_fft(self):
        if not self.is_data_loaded:
            return
        self.end_fft -= 15
        self.start_fft = self.end_fft - 15
        self.end_fft = max(self.end_fft, 0)
        self.start_fft = max(self.start_fft, 0)
        self.fft_has_changed = True
        self.update_fft_plot()

    def update_fft_plot(self):
        if not self.fft_has_changed:
            return
        for ax in self.axs_fft:
            ax.clear()
        xf = np.fft.fftfreq(self.Vh.shape[1], 1/2000)
        count = 0
        for i in range(self.start_fft, min(self.end_fft, self.Vh.shape[0])):
            # f, t, Zxx = signal.stft(self.Vh[i, :], 2000, nperseg=20)
            # # plt.figure('STFT Magnitude')
            # self.axs_fft[count].pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
            yf = np.fft.fft(self.Vh[i, :])
            self.axs_fft[count].plot(xf[xf > 0], np.abs(yf[xf > 0]))
            self.axs_fft[count].set_ylabel(f"Signal {i+1}", rotation=0, labelpad=30)
            count += 1
        self.canvas_fft.draw()

    def _create_previous_button(self):
        self.button_previous = tk.Button(master=self.master, text="Previous\nframe", command=self.previous_frame)

    def _create_next_muscle_button(self):
        self.button_next_muscle = tk.Button(master=self.master, text="Next muscle", command=self.next_muscle)

    def _create_previous_muscle_button(self):
        self.button_previous_muscle = tk.Button(master=self.master, text="Previous muscle", command=self.previous_muscle)

    def _create_open_file_button(self):
        self.button_open_file = tk.Button(master=self.master, text="Open file", command=self.open_file)

    def _create_update_signal_button(self):
        self.button_update_signal = tk.Button(master=self.fft_window, text="update signal", command=self._update_signal_to_remove)

    def _create_signal_check_buttons(self):
        self.signal_to_keep_buttons = []
        self.signal_to_remove_idx = []
        for i in range(120):
            if i == 0:
                name = "All"
            elif i == 1:
                name = "Clear all"
            elif i % 2 != 0:
                continue
            else:
                name = str(i-1) + '-' + str(i)
            self.signal_to_remove_idx.append(tk.IntVar(value=0))
            self.signal_to_keep_buttons.append(tk.Checkbutton(self.fft_window, text=name,
                        variable=self.signal_to_remove_idx[-1],
                        onvalue=1,
                        offvalue=0,
                        height=2,
                        width=10))

    def _update_signal_to_remove(self):
        if self.signal_to_remove_idx[0].get() == 1:
            for i in range(2, len(self.signal_to_remove_idx)):
                self.signal_to_remove_idx[i].set(1)
            self.signal_to_remove_idx[0].set(0)
        elif self.signal_to_remove_idx[1].get() == 1:
            for i in range(2, len(self.signal_to_remove_idx)):
                self.signal_to_remove_idx[i].set(0)
            self.signal_to_remove_idx[1].set(0)
        self._update_figure()

    def next_frame(self):
        if not self.is_data_loaded:
            return
        if self.current_frame == self.init_data.shape[0] - 1:
            return
        self.current_frame += 1
        self.frame_has_changed = True
        self._update_figure()
        self.frame_box.delete(1.0, "end")
        self.frame_box.insert('end', str(self.current_frame))

    def previous_frame(self):
        if not self.is_data_loaded or self.current_frame == 0:
            return
        self.current_frame -= 1
        self.frame_has_changed = True
        self._update_figure()
        self.frame_box.delete(1.0, "end")
        self.frame_box.insert('end', str(self.current_frame))

    def next_muscle(self):
        if not self.is_data_loaded:
            return
        if self.current_muscle == self.init_data.shape[-1] - 1:
            return
        self.current_muscle += 1
        self.muscle_has_changed = True
        self._update_figure()

    def previous_muscle(self):
        if not self.is_data_loaded or self.current_muscle == 0:
            return
        self.current_muscle -= 1
        self.muscle_has_changed = True
        self._update_figure()

    def save_json(self):
        pass

    def open_file(self):
        self.file_path = filedialog.askopenfilename(initialdir="/", title="Select file", filetypes=(("txt files", "*.txt"), ("all files", "*.*"),
                                                                                                    ("bio files", "*.bio*")))
        if self.file_path == "":
            return
        self.load_data(self.file_path)
        self._update_figure()

    def run(self):
        # empty zone on the top of the window
        self.master.title("Annotation tool")
        self.button_quit.pack(side=tk.BOTTOM)
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        # self.button_open_file.pack(side=tk.TOP)
        self.button_open_file.place(x=1, y=0)
        self.button_previous.pack(side=tk.LEFT)
        self.button_next.pack(side=tk.LEFT)
        self.button_previous_muscle.pack(side=tk.TOP)
        self.button_next_muscle.pack(side=tk.TOP)
        self.frame_box.pack(side=tk.LEFT)
        self.slider_signal.pack(side=tk.RIGHT)
        self.slider_window.pack(side=tk.RIGHT)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        tk.mainloop()


if __name__ == '__main__':
    gui = ImageFrame()
    gui.run()

