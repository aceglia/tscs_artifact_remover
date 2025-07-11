import json

import os

import ctypes

from io_utils import load_txt_file, load_bio_file, handle_init_data
from typing import Union, List
from decomposition_utils import compute_svd, remove_singular_values
from processing_utils import compute_envelope, compute_signal_comparison, filter_data
import matplotlib.pyplot as plt
import numpy as np
from gui_utils import Window


class ArtefactRemover:
    def __init__(self, data: Union[str, List[str]] = None, plot_figure=False):
        # init some variables
        self.ratio = None
        self.init_data = None
        self.chanels_names = []
        self.u, self.s, self.v, self.hankel_matrix = None, None, None, None
        self.signal_reduced = None
        self.s_reduced = None

        # init flags
        self.is_txt_file = False
        self.is_data_loaded = False

        self.plot_figure = plot_figure
        if data is not None:
            self.load_data(data, delimiter='\t')


    def _load_init_data(self, data):
        data = handle_init_data(data, center=True, signal_filter=True)

    def _load_files(self, path, delimiter='\t', center=True, signal_filter=True, **kwargs):
        if path.endswith('.txt'):
            self.is_txt_file = path.endswith('.txt')
            return load_txt_file(path, delimiter, center=center, signal_filter=signal_filter, **kwargs)
        elif path.endswith('.bio'):
            return load_bio_file(path, center=center, signal_filter=signal_filter, **kwargs)
        else:
            raise ValueError("File format not supported")

    def load_data(self, path, delimiter='\t'):
        if isinstance(path, str):
            self.init_data, self.chanels_names = self._load_files(path, delimiter, center=True, signal_filter=True)
            self.is_data_loaded = True
        elif isinstance(path, np.ndarray):
            self.chanels_names = ['chanel_{}'.format(i) for i in range(path.shape[-1])]
            self.init_data = handle_init_data(path, center=True, signal_filter=True)
            self.is_data_loaded = True


    def get_signal_from_hankel(self, hankel):
        # reconstruct the signal from the hankel matrix using the average of overlapping wndows
        close = hankel.copy()
        max_row, max_col = close.shape
        # Create a list to store anti-diagonals
        flip_matrix = np.fliplr(close)
        offsets = np.arange(max_col- 1, -max_row, -1)
        antidiag = [flip_matrix.diagonal(offset=offset) for offset in offsets]
        reconstructed_signal = np.array([np.mean(antidiag[i]) for i in range(len(antidiag))])
        return reconstructed_signal

    def signal_decomposition(self, data, hankel_size=None, artefactless_signal=None, threshold=None,
                             idx=0, window=None, color=None):
        if not self.is_data_loaded and data is None:
            raise ValueError("Data not loaded")
        # compute rfft
        data_init = data.copy()
        window_init = window
        if window is not None:
            window_init = window.copy()
            window[0] = int(window[0] - hankel_size)
            window[1] = int(window[1] + hankel_size)
            data = data[int(window[0]):int(window[1])]
        self.u, self.s, self.v, self.hankel_matrix = compute_svd(data, n_rows=hankel_size,
                                                                 hankel=None)
        self.s_reduced = remove_singular_values(self.v.copy(), self.s.copy(), threshold=threshold, n_points=50)
        self.signal_reduced = self.get_signal_from_hankel(self.u.copy() @ np.diag(self.s_reduced) @ self.v.copy())
        self.signal_reduced = filter_data(self.signal_reduced[None, :, None])[0, :, 0]
        if window_init is not None:
            signal_reduced = data_init
            signal_reduced[int(window_init[0]):int(window_init[1])] = self.signal_reduced[int(hankel_size):-int(hankel_size)]
            self.signal_reduced = signal_reduced

        if self.plot_figure:
            plt.figure("Singular values")
            plt.plot(self.s, label="Original", color='r')
            plt.plot(self.s_reduced, label="Reduced", color='b')
            # plt.plot(self.s_reduced_flip, label="Reduced flipped", color='g')
            color = 'b' if not color else color
            plt.figure("Signal reduced")
            plt.plot(data_init, label="Original", color='r', alpha=0.3)
            if artefactless_signal is not None:
                plt.plot(artefactless_signal, label="Without artefacts", color='g', alpha=0.5)
            # plt.plot(self.signal_reduced_zero, label="Signal reduced before", color='k')
            # plt.plot(self.signal_reduced_flip_zero, label="Signal reduced before", color='r')
            plt.plot(self.signal_reduced, label=f"Signal reduced_{idx}", color=color)


    def compute_signal_error(self, original_signal, reduced_signal, baseline_idx=None, signal_idx=None, artefactless_signal=None, stim_time=0,
                             json_path=None):
        # emg_envelope = compute_envelope(reduced_signal)
        # emg_envelope_original = compute_envelope(original_signal)
        reduced_signal_rectified = np.abs(reduced_signal)
        original_signal_rectified = np.abs(original_signal)
        artifact_free_rectified = np.abs(artefactless_signal) if artefactless_signal is not None else None
        # plt.figure("Signal reduced")
        # plt.plot(emg_envelope_original, label="original_signal envelope", color='r', alpha=0.5)
        # plt.plot(emg_envelope, label="Signal reduced envelope", color='b')
        # plt.show()
        stim_time_file = None
        if json_path is not None and os.path.exists(json_path):
            with open(json_path, 'r') as f:
                config = json.load(f)
            baseline_idx, signal_idx, stim_time_file = [config[0], config[1]], [config[2], config[3]], config[4]
        else:
            path = json_path if json_path is not None else '_window_times.json'
            Window().get_signal_idxs(reduced_signal_rectified, baseline_idx, signal_idx, json_path=path,
                                     original_signal=reduced_signal)
            if baseline_idx is None or signal_idx is None:
                with open(path, 'r') as f:
                    config = json.load(f)
                baseline_idx, signal_idx, stim_time_file = [config[0], config[1]], [config[2], config[3]], config[4]

        # ratio = np.mean(emg_envelope[int(signal_idx[0]):int(signal_idx[1])]) / np.mean(
        # emg_envelope[int(baseline_idx[0]):int(baseline_idx[1])])
        # original_ratio = np.mean(emg_envelope_original[int(signal_idx[0]):int(signal_idx[1])]) / np.mean(
        #     emg_envelope_original[int(baseline_idx[0]):int(baseline_idx[1])])
        signal_reduced = reduced_signal_rectified[int(signal_idx[0]):int(signal_idx[1])]
        signal_original = original_signal_rectified[int(signal_idx[0]):int(signal_idx[1])]
        signal_artifactfree = artifact_free_rectified[int(signal_idx[0]):int(signal_idx[1])] if artifact_free_rectified is not None else None
        baseline_reduced = reduced_signal_rectified[int(baseline_idx[0]):int(baseline_idx[1])]
        baseline_original = original_signal_rectified[int(baseline_idx[0]):int(baseline_idx[1])]
        baseline_artifactfree = artifact_free_rectified[int(baseline_idx[0]):int(baseline_idx[1])] if artifact_free_rectified is not None else None
        shape_baseline = baseline_reduced.shape[0] // 4
        shape_signal = signal_reduced.shape[0] // 4
        shape_to_take = min(shape_baseline, shape_signal)
        ratio = np.mean(-np.sort(-signal_reduced)[:shape_to_take]) / np.mean(-np.sort(-baseline_reduced)[:shape_to_take])
        original_ratio = np.mean(-np.sort(-signal_original)[:shape_to_take]) / np.mean(-np.sort(-baseline_original)[:shape_to_take])
        artifactfree_ratio = None
        if artefactless_signal is not None:
            artifactfree_ratio = np.mean(-np.sort(-signal_artifactfree)[:shape_to_take]) / np.mean(-np.sort(-baseline_artifactfree)[:shape_to_take])
        self.ratio = ratio
        self.initial_ratio = original_ratio
        self.artefactless_ratio = artifactfree_ratio if artefactless_signal is not None else None
        text = f"emg/baseline: {ratio:.2f} (vs: {original_ratio:.2f})"
        delay = int(stim_time + 0.016 * 2000)
        delay_end = int(stim_time + 0.025 * 2000)
        text += f"; max: {max(reduced_signal[delay:delay_end])}"
        if artefactless_signal is not None:
            if stim_time_file is not None:
                stim_time = stim_time_file
            elif stim_time is None:
                stim_time = 0
            pearson, final_lag, peaks_error = compute_signal_comparison(reduced_signal, artefactless_signal, stim_time)
            self.pearson = pearson
            text += f"; pearson: {pearson:.4f}; lag: {int(final_lag)}; peaks diff: {peaks_error:.5f}"
            initial_pearson, final_lag, peaks_error = compute_signal_comparison(reduced_signal, original_signal, stim_time)
            self.initial_pearson = initial_pearson
        if self.plot_figure:
            y_min, y_max = plt.ylim()
            x_min, x_max = plt.xlim()
            plt.text(x_min, y_max - 0.05, text)
        print(text)

    def compute_frequency_analysis(self, original_signal, reduced_signal, artefactless_signal=None):
        if not self.is_data_loaded:
            raise ValueError("Data not loaded")
        data_to_compute = [original_signal, reduced_signal]
        if artefactless_signal is not None:
            data_to_compute.extend([artefactless_signal])
        data_name = ["With artefacts", "Reduced", "Without artefacts"]
        text = ""
        if self.plot_figure:
            plt.figure("Frequency analysis")
        mdfs = []
        self.mdfs = []
        for i in range(len(data_to_compute)):
            data = data_to_compute[i]
            fft_data = np.fft.fft(data)
            freq = np.fft.fftfreq(len(data), 1 / 2000)
            if self.plot_figure:
                # plt.hist(np.abs(fft_data[freq > 0]), bins=100, color='skyblue', edgecolor='black')
                plt.plot(np.abs(fft_data[freq > 0]), label=data_name[i])
            amp = np.abs(fft_data[freq > 0])
            energy = amp ** 2
            energy_cumsum = np.cumsum(energy)
            mdfs.append(freq[np.where(energy_cumsum > np.max(energy_cumsum) / 2)[0][0]])
            text += f"{data_name[i]}: MDF: {mdfs[-1]:.2f} Hz\n"
            self.mdfs.append(mdfs[-1])
        # self.mdfs = mdfs
        print(self.hankel_matrix.shape)
        print(text)

    def plot(self):
        if not self.plot_figure:
            return
        plt.legend()
        plt.show()




if __name__ == '__main__':
    synth = False
    # path_file = "synth_stim_artifact.bio" if synth else r"test_stim_artifact.txt"
    path_file = "synth_stim_artifact_all.bio" if synth else r"test001.txt"
    frame_idx = 0 if synth else 14  # index of the frame to process
    # json_path = 'windows_synth_signal.json' if synth else 'windows_test_signal.json'
    json_path = 'synth_data_window.json' if synth else f'test_emg_{frame_idx}.json'
    idx_to_remove = 0 if synth else 0  # index of the channel to remove
    path_file = r"D:\Documents\Udem\Postdoctorat\Projet transfert nerveux\data\test_HB_001\test_mapping_HB005.txt"

    artefact_remover = ArtefactRemover(data=path_file, plot_figure=True)
    signal_to_remove = artefact_remover.init_data[frame_idx, :, idx_to_remove]
    artefactless_signal = artefact_remover.init_data[frame_idx, :, 0] if synth else None
    # txt_file = "result_optim_synth_data_with_artifact_all.txt" 

    size = 500
    # if os.path.exists(txt_file):
    #     with open(txt_file) as f:
    #         data = f.readlines()
    #         size = int(float(data[-1].split(",")[1]))
    artefact_remover.signal_decomposition(signal_to_remove,
                                          hankel_size=size, artefactless_signal=artefactless_signal,
                                          threshold=None)
    artefact_remover.plot()
    # artefact_remover.signal_decomposition(artefact_remover.signal_reduced,
    #                                     hankel_size=size, artefactless_signal=artefactless_signal,
    #                                     backward_pass=True, threshold=10, color='g')
    #
    # size = 400
    # artefact_remover.signal_decomposition(artefact_remover.signal_reduced,
    #                                       hankel_size=size,
    #                                       idx=1, artefactless_signal=artefactless_signal,
    #                                       threshold=7, backward_pass=True, window=[3500, 3900], color='g')
    # plot short fourrier transform
    # plt.figure("Frequency analysis")
    # from scipy import signal
    # f, t, Zxx = signal.stft(artefact_remover.signal_reduced, 2000, nperseg=20)
    # plt.figure('STFT Magnitude')
    #
    # plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    artefact_remover.compute_signal_error(
        original_signal=signal_to_remove,
        reduced_signal=artefact_remover.signal_reduced, artefactless_signal=artefactless_signal,
        json_path=json_path)
    artefact_remover.compute_frequency_analysis(
        original_signal=signal_to_remove,
        reduced_signal=artefact_remover.signal_reduced,
        artefactless_signal=artefactless_signal)
    

    artefact_remover.plot()
