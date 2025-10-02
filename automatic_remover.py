import json

import os

import time

import scipy

from io_utils import load_txt_file, load_bio_file, handle_init_data
from typing import Union, List
from decomposition_utils import compute_svd, remove_singular_values
from processing_utils import compute_signal_comparison, filter_data
import matplotlib.pyplot as plt
import numpy as np


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

    @staticmethod
    def get_signal_from_hankel_fast(hankel: np.ndarray) -> np.ndarray:
        # Flip left-right to make antidiagonals become diagonals
        flip_matrix = np.fliplr(hankel)

        # Extract all possible anti-diagonal sums in one go
        # summed = np.add.reduceat(
        #     flip_matrix.ravel(),
        #     np.r_[0, np.cumsum(np.repeat(np.arange(1, flip_matrix.shape[1]+1), 1))[:-1]]
        # )

        # Alternative simpler + clearer method:
        # summed = np.array([flip_matrix.diagonal(offset=o).mean() 
        #                    for o in range(flip_matrix.shape[1]-1, -flip_matrix.shape[0], -1)])

        # Instead of looping: use bincount to compute mean per antidiagonal
        rows, cols = np.indices(flip_matrix.shape)
        antidiag_index = rows + cols   # constant along antidiagonals

        summed = np.bincount(antidiag_index.ravel(), weights=flip_matrix.ravel())
        counts = np.bincount(antidiag_index.ravel())
        
        return summed / counts
    
    def all_antidiagonals(arr):
        m, n = arr.shape
        flat = arr.ravel()
        rows, cols = np.indices((m, n))
        
        # group by (row+col), which is constant on antidiagonals
        keys = (rows + cols).ravel()
        order = np.argsort(keys)
        
        flat_sorted = flat[order]
        keys_sorted = keys[order]
        
        # split into diagonals
        splits = np.flatnonzero(np.diff(keys_sorted)) + 1
        return np.split(flat_sorted, splits)

    def get_signal_from_hankel(self, hankel):
        # close = hankel.copy()
        max_row, max_col = hankel.shape
        flip_matrix = np.fliplr(hankel)
        offsets = np.arange(max_col- 1, -max_row, -1)
        reconstructed_signal = np.array([np.mean(flip_matrix.diagonal(offset=offset)) for offset in offsets])
        return reconstructed_signal

    def signal_decomposition(self, data, hankel_size=None, artefactless_signal=None, threshold=None,
                             idx=0, window=None, color=None, randomized=True, hankel=None):
        if not self.is_data_loaded and data is None:
            raise ValueError("Data not loaded")
        # compute rfft
        # data_init = data.copy()
        # window_init = window
        # if window is not None:
        #     window_init = window.copy()
        #     window[0] = int(window[0] - hankel_size)
        #     window[1] = int(window[1] + hankel_size)
        #     data = data[int(window[0]):int(window[1])]
        self.u, self.s, self.v, self.hankel_matrix = compute_svd(data, n_rows=hankel_size,
                                                                 hankel=hankel, randomized=randomized)
        self.s_reduced = remove_singular_values(self.v, self.s, threshold=threshold, n_points=50)
        # self.signal_reduced = self.get_signal_from_hankel(self.u @ np.diag(self.s_reduced) @ self.v)
        self.signal_reduced = self.get_signal_from_hankel((self.u * self.s_reduced) @ self.v)
        self.signal_reduced = filter_data(self.signal_reduced[None, :, None])[0, :, 0]
        # if window_init is not None:
        #     signal_reduced = data_init
        #     signal_reduced[int(window_init[0]):int(window_init[1])] = self.signal_reduced[int(hankel_size):-int(hankel_size)]
        #     self.signal_reduced = signal_reduced

        if self.plot_figure:
            plt.figure("Singular values")
            plt.plot(self.s, label="Original", color='r')
            plt.plot(self.s_reduced, label="Reduced", color='b')
            # plt.plot(self.s_reduced_flip, label="Reduced flipped", color='g')
            color = 'b' if not color else color
            plt.figure("Signal reduced")
            plt.plot(data, label="Original", color='r', alpha=0.3)
            if artefactless_signal is not None:
                plt.plot(artefactless_signal, label="Without artefacts", color='g', alpha=0.5)
            # plt.plot(self.signal_reduced_zero, label="Signal reduced before", color='k')
            # plt.plot(self.signal_reduced_flip_zero, label="Signal reduced before", color='r')
            plt.plot(self.signal_reduced, label=f"Signal reduced_{idx}", color=color)
        return self.signal_reduced


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
    synth = True
    # path_file = "synth_stim_artifact.bio" if synth else r"test_stim_artifact.txt"
    path_file = "synth_stim_artifact_new.bio" if synth else r"test001.txt"
    frame_idx = 0 if synth else 7  # index of the frame to process
    # json_path = 'windows_synth_signal.json' if synth else 'windows_test_signal.json'
    json_path = 'synth_data_window.json' if synth else f'test_emg_{frame_idx}.json'
    idx_to_remove = 5 if synth else 0  # index of the channel to remove
    # path_file = r"D:\Documents\Udem\Postdoctorat\Projet transfert nerveux\data\test_HB_001\test_mapping_HB005.txt"

    artefact_remover = ArtefactRemover(data=path_file, plot_figure=False)
    # for frame_idx in range(artefact_remover.init_data.shape[0]):
    #     plt.plot(artefact_remover.init_data[frame_idx, :, 0], label="Original signal")
    #     plt.show()

    signal_to_remove = artefact_remover.init_data[frame_idx, :, idx_to_remove]
    artefactless_signal = artefact_remover.init_data[frame_idx, :, 0] if synth else None
    # txt_file = "result_optim_synth_data_with_artifact_all.txt" 

    size = 350
    # if os.path.exists(txt_file):
    #     with open(txt_file) as f:
    #         data = f.readlines()
    #         size = int(float(data[-1].split(",")[1]))
    update_wind = 20
    signal_cleaned = np.zeros_like(signal_to_remove)
    signal_filtered = np.zeros_like(signal_to_remove)
    signal_to_filter = np.zeros_like(signal_to_remove)
    wind_size_tot_svd = 1500
    wind_size_tot = 8000

    def append_data(data, data_appended):
        # if data_appended size reach wind_size_tot_svd, discard old data and append new data
        if data_appended is None:
            data_appended = data
        else:
            data_appended = np.concatenate((data_appended[-(wind_size_tot_svd - data.shape[0]):], data), axis=0)
        return data_appended
    
    # signal_to_remove = np.arange(1520)
    # start_time = time.time()
    # # scipy.linalg.hankel(new_data[::-1][1:], new_data)
    # hankel_tot = scipy.linalg.hankel(signal_to_remove[:int(size)], signal_to_remove[int(size - 1):])
    # print("hankel --- %s seconds ---" % (time.time() - start_time))
    # new_data = signal_to_remove[1500:]
    # hankel_ref = scipy.linalg.hankel(signal_to_remove[:1500][:int(size)], signal_to_remove[:1500][int(size - 1):])
    import time

    signal_to_remove_tmp = None
    for i in range(signal_to_remove.shape[0] // update_wind):
        new_data = signal_to_remove[i*update_wind:(i+1)*update_wind]
        signal_to_remove_tmp = append_data(new_data, signal_to_remove_tmp)
        start_time = time.time()
        if signal_to_remove_tmp.shape[0] >= wind_size_tot_svd:
            artefact_remover.signal_decomposition(signal_to_remove_tmp,
                                                hankel_size=size, artefactless_signal=artefactless_signal,
                                                threshold=None, randomized=True)
            signal_cleaned[i*update_wind:(i+1)*update_wind] = artefact_remover.signal_reduced[-update_wind:]
        print("SVD --- %s seconds ---" % (time.time() - start_time))

    signal_to_remove_tmp = None
    for i in range(signal_to_remove.shape[0] // update_wind):
        start_time = time.time()
        if signal_to_remove_tmp is None:
            signal_to_remove_tmp = signal_to_remove[(i)*update_wind:(i+1)*update_wind]
        else:
            signal_to_remove_tmp = np.concatenate((signal_to_remove_tmp[-wind_size_tot + update_wind:],
                                                    signal_to_remove[(i)*update_wind:(i+1)*update_wind]),
                                                      axis=0)
            
        # if signal_to_remove_tmp.shape[0] > wind_size_tot_svd:
        original_fft = np.fft.fft(signal_to_remove_tmp)

        # get peaks in fft usng moving windows and find peak function in scipy
        freq = np.fft.fftfreq(len(original_fft), 1 / 2000)
        # if signal_to_remove_tmp.shape[0] == wind_size_tot:
        #     plt.plot(freq, np.abs(original_fft))
        #     plt.show()
        fft = np.abs(original_fft[freq > 0])
        freq = freq[freq > 0]
        wind_size = 200
        total_wind = int(len(fft) / wind_size)
        total_idx = 0
        peaks = []
        for w in range(total_wind):
            mean = np.mean(fft[w*wind_size:(w+1)*wind_size])
            sd = np.std(fft[w*wind_size:(w+1)*wind_size])
            # find peaks in fft 
            peaks_tmp, _ = scipy.signal.find_peaks(fft[w*wind_size:(w+1)*wind_size], height=mean + 8*sd)
            peaks.extend([p + w*wind_size for p in peaks_tmp])
        fs = 2000
        filtered_signal_tmp = signal_to_remove_tmp.copy()
        for p in peaks:
            f_noise = freq[p]
            w0 = f_noise 
            Q = 80  # Quality factor (determines bandwidth)

            # Design the notch filter
            b, a = scipy.signal.iirnotch(w0, Q, fs=fs)

            # Apply the filter to the signal
            filtered_signal_tmp = scipy.signal.filtfilt(b, a, filtered_signal_tmp)
        signal_filtered[i*update_wind:(i+1)*update_wind] = filtered_signal_tmp[-update_wind:]
        print("filter --- %s seconds ---" % (time.time() - start_time))

    plt.plot(signal_to_remove, label="Original signal", alpha=0.3)
    plt.plot(signal_filtered, label="Filtered signal")
    plt.plot(signal_cleaned, label="Signal reduced")
    plt.plot(artefactless_signal, label="Without artefacts", alpha=0.5)
    plt.legend()
    # artefact_remover.compute_signal_error(
    #     original_signal=signal_to_remove[wind_size_tot_svd:],
    #     reduced_signal=signal_cleaned[wind_size_tot_svd:], artefactless_signal=artefactless_signal[wind_size_tot_svd:],
    #     json_path=json_path)
    # artefact_remover.compute_frequency_analysis(
    #     original_signal=signal_to_remove[wind_size_tot_svd:],
    #     reduced_signal=signal_cleaned[wind_size_tot_svd:],
    #     artefactless_signal=artefactless_signal[wind_size_tot_svd:])
    
    # artefact_remover.compute_signal_error(
    #     original_signal=signal_to_remove[wind_size_tot_svd:],
    #     reduced_signal=signal_filtered[wind_size_tot_svd:], artefactless_signal=artefactless_signal[wind_size_tot_svd:],
    #     json_path=json_path)
    # artefact_remover.compute_frequency_analysis(
    #     original_signal=signal_to_remove[wind_size_tot_svd:],
    #     reduced_signal=signal_filtered[wind_size_tot_svd:],
    #     artefactless_signal=artefactless_signal[wind_size_tot_svd:])
    plt.show()
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
