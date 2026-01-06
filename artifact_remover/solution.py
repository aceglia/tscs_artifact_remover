import pickle

import numpy as np

from artifact_remover.plot_utils import PlotSolution


class Analysis:
    def __init__(self):
        self.plot = PlotSolution()

    def initialize(self, dict):
        for key, value in dict.items():
            setattr(self, key, value)

    def compute_signal_error(
        self,
        original_signal,
        reduced_signal,
        baseline_idx=None,
        signal_idx=None,
        artefactless_signal=None,
        stim_time=0,
        json_path=None,
    ):
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
            with open(json_path, "r") as f:
                config = json.load(f)
            baseline_idx, signal_idx, stim_time_file = (
                [config[0], config[1]],
                [config[2], config[3]],
                config[4],
            )
        else:
            path = json_path if json_path is not None else "_window_times.json"
            Window().get_signal_idxs(
                reduced_signal_rectified,
                baseline_idx,
                signal_idx,
                json_path=path,
                original_signal=reduced_signal,
            )
            if baseline_idx is None or signal_idx is None:
                with open(path, "r") as f:
                    config = json.load(f)
                baseline_idx, signal_idx, stim_time_file = (
                    [config[0], config[1]],
                    [config[2], config[3]],
                    config[4],
                )

        # ratio = np.mean(emg_envelope[int(signal_idx[0]):int(signal_idx[1])]) / np.mean(
        # emg_envelope[int(baseline_idx[0]):int(baseline_idx[1])])
        # original_ratio = np.mean(emg_envelope_original[int(signal_idx[0]):int(signal_idx[1])]) / np.mean(
        #     emg_envelope_original[int(baseline_idx[0]):int(baseline_idx[1])])
        signal_reduced = reduced_signal_rectified[int(signal_idx[0]) : int(signal_idx[1])]
        signal_original = original_signal_rectified[int(signal_idx[0]) : int(signal_idx[1])]
        signal_artifactfree = (
            artifact_free_rectified[int(signal_idx[0]) : int(signal_idx[1])]
            if artifact_free_rectified is not None
            else None
        )
        baseline_reduced = reduced_signal_rectified[int(baseline_idx[0]) : int(baseline_idx[1])]
        baseline_original = original_signal_rectified[int(baseline_idx[0]) : int(baseline_idx[1])]
        baseline_artifactfree = (
            artifact_free_rectified[int(baseline_idx[0]) : int(baseline_idx[1])]
            if artifact_free_rectified is not None
            else None
        )
        shape_baseline = baseline_reduced.shape[0] // 4
        shape_signal = signal_reduced.shape[0] // 4
        shape_to_take = min(shape_baseline, shape_signal)
        ratio = np.mean(-np.sort(-signal_reduced)[:shape_to_take]) / np.mean(
            -np.sort(-baseline_reduced)[:shape_to_take]
        )
        original_ratio = np.mean(-np.sort(-signal_original)[:shape_to_take]) / np.mean(
            -np.sort(-baseline_original)[:shape_to_take]
        )
        artifactfree_ratio = None
        if artefactless_signal is not None:
            artifactfree_ratio = np.mean(-np.sort(-signal_artifactfree)[:shape_to_take]) / np.mean(
                -np.sort(-baseline_artifactfree)[:shape_to_take]
            )
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
            initial_pearson, final_lag, peaks_error = compute_signal_comparison(
                reduced_signal, original_signal, stim_time
            )
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
            energy = amp**2
            energy_cumsum = np.cumsum(energy)
            mdfs.append(freq[np.where(energy_cumsum > np.max(energy_cumsum) / 2)[0][0]])
            text += f"{data_name[i]}: MDF: {mdfs[-1]:.2f} Hz\n"
            self.mdfs.append(mdfs[-1])
        # self.mdfs = mdfs
        print(self.hankel_matrix.shape)
        print(text)


class Solution:
    def __init__(self):
        self.output = None
        self.input = None
        self.intermediate = None
        self.data_init = None
        self.unfiltered_signal = None
        self.signal_reduced = None
        self.u = None
        self.s = None
        self.v = None
        self.s_reduced = None
        self.is_empty = True

        self.analysis = Analysis()
        self.plot = PlotSolution()

    def _from_dict(self, dict):
        for key, value in dict.items():
            setattr(self, key, value)

    @staticmethod
    def _stack_field(data, key):
        try:
            return np.stack([d[key] for d in data])
        except KeyError as e:
            raise KeyError(f"Missing key '{key}' in decomposition output") from e

    def from_signal_decomposition(self, decomposition_dict, initial_data_shape=None):
        decomposition_list = decomposition_dict if isinstance(decomposition_dict, list) else [decomposition_dict]
        self.output = self._stack_field(decomposition_list, "output")
        self.s = self._stack_field(decomposition_list, "s")
        self.s_reduced = self._stack_field(decomposition_list, "s_reduced")
        self.unfiltered_signal = self._stack_field(decomposition_list, "unfiltered_signal")
        self.init_data = self._stack_field(decomposition_list, "data")

        self.output = self.output.reshape(initial_data_shape)
        self.unfiltered_signal = self.unfiltered_signal.reshape(initial_data_shape)
        self.init_data = self.init_data.reshape(initial_data_shape)

        self.s = self.s.reshape((initial_data_shape[0], initial_data_shape[1], -1))
        self.s_reduced = self.s_reduced.reshape((initial_data_shape[0], initial_data_shape[1], -1))

        # self.analysis.initialize(decomposition_dict)
        # self.plot_sol.initialize(decomposition_dict)
        self.is_empty = False

    def get(self, key):
        if not isinstance(key, list):
            key = [key]
        to_return = []
        for k in key:
            if hasattr(self, key):
                to_return.append(getattr(self, key))
            else:
                raise RuntimeError("Class solution do not have attribute:" + key)
        return to_return

    def _get_to_save(self):
        pass

    def save(self, path):
        dict_to_save = self._get_to_save()
        with open(path, "wb") as f:
            pickle.dump(dict_to_save, f)

    def plot(self, signals=True, fft=False, singular_values=False, stack_batch=False):
        pass

    def analyse(self):
        pass
