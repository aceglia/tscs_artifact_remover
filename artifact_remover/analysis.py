import numpy as np
import scipy
from artifact_remover.processing_utils import median_frequency


class Analysis:
    def __init__(
        self,
        compute_signal_error=True,
        compute_frequency_analysis=True,
        average_batch=False,
        average_channels=False,
        data_rate=2000,
    ):
        self.compute_frequency_analysis = compute_frequency_analysis
        self.compute_signal_error = compute_signal_error
        self.average_batch = average_batch
        self.average_channels = average_channels
        self.mdfs = None
        self.mdf_init_data = None
        self.mdf_processed_data = None
        self.correlation = None
        self.data_rate = data_rate
        self.lag = None
        self.rmse = None

    def _average_over_defined_axes(self, data):
        axes = []
        if self.average_batch:
            axes.append(0)
        if self.average_channels:
            axes.append(1)
        if len(axes) == 0:
            return data
        for i in range(len(data)):
            if data[i] is not None:
                for axis in sorted(axes, reverse=True):
                    data[i] = np.mean(data[i], axis=axis, keepdims=True)
        return data

    def process(self, initial_signal, processed_signal, gt_signals=None):
        if self.compute_signal_error and gt_signals is None:
            raise RuntimeError("Ground truth signals not provided. Impossible to compute correlation.")
        if self.compute_signal_error:
            self.correlation, self.lag, self.rmse = self._compare_signal(gt_signals, processed_signal)
        if self.compute_frequency_analysis:
            self.mdfs = self._perform_frequency_analysis(initial_signal, processed_signal, gt_signals)
            self.mdf_init_data = self.mdfs[0]
            self.mdf_processed_data = self.mdfs[1]
        results = self._average_over_defined_axes(
            [self.mdf_init_data, self.mdf_processed_data, self.correlation, self.lag, self.rmse]
        )
        dict_results = {
            "mdf_init": results[0],
            "mdf_processed": results[1],
            "correlation": results[2],
            "lag": results[3],
            "rmse": results[4],
        }
        self._dict_results = dict_results
        return dict_results

    def get_results(self):
        if self._dict_results is None:
            raise RuntimeError("No results available. Please run process() method first.")
        return self._dict_results

    def _flatten(self, data):
        original_shape = data.shape
        flattened_data = data.reshape(-1, data.shape[-1])
        return flattened_data, original_shape

    def _unflatten(self, flattened_data, original_shape):
        return flattened_data.reshape(original_shape)

    def _compare_signal(self, ground_truth_signal, signal_to_compare):
        assert (
            ground_truth_signal.shape == signal_to_compare.shape
        ), "Ground truth and signal to compare must have the same shape."

        ground_truth_signal, gt_shape = self._flatten(ground_truth_signal)
        signal_to_compare, _ = self._flatten(signal_to_compare)
        correlation = np.vstack(
            [np.correlate(gt, sc, mode="same") for gt, sc in zip(ground_truth_signal, signal_to_compare)]
        )
        lag = np.argmax(correlation, axis=-1) - (ground_truth_signal.shape[-1] / 2)
        pearson = np.vstack(
            [scipy.stats.pearsonr(gt, sc).statistic for gt, sc in zip(ground_truth_signal, signal_to_compare)]
        )
        rmse = np.sqrt(np.mean((ground_truth_signal - signal_to_compare) ** 2, axis=-1))
        return pearson.reshape(gt_shape[:-1]), lag.reshape(gt_shape[:-1]), rmse.reshape(gt_shape[:-1])

    def _perform_frequency_analysis(self, original_signal, reduced_signal, ground_truth_signal=None):
        data_to_compute = [original_signal, reduced_signal]
        if ground_truth_signal is not None:
            data_to_compute.extend([ground_truth_signal])
        mdfs = []
        for i in range(len(data_to_compute)):
            mdf = median_frequency(data_to_compute[i], fs=self.data_rate)
            mdfs.append(mdf)
        return mdfs
