import pickle

import numpy as np

from artifact_remover.plot_utils import PlotSolution
from artifact_remover.analysis import Analysis


class Solution:
    def __init__(self, data_rate=None):
        self.data_init = None
        self.unfiltered_signal = None
        self.output = None
        self.u = None
        self.s = None
        self.v = None
        self.s_reduced = None
        self.is_empty = True
        self.analysis = None
        self.data_rate = data_rate

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
        self.unfiltered_output = self._stack_field(decomposition_list, "unfiltered_signal")
        self.init_data = self._stack_field(decomposition_list, "data")

        self.output = self.output.reshape(initial_data_shape)
        self.unfiltered_output = self.unfiltered_output.reshape(initial_data_shape)
        self.init_data = self.init_data.reshape(initial_data_shape)

        self.s = self.s.reshape((initial_data_shape[0], initial_data_shape[1], -1))
        self.s_reduced = self.s_reduced.reshape((initial_data_shape[0], initial_data_shape[1], -1))
        self.is_empty = False

    def from_notch_filter(self, out_dict, initial_data_shape=None):
        decomposition_list = out_dict if isinstance(out_dict, list) else [out_dict]
        self.output = self._stack_field(decomposition_list, "output")
        self.s = None
        self.s_reduced = None
        self.unfiltered_output = None
        self.init_data = self._stack_field(decomposition_list, "data")

        self.output = self.output.reshape(initial_data_shape)
        self.unfiltered_output = None
        self.init_data = self.init_data.reshape(initial_data_shape)

        self.s = None
        self.s_reduced = None
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

    def update(self, key, value):
        if not isinstance(key, list):
            key = [key]
            value = [value]
        for k, item in zip(key, value):
            setattr(self, k, item)
        
    def add_data(self, key, value, axis=-1):
        if not isinstance(key, list):
            key = [key]
            value = [value]
        if not isinstance(axis, list):
            axis = [axis] * len(key)
        for k, item, ax in zip(key, value, axis):
            setattr(self, k, np.concatenate((getattr(self, k), item), axis=ax))


    def _get_all_decomposition_output(self):
        keys = ["init_data", "output", "unfiltered_output", "u", "v", "s", "s_reduced"]
        return {key: attrib for key, attrib in self.__dict__.items() if key in keys}

    def save(self, path):
        dict_to_save = self._get_all_decomposition_output()
        with open(path, "wb") as f:
            pickle.dump(dict_to_save, f)

    def plot(self, signals=True, fft=False, singular_values=False, stack_batch=False, show_analysis=False):
        if show_analysis and self.analysis is None:
            raise RuntimeError("No analysis to show. Please run analyse() method before plotting analysis results.")
        plotter = PlotSolution(signals=signals, fft=fft, singular_values=singular_values, data_rate=self.data_rate)
        results = self.analysis.get_results() if show_analysis else None
        plotter.plot(self._get_all_decomposition_output(), stack_batch=stack_batch, analysis=results)

    def analyse(
        self,
        compute_signal_error=False,
        compute_frequency_analysis=True,
        groundtruth_signals=None,
        output_filtered=True,
        average_batch=False,
        average_channels=False,
    ) -> dict:
        self.analysis = Analysis(
            compute_signal_error,
            compute_frequency_analysis,
            average_batch=average_batch,
            average_channels=average_channels,
            data_rate=self.data_rate
        )
        output = self.output if output_filtered else self.unfiltered_output
        return self.analysis.process(self.init_data, output, gt_signals=groundtruth_signals)
