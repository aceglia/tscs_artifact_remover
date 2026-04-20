import pickle

import numpy as np

from artifact_remover.plot_utils import PlotSolution
from artifact_remover.analysis import Analysis


class Solution:
    """
    Solution object to store the decomposition output and provide methods to plot and analyse it.
    """
    def __init__(self, data_rate: float=None):
        """
        Initialize the Solution object.
        Parameters:
        -----------
        data_rate: float, optional
            The sampling rate of the signal.
        """
        self.data_init = None
        self.output = None
        self.u = None
        self.s = None
        self.v = None
        self.s_reduced = None
        self.is_empty = True
        self.analysis = None
        self.data_rate = data_rate

    def _from_dict(self, dict: dict) -> None:
        """
        Set attribute of the Solution object from a dictionary.
        Parameters:
        -----------
        dict: dict
            The dictionary containing the decomposition output.

        """
        for key, value in dict.items():
            setattr(self, key, value)

    @staticmethod
    def _stack_field(data: list, key: str) -> np.ndarray:
        """
        Stack the field of the decomposition output if solved by windows.
        Parameters:
        -----------
        data: list
            The list of dictionaries containing the decomposition output.
        key: str
            The key of the field to stack.

        Raises:
        -------
        KeyError: If the key is not in the decomposition output.

        Returns:
        --------
            np.ndarray: The stacked field.

        """
        try:
            return np.stack([d[key] for d in data])
        except KeyError as e:
            raise KeyError(f"Missing key '{key}' in decomposition output") from e

    def from_signal_decomposition(self, decomposition_dict: dict, initial_data_shape: tuple=None)->None:
        """
        Initialize the Solution object from the decomposition output.

        Parameters:
        -----------
        decomposition_dict: dict
            The dictionary containing the decomposition output from the ArtifactRemover class.
        initial_data_shape: tuple, optional
            The initial shape of the data before decomposition.

        """
        decomposition_list = decomposition_dict if isinstance(decomposition_dict, list) else [decomposition_dict]
        self.output = self._stack_field(decomposition_list, "output")
        self.s = self._stack_field(decomposition_list, "s")
        self.s_reduced = self._stack_field(decomposition_list, "s_reduced")
        self.init_data = self._stack_field(decomposition_list, "data")

        self.output = self.output.reshape(initial_data_shape)
        self.init_data = self.init_data.reshape(initial_data_shape)

        self.s = self.s.reshape((initial_data_shape[0], initial_data_shape[1], -1))
        self.s_reduced = self.s_reduced.reshape((initial_data_shape[0], initial_data_shape[1], -1))
        self.is_empty = False

    def from_notch_filter(self, out_dict: dict, initial_data_shape: tuple=None)->None:
        """
        Initialize the Solution object from the notch filter output.

        Parameters:
        -----------
        out_dict: dict
            The dictionary containing the notch filter output from the ArtifactRemover class.
        initial_data_shape: tuple, optional
            The initial shape of the data before filtering.

        """
        decomposition_list = out_dict if isinstance(out_dict, list) else [out_dict]
        self.output = self._stack_field(decomposition_list, "output")
        self.s = None
        self.s_reduced = None
        self.init_data = self._stack_field(decomposition_list, "data")

        self.output = self.output.reshape(initial_data_shape)
        self.init_data = self.init_data.reshape(initial_data_shape)

        self.s = None
        self.s_reduced = None
        self.is_empty = False

    def get(self, key: str) -> np.ndarray:
        """
        Get value of the attribute of the Solution object.
        Parameters:
        -----------
        key: str
            The key of the attribute to get.

        Raises:
        -------
        RuntimeError: If the class solution do not have attribute: key.
        
        Returns:
        --------
        np.ndarray: The value of the attribute.
        """
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
        keys = ["init_data", "output", "u", "v", "s", "s_reduced"]
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
        average_batch=False,
        average_channels=False,
    ) -> dict:
        self.analysis = Analysis(
            compute_signal_error,
            compute_frequency_analysis,
            average_batch=average_batch,
            average_channels=average_channels,
            data_rate=self.data_rate,
        )
        return self.analysis.process(self.init_data, self.output, gt_signals=groundtruth_signals)
