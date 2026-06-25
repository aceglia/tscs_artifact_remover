import pickle

import numpy as np

from star_emg.plot_utils import PlotSolution
from star_emg.processing_utils import Quality


class Solution:
    """
    Solution object to store the decomposition output and provide methods to plot and analyse it.
    """

    def __init__(self, data_rate):
        """
        Initialize the Solution object.
        Parameters:
        -----------
        data_rate: float
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
        self.ground_truth = None

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
            print(f"WARNING: Missing key '{key}' in decomposition output")

    def from_signal_decomposition(self, decomposition_dict: dict, initial_data_shape: tuple = None) -> None:
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
        if np.any(self.s, axis=-1):
            self.s = self.s.reshape((initial_data_shape[0], initial_data_shape[1], -1))
            self.s_reduced = self.s_reduced.reshape((initial_data_shape[0], initial_data_shape[1], -1))
        else:
            self.s = None
            self.s_reduced = None
        self.is_empty = False

    def from_notch_filter(self, out_dict: dict, initial_data_shape: tuple = None) -> None:
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

    def from_dict(self, dict: dict) -> None:
        """
        Initialize the Solution object from a dictionary.
        Parameters:
        -----------
        dict: dict
            The dictionary containing the decomposition output.

        """
        self.output = dict["output"]
        self.init_data = dict["init_data"]
        if "s" in dict:
            self.s = dict["s"]
        if "s_reduced" in dict:
            self.s_reduced = dict["s_reduced"]
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

    def plot(self, signals=True, fft=False, singular_values=False, stack_epochs=False, show_analysis=False):
        """
        Plot the decomposition output.
        Parameters:
        -----------
        signals: bool, optional
            Whether to plot the signals.
        fft: bool, optional
            Whether to plot the FFT.
        singular_values: bool, optional
            Whether to plot the singular values.
        stack_epochs: bool, optional
            Whether to stack the epochs of signals.
        show_analysis: bool, optional
            Whether to show the analysis results.

        Raises:
        -------
        RuntimeError: If the class solution do not have attribute: key.

        Returns:
        --------
        None
        """
        if show_analysis and self.analysis is None:
            raise RuntimeError("No analysis to show. Please run analyse() method before plotting analysis results.")
        plotter = PlotSolution(signals=signals, fft=fft, singular_values=singular_values, data_rate=self.data_rate)
        results = self.quality if show_analysis else None
        plotter.plot(self._get_all_decomposition_output(), stack_epochs=stack_epochs, analysis=results)

    def _convert_quality_to_dict(self, quality):
        dict_to_return = {
            "kurtosis": [quality[0][0], quality[1][0]],
            "Line Length": [quality[0][1], quality[1][1]],
            "Median frequency": [quality[0][2], quality[1][2]],
            "FFT Amplitude": [quality[0][3], quality[1][3]],
        }
        if self.ground_truth is not None:
            dict_to_return["Kurtosis"].append([quality[2][0]])
            dict_to_return["Line Length"].append([quality[2][1]])
            dict_to_return["Median frequency"].append([quality[2][2]])
            dict_to_return["FFT Amplitude"].append([quality[2][3]])

        return dict_to_return

    def analyse(self, ground_truth=None, **kwargs) -> dict:
        """
        Run quality analysis on the decomposition output using the Quality class.
        Parameters:
        -----------
        ground_truth: np.ndarray, optional
            The ground truth signal to compare with the decomposition output.
        **kwargs: dict
            Additional keyword arguments to pass to the Quality class.
            The possible keys are:
            - kw: int, optional
                The size of the window use for kurtosis analysis.
            - maxw: int, optional
                The size of the window to compute the maximum amplitude of the FFT.
            - percentile: list, optional
                The percentile to use for the quality analysis.
            - fft_freqs: list, optional
                The frequencies to use for the FFT analysis.

        Returns:
        --------
        dict: dict
            The quality analysis results.
        """
        if self.data_rate is None:
            raise RuntimeError(
                "Data rate is required for quality analysis. Please provide data_rate when initializing the Solution object."
            )
        self.analysis = Quality()
        self.ground_truth = ground_truth
        self.quality = self._convert_quality_to_dict(
            self.analysis.compute_quality(
                self.init_data, self.output, ground_truth=self.ground_truth, fs=self.data_rate, **kwargs
            )
        )
        return self.quality
