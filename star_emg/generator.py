from typing import Union

import numpy as np
import control as ct
from biosiglive import save

from functools import partial
from .generator_utils import Modulator
from .generator_app.generator_utils import get_from_range


class ArtifactGenerator:
    """
    A class for generating artifacts in signals based on a specified transfer function and modulation. The ArtifactGenerator class provides methods to create artifact templates, apply them to input signals, and save the generated signals with artifacts. The class allows for customization of artifact parameters such as stimulation frequency, sampling rate, amplitude, delays, and modulation settings.
    """

    def __init__(self):
        self.fs = 2000
        self.freq = 30
        self.amplitude = 1
        self.outpup_sampling = 2000
        self.real_duration = 0.007
        self.transfert_fct = None
        self.init_signal = None
        self.signal_with_artifacts = None
        self.artifact_params = {}

    def _get_transfert_fct(self, num: list = [1], den: list = [0.02, 0.5, 12]) -> ct.TransferFunction:
        """
        Gets the transfer function based on the provided numerator and denominator coefficients. The transfer function is created using the control library and is stored as an attribute of the ArtifactGenerator class for later use in generating artifacts.
        """
        self.transfert_fct = ct.TransferFunction(num, den)
        return self.transfert_fct

    def _get_single_step_response(self, num: list = [1], den: list = [0.02, 0.5, 12], T: float = 1) -> tuple:
        """
        Gets the step response of the system defined by the transfer function. The step response is calculated using the control library and is returned as a tuple containing the time vector and the response values. This method is used to create the artifact template based on the specified transfer function.
        """
        T = np.linspace(0, T, 1000)
        sys = self._get_transfert_fct(num, den)
        t, y = ct.step_response(sys, T)
        return t, y

    @staticmethod
    def shift_signal(t: np.ndarray, y: np.ndarray, delay: float) -> np.ndarray:
        """
        Shifts the input signal by a specified delay. The signal is shifted by interpolating the values of the signal at the time points corresponding to the delay. This method is used to create the biphasic response template by shifting the step response of the system.
        Parameters:
        -----------
        t: np.ndarray
            The time vector corresponding to the signal values.
        y: np.ndarray
            The signal values to be shifted.
        delay: float
            The amount of time by which to shift the signal.

        Returns:
        --------
        np.ndarray
            The shifted signal after applying the specified delay.
        """
        y_shift = np.zeros_like(y)
        idx = t >= delay
        y_shift[idx] = np.interp(t[idx] - delay, t, y)
        return y_shift

    def _get_biphasic_response_template(
        self,
        amplitude: float = 1,
        delay_1: Union[float, list] = 0.1,
        delay_2: Union[float, list] = 0.2,
        num: list = [1],
        den: list = [0.02, 0.5, 12],
        T: float = 1,
        factors: list = [1, 2, 1],
    ) -> np.ndarray:
        """
        Gets the biphasic response template based on the specified parameters. The template is created by combining the step response of the system with shifted versions of the response to create a biphasic shape. The resulting template is normalized and scaled by the specified amplitude before being returned as a numpy array.

        """
        dic = self._get_random_params(
            amplitude=amplitude, delay_1=delay_1, delay_2=delay_2, num=num, den=den, T=T, factors=factors
        )
        t, y = self._get_single_step_response(dic["num"], dic["den"], dic["T"])
        y_template = (
            dic["factors"][0] * y
            - dic["factors"][1] * self.shift_signal(t, y, dic["delay_1"])
            + dic["factors"][2] * self.shift_signal(t, y, dic["delay_2"])
        )
        normalized = (y_template / np.max(y_template)) - (y_template / np.max(y_template))[-1]
        return normalized * dic["amplitude"]

    def _get_random_params(self, **kwargs) -> dict:
        """
        Gets the random parameters based on the range provided in the keyword arguments. The parameters that can be randomized include "amplitude", "delay_1", "delay_2", "num", "den", "T", and "factors". The random values are generated using the "get_from_range" function for parameters that are specified as lists, and the resulting random parameters are returned as a dictionary.
        """
        random_dict = kwargs.copy()
        for key, value in kwargs.items():
            if key in ["num", "den", "factors"]:
                random_dict[key] = [
                    self._get_random_params(**{"tmp": kwargs[key][i]})["tmp"] for i in range(len(kwargs[key]))
                ]
            else:
                if isinstance(value, list):
                    random_dict[key] = get_from_range(value)
        return random_dict

    def _sample_to_frequency(self, signal, duration, fs):
        duration = self._get_random_params(duration=duration)["duration"]
        target_points = np.round(fs * duration, 0).astype(int)
        return np.interp(np.linspace(0, 1, target_points), np.linspace(0, 1, len(signal)), signal)

    def generate_artifact(
        self,
        stimulation_frequency: float = 30,
        sampling_rate: float = 2000,
        amplitude: float = 1,
        delay_1: Union[float, list] = 0.1,
        delay_2: Union[float, list] = 0.2,
        num: list = [1],
        den: list = [0.02, 0.5, 12],
        phase_inversion: bool = False,
        artifact_duration: Union[float, list] = 0.007,
        output_shape: int = 10000,
        T: float = 1,
        factors: list = [1, 2, 1],
    ) -> np.ndarray:
        """
        Generates a artifact signal based on the specified parameters. The artifact signal is created by combining the biphasic response template with a modulated version of the template. The resulting signal is returned as a numpy array.
        Parameters:
        -----------
        stimulation_frequency: float
            The frequency of the stimulation in Hz.
        sampling_rate: float
            The sampling rate of the signal in Hz.
        amplitude: float
            The amplitude of the artifact signal.
        delay_1: Union[float, list]
            The delay for the first shifted response in seconds or a list of min max range to randomly select a value in between.
        delay_2: Union[float, list]
            The delay for the second shifted response in seconds or a list of min max range to randomly select a value in between.
        num: list
            the numerator coefficients for the transfer function or a list of min max range to randomly select a value in between.
        den: list
            the denominator coefficients for the transfer function or a list of min max range to randomly select a value in between.
        phase_inversion: bool
            Whether to apply phase inversion to the artifact signal.
        artifact_duration: Union[float, list]
            The duration of the artifact template in seconds or a list of possible values to randomize from.
        output_shape: int
            The length of the output artifact signal in samples.
        T: float
            The duration of the step response in seconds.
        factors: list
            The factors to apply to the original, first shifted, and second shifted responses in the biphasic template or a list of possible values to randomize from.

        Returns:
        --------
        np.ndarray
            The generated artifact signal as a numpy array.
        """
        zeros_signal = np.zeros((output_shape))
        stim_spacing = (1 / stimulation_frequency) * sampling_rate
        idx_stim = np.round(np.arange(0, zeros_signal.shape[-1], stim_spacing), 0).astype(int)
        template_partial = partial(
            self._get_biphasic_response_template,
            amplitude=amplitude,
            delay_1=delay_1,
            delay_2=delay_2,
            num=num,
            den=den,
            T=T,
            factors=factors,
        )
        template_sampled_partial = partial(self._sample_to_frequency, duration=artifact_duration, fs=sampling_rate)
        for stim in range(len(idx_stim)):
            factor = (-1) ** stim if phase_inversion else 1
            artifact_tmp = factor * template_sampled_partial(signal=template_partial())
            if idx_stim[stim] + len(artifact_tmp) > len(zeros_signal):
                artifact_tmp = artifact_tmp[: len(zeros_signal) - idx_stim[stim]]
            zeros_signal[idx_stim[stim] : idx_stim[stim] + len(artifact_tmp)] = artifact_tmp
        self.template_sampled = zeros_signal
        return self.template_sampled

    def apply_artifact_to_signal(
        self,
        signal,
        artifact_duration=0.007,
        stimulation_frequency=30,
        sampling_rate=2000,
        delay_1=0.1,
        delay_2=0.2,
        num=[9],
        den=[0.2, 0.5, 5],
        amplitude=1,
        factors=[1, 2, 1],
        phase_inversion=False,
        modulator: Modulator = None,
    ) -> np.ndarray:
        """
        Applies the generated artifact to the input signal based on the specified parameters. The method generates the artifact template and modulates it using the provided Modulator instance. The modulated artifact is then added to the input signal, and the resulting signal with artifacts is returned as a numpy array.
        """
        self.set_artifact_params(
            stimulation_frequency=stimulation_frequency,
            sampling_rate=sampling_rate,
            artifact_duration=artifact_duration,
            delay_1=delay_1,
            delay_2=delay_2,
            num=num,
            den=den,
            phase_inversion=phase_inversion,
            amplitude=amplitude,
            output_shape=len(signal),
            factors=factors,
        )

        init_dim = signal.ndim
        if init_dim == 1:
            signal = signal[None]
        elif init_dim == 2:
            pass
        elif init_dim > 2:
            raise ValueError("Provided signal should be a 3D array (epochs, samples) or (samples)")

        self.init_signal = signal.copy()
        self.signal_with_artifacts = np.zeros_like(signal)
        init_shape = signal.shape
        signal = signal.reshape((signal.shape[0] * signal.shape[1]))
        artifact_template = self.generate_artifact(
            stimulation_frequency=stimulation_frequency,
            sampling_rate=sampling_rate,
            artifact_duration=artifact_duration,
            delay_1=delay_1,
            delay_2=delay_2,
            num=num,
            den=den,
            phase_inversion=phase_inversion,
            amplitude=amplitude,
            output_shape=len(signal),
            factors=factors,
        )

        artifacts = modulator.apply_modulation(artifact_template)
        signal_with_artifacts = np.asanyarray(signal + artifacts).astype(np.float64)
        self.signal_with_artifacts = signal_with_artifacts.reshape(init_shape)
        if init_dim == 1:
            self.signal_with_artifacts = self.signal_with_artifacts[0, :]
        return self.signal_with_artifacts

    def set_artifact_params(self, **kwargs):
        """
        Sets the artifact parameters based on the provided keyword arguments. The parameters are stored in a dictionary attribute of the ArtifactGenerator class for later use when saving the generated signals with artifacts.
        """
        self.artifact_params = kwargs.copy()

    def save(self, file_path: str):
        """
        Saves the generated signal with artifacts and the corresponding parameters to a file with the extension '.bio'. The method creates a dictionary containing the artifact parameters, the initial signal, and the signal with artifacts, and then saves this dictionary to the specified file path using the "save" function from the biosiglive library.
        """
        dict_to_save = self.artifact_params.copy()
        dict_to_save["init_signal"] = self.init_signal
        dict_to_save["signal_with_artifacts"] = self.signal_with_artifacts
        save(dict_to_save, file_path, safe=False)
