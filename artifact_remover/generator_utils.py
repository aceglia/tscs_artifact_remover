import numpy as np


class Modulator:
    """
    Class to apply modulation to a signal. The modulation can be applied in three ways: "steps", "linear", and "constant". The modulation type can be set using the "modulation_type" attribute, and the parameters for each modulation type can be set using the "set_modulation_config" method. The modulated signal can be obtained by calling the "apply_modulation" method with the input signal.
    """

    def __init__(self, **kwargs):
        self.init_signal = None
        self.output = None
        self.modulation_type = "steps"
        self.min = 0
        self.max = 1
        self.step_inc = None
        self.step_length = None
        self.constant_factor = 1
        self.fct_mapping = {"steps": self._apply_steps, "linear": self._apply_linear, "constant": self._apply_constant}
        self.set_modulation_config(**kwargs)

    def _apply_linear(self, signal: np.ndarray) -> np.ndarray:
        """
        Applies a linear modulation to the input signal. The modulation is applied by multiplying the input signal with a linear factor that increases from "min" to "max" across the length of the signal.

        Parameters:
        -----------
        signal: np.ndarray
            The input signal to be modulated.

        Returns:
        --------
        np.ndarray
            The modulated signal after applying linear modulation.

        """
        linear_factor = np.linspace(self.min, self.max, len(signal))
        return signal * linear_factor

    def _apply_steps(self, signal: np.ndarray) -> np.ndarray:
        """
        Applies a step modulation to the input signal. The modulation is applied by multiplying the input signal with a step factor that changes at regular intervals defined by "step_inc" or "step_length". The step factor takes values from "min" to "max" in increments of "step_inc" or in equal steps defined by "step_length".

        Parameters:
        -----------
        signal: np.ndarray
            The input signal to be modulated.

        Returns:
        --------
        np.ndarray
            The modulated signal after applying step modulation.
        """
        if self.step_inc is not None:
            steps = np.arange(self.min, self.max, self.step_inc)
            n_frame_per_step = len(signal) // len(steps)
        elif self.step_length:
            n_frame_per_step = self.step_length
            steps = np.linspace(self.min, self.max, len(signal) // n_frame_per_step)
        elif self.step_length is not None and self.step_inc is not None:
            raise ValueError("Both step_length and step_inc cannot be set at the same time.")
        n_steps = len(steps)
        remaining = len(signal) % n_frame_per_step
        modulated_signal = np.zeros_like(signal)
        for i in range(n_steps):
            modulated_signal[i * n_frame_per_step : (i + 1) * n_frame_per_step] = (
                signal[i * n_frame_per_step : (i + 1) * n_frame_per_step] * steps[i]
            )
        if remaining != 0:
            modulated_signal[-remaining:] = signal[-remaining:] * steps[-1]
        return modulated_signal

    def _apply_constant(self, signal: np.ndarray) -> np.ndarray:
        """
        Applies a constant modulation to the input signal. The modulation is applied by multiplying the input signal with a constant factor defined by "constant_factor".

        Parameters:
        -----------
        signal: np.ndarray
            The input signal to be modulated.
        Returns:
        --------
        np.ndarray
            The modulated signal after applying constant modulation.
        """
        return signal * self.constant_factor

    def apply_modulation(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        """
        Applies the specified modulation to the input signal. The modulation type and its parameters can be set using the "set_modulation_config" method or passed as keyword arguments to this method. The modulated signal is returned as output.

        Parameters:
        -----------
        signal: np.ndarray
            The input signal to be modulated.
        **kwargs:
            Additional keyword arguments to set modulation parameters. These can include "modulation_type", "min", "max", "step_inc", "step_length", and "constant_factor".

        Returns:
        --------
        np.ndarray
            The modulated signal after applying the specified modulation.
        """

        self.set_modulation_config(**kwargs)
        self.init_signal = signal
        self.output = self.fct_mapping[self.modulation_type](signal)
        return self.output

    def set_modulation_config(self, **kwargs):
        """
        Sets the modulation configuration parameters based on the provided keyword arguments. The parameters that can be set include "modulation_type", "min", "max", "step_inc", "step_length", and "constant_factor". These parameters will be used when applying modulation to the input signal.
        Parameters:
        -----------
        **kwargs:
            Additional keyword arguments to set modulation parameters. These can include "modulation_type", "min", "max", "step_inc", "step_length", and "constant_factor".
        """

        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_modulation_config(self) -> dict:
        """
        Gets the current modulation configuration parameters as a dictionary. The returned dictionary includes all the attributes of the Modulator class except for "init_signal", "output", and "fct_mapping". This method can be used to retrieve the current modulation settings for reference or debugging purposes.
        Returns:
        --------
        dict
            A dictionary containing the current modulation configuration parameters, excluding "init_signal", "output", and "fct_mapping".
        """
        config = {}
        for key, value in self.__dict__.items():
            if key not in ["init_signal", "output", "fct_mapping"]:
                config[key] = value
        return config
