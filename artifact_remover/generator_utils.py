import numpy as np


class Modulator:
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

    def _apply_linear(self, signal):
        linear_factor = np.linspace(self.min, self.max, len(signal))
        return signal * linear_factor

    def _apply_steps(self, signal):
        if self.step_inc is not None:
            steps = np.arange(self.min, self.max, self.step_inc)
            n_frame_per_step = len(signal) // len(steps)
        elif self.step_length:
            n_frame_per_step = self.step_length
            steps = np.linspace(self.min, self.max, len(signal)//n_frame_per_step)
        elif self.step_length is not None and self.step_inc is not None:
            raise ValueError("Both step_length and step_inc cannot be set at the same time.")
        n_steps = len(steps)
        remaining = len(signal) % n_frame_per_step
        modulated_signal = np.zeros_like(signal)
        for i in range(n_steps):
            modulated_signal[i * n_frame_per_step : (i + 1) * n_frame_per_step] = (
                signal[i * n_frame_per_step : (i + 1) * n_frame_per_step] * steps[i]
            )
        if remaining !=0:
            modulated_signal[-remaining:] = signal[-remaining:] * steps[-1]
        return modulated_signal

    def _apply_constant(self, signal):
        return signal * self.constant_factor

    def apply_modulation(self, signal, **kwargs):
        self.set_modulation_config(**kwargs)
        self.init_signal = signal
        self.output = self.fct_mapping[self.modulation_type](signal)
        return self.output

    def set_modulation_config(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_modulation_config(self):
        config = {}
        for key, value in self.__dict__.items():
            if key not in ["init_signal", "output", "fct_mapping"]:
                config[key] = value
        return config
