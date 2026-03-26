import numpy as np
import control as ct
from biosiglive import save

from functools import partial
from .generator_utils import Modulator
from .generator_app.generator_utils import get_from_range


class ArtifactGenerator:
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

    def _get_transfert_fct(self, num=[1], den=[0.02, 0.5, 12]):
        self.transfert_fct = ct.TransferFunction(num, den)
        return self.transfert_fct

    def _get_single_step_response(self, num=[1], den=[0.02, 0.5, 12], T=1):
        T = np.linspace(0, T, 1000)
        sys = self._get_transfert_fct(num, den)
        t, y = ct.step_response(sys, T)
        return t, y

    @staticmethod
    def shift_signal(t, y, delay):
        y_shift = np.zeros_like(y)
        idx = t >= delay
        y_shift[idx] = np.interp(t[idx] - delay, t, y)
        return y_shift

    def _get_biphasic_response_template(
        self, amplitude=1, delay_1=0.1, delay_2=0.2, num=[1], den=[0.02, 0.5, 12], T=1, factors=[1, 2, 1]
    ):
        dic = self._get_random_params(amplitude=amplitude, delay_1=delay_1, delay_2=delay_2, num=num, den=den, T=T, factors=factors)
        t, y = self._get_single_step_response(dic['num'], dic['den'], dic['T'])
        y_template = dic['factors'][0] * y - dic['factors'][1] * self.shift_signal(t, y, dic['delay_1']) + dic['factors'][2] * self.shift_signal(t, y, dic['delay_2'])
        normalized = (y_template / np.max(y_template)) - (y_template / np.max(y_template))[-1]
        return normalized * dic['amplitude']
    
    def _get_random_params(self, **kwargs):
        random_dict = kwargs.copy()
        for key, value in kwargs.items():
            if key in ['num', 'den', 'factors']:
                random_dict[key] = [self._get_random_params(**{'tmp': kwargs[key][i]})['tmp'] for i in range(len(kwargs[key]))]
            else:
                if isinstance(value, list):
                    random_dict[key] = get_from_range(value)
        return random_dict
    
    def _sample_to_frequency(self, signal, duration, fs):
        duration = self._get_random_params(duration=duration)['duration']
        target_points = np.round(fs * duration, 0).astype(int)
        return np.interp(np.linspace(0, 1, target_points), np.linspace(0, 1, len(signal)), signal)

    def generate_artifact(
        self,
        stimulation_frequency=30,
        sampling_rate=2000,
        amplitude=1,
        delay_1=0.1,
        delay_2=0.2,
        num=[1],
        den=[0.02, 0.5, 12],
        phase_inversion=False,
        artifact_duration=0.007,
        output_shape=10000,
        T=1, 
        factors=[1, 2, 1]
    ):
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
            factors=factors
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
    ):
        self.set_artifact_params(stimulation_frequency=stimulation_frequency,
            sampling_rate=sampling_rate,
            artifact_duration=artifact_duration,
            delay_1=delay_1,
            delay_2=delay_2,
            num=num,
            den=den,
            phase_inversion=phase_inversion,
            amplitude=amplitude,
            output_shape=len(signal), 
            factors=factors)
        
        init_dim = signal.ndim
        if init_dim == 1:
            signal = signal[None]
        elif init_dim == 2:
            pass
        elif init_dim > 2:
            raise ValueError('Provided signal should be a 3D array (epochs, samples) or (samples)')
        
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
            factors=factors)
        
        artifacts = modulator.apply_modulation(artifact_template)
        signal_with_artifacts = signal + artifacts
        self.signal_with_artifacts = signal_with_artifacts.reshape(init_shape)
        if init_dim == 1:
            self.signal_with_artifacts = self.signal_with_artifacts[0, :]
        return self.signal_with_artifacts

    def set_artifact_params(self, **kwargs):
        self.artifact_params = kwargs.copy()

    def save(self, file_path):
        dict_to_save = self.artifact_params.copy()
        dict_to_save['init_signal'] = self.init_signal
        dict_to_save['signal_with_artifacts'] = self.signal_with_artifacts
        save(dict_to_save, file_path, safe=False)
