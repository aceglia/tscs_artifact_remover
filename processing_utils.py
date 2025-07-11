import numpy as np
import scipy
import scipy.signal as signal
from biosiglive import OfflineProcessing


def _butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def _bandpass_filter(data, cutoff, fs, order=4):
    b, a = _butter_bandpass(cutoff[0], cutoff[1], fs, order)
    y = signal.filtfilt(b, a, data)
    return y


def _butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def _butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = _butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def _butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def _butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = _butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def filter_data(data, cutoff=450.0, order=5, fs=2000.0, filter_type='low'):
    if filter_type == 'low':
        filter_function = _butter_lowpass_filter
    elif filter_type == 'band':
        filter_function = _bandpass_filter
    elif filter_type == 'high':
        filter_function = _butter_highpass_filter
    else:
        raise ValueError('Invalid filter type')
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        for k in range(data.shape[-1]):
            filtered_data[i, :, k] = filter_function(data[i, :, k], cutoff, fs, order=order)
    return filtered_data


def compute_envelope(data, fs=2000.0):
    proc_emg = OfflineProcessing(fs)
    emg_envelope = proc_emg.process_emg(data[None, :],
                                        band_pass_filter=True,
                                        low_pass_filter=True,
                                        moving_average=False,
                                        centering=True,
                                        absolute_value=True)[0, :]
    return emg_envelope


def compute_signal_comparison(data, ref_data, n_frame_stim=6000):
    correlation = signal.correlate(data[:], ref_data[:data.shape[0]])
    lag = signal.correlation_lags(data.shape[0], ref_data[:data.shape[0]].shape[0])
    correlation /= np.max(correlation)
    final_lag = lag[np.argmax(correlation)]
    pearson = \
        scipy.stats.pearsonr(data[100:], ref_data[100:data.shape[0]])[
            0]  # signal_svd = self._butter_lowpass_filter(signal_svd, 450.0, 2000, order=4)
    peak_to_peak_data = scipy.signal.find_peaks(np.abs(ref_data[
                                                       int(n_frame_stim + 0.1 * 2000):int(
                                                           n_frame_stim + 0.4 * 2000)]), height=0.08)
    amplitude_ref = np.sum(peak_to_peak_data[1]["peak_heights"])
    peak_to_peak_data = scipy.signal.find_peaks(
        np.abs(data[int(n_frame_stim + 0.1 * 2000):int(n_frame_stim + 0.4 * 2000)]), height=0.08)
    peaks = [p for p in peak_to_peak_data[1]["peak_heights"] if p < 0.4]
    amplitude_proc = np.sum(peak_to_peak_data[1]["peak_heights"])
    peaks_error = amplitude_ref - amplitude_proc
    return pearson, final_lag, peaks_error