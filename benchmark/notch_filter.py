import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
from scipy.io import loadmat

from artifact_remover.automatic_remover import ArtefactRemover
from processing_utils import filter_data


def extract_data(file_path):
    mat_file = loadmat(file_path)
    dic_tot = {}
    for key in mat_file.keys():
        if "Ch" in key:
            dic_tmp = {}
            items = list(mat_file[key][0][0].dtype.fields.keys())
            for i in range(len(items)):
                dic_tmp[items[i]] = mat_file[key][0][0][i]
            dic_tot[key] = dic_tmp
    return dic_tot

if __name__ == '__main__':
    file_path = r'd:\Downloads\T1_008_arm_sa_002.mat'
    data_dict = extract_data(file_path)
    keys = list(data_dict.keys())[1:]

    artefact_remover = ArtefactRemover(data=data_dict[keys[-1]]["values"][None, ...],
                                        plot_figure=False)
    signal_to_remove = artefact_remover.init_data[0, :, 0]
    cutoff = 450.0
    fs = 2000
    filter_type = 'low'
    order = 2
    signal_to_remove = filter_data(signal_to_remove[None, :, None], cutoff, order, fs, filter_type)
    original_fft = np.fft.fft(signal_to_remove[0, :, 0])
    # get peaks in fft usng moving windows and find peak function in scipy
    freq = np.fft.fftfreq(len(original_fft), 1 / 2000)
    fft = np.abs(original_fft[freq > 0])
    freq = freq[freq > 0]
    wind_size = 4000
    total_wind = int(len(fft) / wind_size)
    total_idx = 0
    peaks = []
    for i in range(total_wind):
        mean = np.mean(fft[i*wind_size:(i+1)*wind_size])
        sd = np.std(fft[i*wind_size:(i+1)*wind_size])
        # find peaks in fft 
        peaks_tmp, _ = scipy.signal.find_peaks(fft[i*wind_size:(i+1)*wind_size], height=mean + 8*sd)
        peaks.extend([p + i*wind_size for p in peaks_tmp])
    

    # plt.plot(freq, fft)
    # plt.scatter(freq[peaks], fft[peaks], color='r')
    # remove peaks from signal and plot
    # Notch filter parameters
    # put all peaks to zero and pass the signal back in the temporal
    # domain to remove the notch filter


    filtered_signal = signal_to_remove[0, :, 0].copy()
    for p in peaks:
        f_noise = freq[p]
        w0 = f_noise #/ (fs / 2)  # Normalized frequency to remove (f_noise / Nyquist frequency)
        Q = 150  # Quality factor (determines bandwidth)

        # Design the notch filter
        b, a = scipy.signal.iirnotch(w0, Q, fs=fs)

        # Apply the filter to the signal
        filtered_signal = scipy.signal.filtfilt(b, a, filtered_signal)
    fft_filtered = np.fft.fft(filtered_signal)
    freq_filtered = np.fft.fftfreq(filtered_signal.shape[0], 1/2000)
    plt.figure('fft')
    plt.plot(freq_filtered, np.abs(fft_filtered))
    
    plt.figure()
    plt.plot(signal_to_remove[0, :, 0])
    plt.plot(filtered_signal)

    plt.show()

    plt.figure('fft')
    plt.plot(np.abs(original_fft))
    plt.show()




    original_fft = np.fft.fft(signal_to_remove[n_batch-batch_tot * window_size:n_batch*window_size])
    cleaned_fft = np.fft.fft(total_signal[n_batch-batch_tot * window_size:n_batch*window_size])
    x_fft = np.fft.fftfreq(signal_to_remove[n_batch-batch_tot * window_size:n_batch*window_size].shape[0], 1/2000)
    plt.figure('frequency')
    plt.plot(x_fft, np.abs(original_fft), label='Original FFT')
    plt.plot(x_fft, np.abs(cleaned_fft), label='Cleaned FFT')
    plt.legend()
    plt.figure('signal')
    plt.plot(signal_to_remove[n_batch-batch_tot * window_size:n_batch*window_size], label='Original signal')
    plt.plot(total_signal[n_batch-batch_tot * window_size:n_batch*window_size], label='Cleaned signal')
    plt.legend()
    plt.show()