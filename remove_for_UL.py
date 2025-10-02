import scipy
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

from automatic_remover import ArtefactRemover


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
    # Load the mat file
    file_path = r'd:\Downloads\T1_008_arm_sa_002.mat'
    data_dict = extract_data(file_path)
    keys = list(data_dict.keys())[1:]

    artefact_remover = ArtefactRemover(data=data_dict[keys[-3]]["values"][None, ...],
                                        plot_figure=False)
    signal_to_remove = artefact_remover.init_data[0, :, 0]
    window_size = 10000
    n_batch = signal_to_remove.shape[0] // window_size
    total_signal = np.zeros_like(signal_to_remove)
    batch_tot = 50
    # import time
    for i in range(n_batch - batch_tot, n_batch):
        # tic = time.time()
        start = i * window_size
        end = (i + 1) * window_size
        signal_tmp = signal_to_remove[start:end]
        signal_cleand_tmp = artefact_remover.signal_decomposition(signal_tmp,
                                                hankel_size=1000, artefactless_signal=None,
                                                threshold=None, randomized=False)
        # print(f"Time elapsed for batch {i}: {time.time() - tic}")
        total_signal[start:end] = signal_cleand_tmp
    

    original_fft = np.fft.fft(signal_to_remove)
    # get peaks in fft usng moving windows and find peak function in scipy
    freq = np.fft.fftfreq(len(original_fft), 1 / 2000)
    # plt.plot(freq, np.abs(original_fft))
    # plt.show()
    fft = np.abs(original_fft[freq > 0])
    freq = freq[freq > 0]
    wind_size = 500
    total_wind = int(len(fft) / wind_size)
    total_idx = 0
    peaks = []
    for i in range(total_wind):
        mean = np.mean(fft[i*wind_size:(i+1)*wind_size])
        sd = np.std(fft[i*wind_size:(i+1)*wind_size])
        # find peaks in fft 
        peaks_tmp, _ = scipy.signal.find_peaks(fft[i*wind_size:(i+1)*wind_size], height=mean + 8*sd)
        peaks.extend([p + i*wind_size for p in peaks_tmp])
    fs = 2000
    filtered_signal = signal_to_remove.copy()
    for p in peaks:
        f_noise = freq[p]
        w0 = f_noise 
        Q = 180  # Quality factor (determines bandwidth)

        # Design the notch filter
        b, a = scipy.signal.iirnotch(w0, Q, fs=fs)

        # Apply the filter to the signal
        filtered_signal = scipy.signal.filtfilt(b, a, filtered_signal)
        
    new_signal = filtered_signal.copy()
    new_signal[:5000] = signal_to_remove[:5000]
    new_signal[-5000:] = signal_to_remove[-5000:]

    original_fft = np.fft.fft(new_signal)
    # get peaks in fft usng moving windows and find peak function in scipy
    freq = np.fft.fftfreq(len(original_fft), 1 / 2000)
    # plt.plot(new_signal)
    # plt.show()
    fft = np.abs(original_fft[freq > 0])
    freq = freq[freq > 0]
    wind_size = 500
    total_wind = int(len(fft) / wind_size)
    total_idx = 0
    peaks = []
    for i in range(total_wind):
        mean = np.mean(fft[i*wind_size:(i+1)*wind_size])
        sd = np.std(fft[i*wind_size:(i+1)*wind_size])
        # find peaks in fft 
        peaks_tmp, _ = scipy.signal.find_peaks(fft[i*wind_size:(i+1)*wind_size], height=mean + 8*sd)
        peaks.extend([p + i*wind_size for p in peaks_tmp])
    fs = 2000
    filtered_signal = new_signal.copy()
    for p in peaks:
        f_noise = freq[p]
        w0 = f_noise 
        Q = 180  # Quality factor (determines bandwidth)

        # Design the notch filter
        b, a = scipy.signal.iirnotch(w0, Q, fs=fs)

        # Apply the filter to the signal
        filtered_signal = scipy.signal.filtfilt(b, a, filtered_signal)
    # plt.figure('signal')
    # plt.plot(new_signal[:], label='Original signal')
    # plt.plot(filtered_signal[:], label='notch signal')
    # plt.show()
    
    original_fft = np.fft.fft(signal_to_remove[n_batch-batch_tot * window_size:n_batch*window_size])
    cleaned_fft = np.fft.fft(total_signal[n_batch-batch_tot * window_size:n_batch*window_size])
    filtered_fft  = np.fft.fft(filtered_signal[n_batch-batch_tot * window_size:n_batch*window_size])
    x_fft = np.fft.fftfreq(signal_to_remove[n_batch-batch_tot * window_size:n_batch*window_size].shape[0], 1/2000)
    plt.figure('frequency')
    plt.plot(x_fft, np.abs(original_fft), label='Original FFT')
    plt.plot(x_fft, np.abs(cleaned_fft), label='Cleaned FFT')
    plt.plot(x_fft, np.abs(filtered_fft), label='Notch filtered FFT')
    plt.legend()
    
    plt.figure('signal')
    # plt.plot(signal_to_remove[n_batch-batch_tot * window_size:n_batch*window_size], label='Original signal')
    plt.plot(filtered_signal[n_batch-batch_tot * window_size:n_batch*window_size], label='notch signal')
    plt.plot(total_signal[n_batch-batch_tot * window_size:n_batch*window_size], label='Cleaned signal')
    # plt.plot(new_signal[n_batch-batch_tot * window_size:n_batch*window_size], label='notch signal')

    plt.legend()
    plt.show()




    


