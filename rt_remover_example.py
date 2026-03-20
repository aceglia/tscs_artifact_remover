import numpy as np
from artifact_remover.rt_automatic_remover import RtArtefactRemover
from artifact_remover.processing_utils import median_frequency, robust_max_percentile
import matplotlib.pyplot as plt
from biosiglive import OfflineProcessing


if __name__ == "__main__":
    # Data need to be either a path to a file or a numpy array
    # If data is a numpy array, it should be of shape (n_batch, n_channels, n_samples)
    path_file = r'D:\Downloads\T1_008_arm_sa_002.mat'
    path_file = r'D:\Documents\Udem\Postdoctorat\stim_spinal_membre_sup\mapping\010_arm_sa_T1_map003.mat'
    # path_file = r'D:\Documents\Programmation\tscs_artifact_remover\007Loc_sa_20_Avec000.mat'

    # path_file = r"test001.txt"
    # path_file = r"test_stim_artifact.txt"

    process_window = 500
    h_delay = 1
    h_size = int((process_window/4) / h_delay)
    print('Hankel matrix size is: ', '(', h_size, ',', process_window - (h_size - 1) * h_delay, ')')
    notch_filter=False
    artefact_remover = RtArtefactRemover(data=path_file, chunk_size=20, window_size=process_window, signal_filter=False, center=True, cutoff=[10, 500])
    output = artefact_remover.process_all_data(
        hankel_size=h_size,
        randomized=False,
        channel_idxs=[3],
        data_window=[40000, 60000],
        # data_window=[artefact_remover.get_init_signal().shape[-1] - 50000, artefact_remover.get_init_signal().shape[-1]-10000],
        threads=1,
        nb_principal_components=70,
        epsilon=None,
        notch_filter=notch_filter,
        quality_factor=100,
        frequency_peaks=30,
        hankel_delay=h_delay, 
        factor=0.35, 
        freq_bounds=[10, 300]
    )
    init_data = artefact_remover.streamer.data_loader.apply_filtering(artefact_remover.get_init_signal())[0, -1, :]
    # init_data = artefact_remover.get_init_signal()[0, -1, :]
    off_proc = OfflineProcessing(artefact_remover.streamer.data_loader.data_rate)
    env_init = off_proc.process_emg(init_data[None, :], moving_average=False, low_pass_filter=True)
    env_out = off_proc.process_emg(output[0, 0, :][None, :], moving_average=False, low_pass_filter=True)

    fig, axes = plt.subplots(2, 1, sharey=True, sharex=True, num='Signal')
    axes[0].plot(init_data, 'r')
    axes[0].plot(output[0, 0, :], 'b', alpha=0.5)
    axes[1].plot(output[0, 0, :], 'b')

    plt.figure("Enveloppes")  
    plt.plot(env_init[0], 'r--')
    plt.plot(env_out[0], 'b--')

    from scipy.fft import rfft, rfftfreq
    
    plt.figure("fft")
    h_2000 = 20000
    h_2141 = 2141 * 20000/2000
    # resample init data to 2141 Hz : 
    from scipy import signal
    init_data_2141 = signal.resample(init_data, 21410)
    freqs_rs = rfftfreq(init_data_2141.shape[-1], d=1/2141)
    abs_fft_init = np.abs(rfft(init_data))
    abs_fft_rs = np.abs(rfft(init_data_2141))
    abs_fft = np.abs(rfft(output[0, 0, :]))
    freq = rfftfreq(init_data.shape[-1], d=1/artefact_remover.streamer.data_loader.data_rate)
    freq_rs = rfftfreq(init_data.shape[-1], d=1/2141)
    plt.plot(freq, abs_fft_init)
    plt.hlines(robust_max_percentile(np.sort(abs_fft_init), 99.2), 
               plt.xlim()[0], plt.xlim()[1], colors='b', linestyles='--', label='mean')
    plt.vlines(median_frequency(init_data, artefact_remover.streamer.data_loader.data_rate), 
               plt.ylim()[0], plt.ylim()[1], colors='b', linestyles='--', label='median')
    plt.plot(freq, abs_fft)
    plt.plot(freq_rs, abs_fft_init)
    plt.hlines(robust_max_percentile(np.sort(abs_fft), 99.9), 
               plt.xlim()[0], plt.xlim()[1], colors='orange', linestyles='--', label='mean')
    plt.vlines(median_frequency(output[0, 0, :], artefact_remover.streamer.data_loader.data_rate), 
               plt.ylim()[0], plt.ylim()[1], colors='orange', linestyles='--', label='median')
    plt.show(block=True)
