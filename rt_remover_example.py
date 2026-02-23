import numpy as np
from artifact_remover.rt_automatic_remover import RtArtefactRemover
from artifact_remover.processing_utils import median_frequency, robust_max_percentile
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Data need to be either a path to a file or a numpy array
    # If data is a numpy array, it should be of shape (n_batch, n_channels, n_samples)
    path_file = r"D:\Documents\Programmation\tscs_artifact_remover\004_TN-SCI_002.txt"
    path_file = r'D:\Downloads\T1_008_arm_sa_002.mat'
    path_file = r'D:\Documents\Programmation\tscs_artifact_remover\007Loc_sa_20_Avec000.mat'

    # path_file = r"test001.txt"
    notch_filter=False
    artefact_remover = RtArtefactRemover(data=path_file, chunk_size=20, window_size=1000, signal_filter=True, center=True)
    output = artefact_remover.process_all_data(
        hankel_size=100,
        randomized=False,
        channel_idxs=[-1],
        # data_window=[0, 5000],
        data_window=[artefact_remover.get_init_signal().shape[-1] - 10000, artefact_remover.get_init_signal().shape[-1]-1000],
        threads=1,
        nb_principal_components=None,
        epsilon=None,
        notch_filter=notch_filter,
        quality_factor=100,
        frequency_peaks=30,
        hankel_delay=5
    )
    plt.figure("signal")
    plt.plot(artefact_remover.get_init_signal()[0, -1, :])
    plt.plot(output[0, 0, :])

    from scipy.fft import rfft, rfftfreq
    
    plt.figure("fft")
    abs_fft_init = np.abs(rfft(artefact_remover.get_init_signal()[0, -1, :]))
    abs_fft = np.abs(rfft(output[0, 0, :]))
    freq = rfftfreq(artefact_remover.get_init_signal().shape[-1], d=1/artefact_remover.streamer.data_loader.data_rate)
    plt.plot(freq, abs_fft_init)
    plt.hlines(robust_max_percentile(np.sort(abs_fft_init), 99.2), 
               plt.xlim()[0], plt.xlim()[1], colors='b', linestyles='--', label='mean')
    plt.vlines(median_frequency(artefact_remover.get_init_signal()[0, -1, :], artefact_remover.streamer.data_loader.data_rate), 
               plt.ylim()[0], plt.ylim()[1], colors='b', linestyles='--', label='median')
    plt.plot(freq, abs_fft)
    plt.hlines(robust_max_percentile(np.sort(abs_fft), 99.9), 
               plt.xlim()[0], plt.xlim()[1], colors='orange', linestyles='--', label='mean')
    plt.vlines(median_frequency(output[0, 0, :], artefact_remover.streamer.data_loader.data_rate), 
               plt.ylim()[0], plt.ylim()[1], colors='orange', linestyles='--', label='median')
    plt.show(block=True)
