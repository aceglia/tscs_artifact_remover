import os
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from artifact_remover.rt_automatic_remover import RtArtifactRemover
from artifact_remover.processing_utils import median_frequency, robust_max_percentile
import matplotlib.pyplot as plt
from artifact_remover.processing_utils import Quality
from biosiglive import load, save

if __name__ == "__main__":
    # Data need to be either a path to a file or a numpy array
    # If data is a numpy array, it should be of shape (n_batch, n_channels, n_samples)
    path_file = r"D:\Downloads\T1_008_arm_sa_002.mat"
    path_file = r"D:\Documents\Udem\Postdoctorat\stim_spinal_membre_sup\mapping\010_arm_sa_T1_map003.mat"
    path_file = r'D:\Documents\Programmation\tscs_artifact_remover\test_data\007Loc_sa_20_Avec000.mat'
    path_file = r'D:\Documents\Programmation\tscs_artifact_remover\test_data\clean_walk_testwith_artifacts_synth_50_steps.bio'
    path_file = r'D:\Documents\Programmation\tscs_artifact_remover\test_data\test001with_artifacts_synth_50_steps.bio'

    # path_file = r"D:\Documents\Programmation\tscs_artifact_remover\test_data\test001.txt"
    # path_file = r"D:\Documents\Programmation\tscs_artifact_remover\test_data\005TMSgaitpdt_001.mat"
    # path_file = r"test_stim_artifact.txt"

    process_window = 600
    h_delay = 1
    h_size = int((process_window / 6) / h_delay)
    h_size = 100
    print("Hankel matrix size is: ", "(", h_size, ",", process_window - (h_size - 1) * h_delay, ")")
    notch_filter = False
    artifact_remover = RtArtifactRemover(
        data=path_file, chunk_size=20, window_size=process_window, signal_filter=False, center=True, cutoff=[10, 500]
    )
    channel = 0
    output = artifact_remover.process_all_data(
        hankel_size=h_size,
        randomized=False,
        channel_idxs=[channel],
        # data_window=[40000, 60000],
        # data_window=[artifact_remover.get_init_signal().shape[-1] - 50000, artifact_remover.get_init_signal().shape[-1]-10000],
        threads=1,
        nb_principal_components=None,
        epsilon=None,
        notch_filter=notch_filter,
        quality_factor=30,
        frequency_peaks=50,
        first_peak=75,
        hankel_delay=h_delay,
        update_svd_every=1,
        factor=0.3,
        freq_bounds=[0, 200],
    )
    dic_to_save = {
        "output": output,
        "init_signal": artifact_remover.streamer.data_loader.apply_filtering(artifact_remover.get_init_signal())[channel, :],
        "data_rate": artifact_remover.streamer.data_loader.data_rate,
    }
    
    if 'synth' in path_file:
        ground_truth_data = load(path_file)['init_signal'].astype(np.float64)
    else:
        ground_truth_data = None
    dic_to_save['ground_truth_data'] = ground_truth_data
    save(dic_to_save, path_file.replace(".bio", "_processed.bio"), safe=False)

    init_data = artifact_remover.streamer.data_loader.apply_filtering(artifact_remover.get_init_signal())[-1, :]
    # quality = Quality((1, 1, init_data.shape[-1]))
    # quality.compute_quality(np.asarray(init_data).astype(np.float64)[None, None], output.astype(np.float64)[None],  fs=artifact_remover.streamer.data_loader.data_rate)
    colors = ('k', 'purple')

    fig, axes = plt.subplots(2, 1, sharey=True, sharex=True, num="Signal")
    axes[0].plot(init_data, color=colors[0], alpha=0.2)
    axes[0].plot(output[0, :], color=colors[1])
    if ground_truth_data is not None:
        axes[0].plot(ground_truth_data[0], "g", alpha=0.5)
    # plt.show(block=True)

    # init_data = artifact_remover.streamer.data_loader.apply_filtering(artifact_remover.get_init_signal())[-1, :]
    from scipy.fft import rfft, rfftfreq
    plt.figure("fft")
    # h_2000 = 20000

    # # resample init data to 2141 Hz :
    abs_fft_init = np.abs(rfft(init_data))
    abs_fft = np.abs(rfft(output[0, :]))
    freq = rfftfreq(init_data.shape[-1], d=1 / artifact_remover.streamer.data_loader.data_rate)
    plt.plot(freq, abs_fft_init, color=colors[0], label="init_data", alpha=0.5)
    plt.hlines(
        robust_max_percentile(abs_fft_init[:np.searchsorted(freq, 200)], 99.5),
        plt.xlim()[0],
        plt.xlim()[1],
        colors=colors[0],
        linestyles="--",
        label="mean",
        alpha=0.5
    )
    plt.vlines(
        median_frequency(init_data, artifact_remover.streamer.data_loader.data_rate),
        plt.ylim()[0],
        plt.ylim()[1],
        colors=colors[0],
        linestyles="--",
        label="median",
        alpha=0.5
    )
    plt.plot(freq, abs_fft, color=colors[1])
    plt.hlines(
        robust_max_percentile(abs_fft[:np.searchsorted(freq, 200)], 99.9),
        plt.xlim()[0],
        plt.xlim()[1],
        colors=colors[1],
        linestyles="--",
        label="mean",
    )
    plt.vlines(
        median_frequency(output[0, :], artifact_remover.streamer.data_loader.data_rate),
        plt.ylim()[0],
        plt.ylim()[1],
        colors=colors[1],
        linestyles="--",
        label="median",
    )
    plt.show(block=True)
