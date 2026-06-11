from artifact_remover.automatic_remover import ArtefactRemover
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Data need to be either a path to a file or a numpy array
    # If data is a numpy array, it should be of shape (n_batch, n_channels, n_samples)
    path_file = r"D:\Documents\Programmation\tscs_artifact_remover\004_TN-SCI_002.txt"
    path_file = r"D:\Downloads\T1_008_arm_sa_002.mat"
    path_file = r"D:\Documents\Programmation\tscs_artifact_remover\test_data\007Loc_sa_20_Avec000.txt"
    path_file = r'D:\Documents\Programmation\tscs_artifact_remover\test_data\clean_walk_testwith_artifacts_synth_80_steps.bio'

    # path_file = r"D:\Documents\Programmation\tscs_artifact_remover\007Loc_sa_20_Avec000_processed.csv"
    path_file = r"D:\Documents\Programmation\tscs_artifact_remover\test_data\test001with_artifacts_synth_50_steps.bio"
    process_window = 500
    h_delay = 1
    h_size = int((process_window / 8) / h_delay)
    print("Hankel matrix size is: ", "(", h_size, ",", process_window - (h_size - 1) * h_delay, ")")
    notch_filter = True

    artefact_remover = ArtefactRemover(data=path_file, signal_filter=True, center=True, cutoff=[10, 500])
    import numpy as np 
    np.savetxt("signal.txt", artefact_remover.get_init_signal()[-1, 0, :])
    plt.plot(artefact_remover.get_init_signal()[-1, 0, :])
    plt.show(block=True)
    sol = artefact_remover.process(
        hankel_size=h_size,
        randomized=False,
        # batch_idxs=list(range(9, 10)),
        # data_window=[0, 10000],
        data_window=[
            artefact_remover.get_init_signal().shape[-1] - (process_window * 4),
            artefact_remover.get_init_signal().shape[-1] - 1000,
        ],
        # data_window=[process_window, process_window*20],
        channel_idxs=None,
        first_peak=40,
        threads=1,
        nb_principal_components=None,
        post_filter=False,
        notch_filter=notch_filter,
        quality_factor=10,
        frequency_peaks=80,

        hankel_delay=h_delay,
        process_window=process_window,
    )
    results = sol.analyse(compute_signal_error=False)
    sol.plot(signals=True, fft=True, singular_values=notch_filter == False, stack_batch=False, show_analysis=False)
