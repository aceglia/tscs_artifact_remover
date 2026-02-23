from artifact_remover.automatic_remover import ArtefactRemover
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Data need to be either a path to a file or a numpy array
    # If data is a numpy array, it should be of shape (n_batch, n_channels, n_samples)
    path_file = r"D:\Documents\Programmation\tscs_artifact_remover\004_TN-SCI_002.txt"
    path_file = r'D:\Downloads\T1_008_arm_sa_002.mat'
    # path_file = r'D:\Documents\Programmation\tscs_artifact_remover\007Loc_sa_20_Avec000.mat'
    # path_file = r"test001.txt"
    process_window = 5000
    h_delay = 5
    h_size = int(-((process_window / 2) - process_window) / h_delay)


    notch_filter=False
    artefact_remover = ArtefactRemover(data=path_file,  signal_filter=True, center=True, cutoff=[10, 500])
    sol = artefact_remover.process(
        hankel_size=h_size,
        randomized=False,
        # batch_idxs=list(range(10, 11)),
        channel_idxs=[-1],
        # data_window=[0, 8000],
        data_window=[artefact_remover.get_init_signal().shape[-1] - (process_window * 4), artefact_remover.get_init_signal().shape[-1]-1000],
        # data_window=[0, process_window],
        threads=1,
        nb_principal_components=None,
        post_filter=False,
        notch_filter=notch_filter,
        quality_factor=30,
        frequency_peaks=30,
        hankel_delay=h_delay,
        process_window=process_window,
    )
    results = sol.analyse(compute_signal_error=False)
    sol.plot(signals=True, fft=True, singular_values=notch_filter == False, stack_batch=False, show_analysis=False)
