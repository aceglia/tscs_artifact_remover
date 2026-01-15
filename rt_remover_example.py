from artifact_remover.rt_automatic_remover import RtArtefactRemover


if __name__ == "__main__":
    # Data need to be either a path to a file or a numpy array
    # If data is a numpy array, it should be of shape (n_batch, n_channels, n_samples)
    path_file = r"D:\Documents\Programmation\tscs_artifact_remover\004_TN-SCI_002.txt"
    path_file = r'D:\Downloads\T1_008_arm_sa_002.mat'
    path_file = r"test001.txt"
    notch_filter=True
    artefact_remover = RtArtefactRemover(data=path_file, chunk_size=20)
    sol = artefact_remover.process_all_data(
        hankel_size=300,
        randomized=False,
        # batch_idxs=[0],
        channel_idxs=[-1],
        # data_window=[artefact_remover.get_init_signal().shape[-1] - 10000, artefact_remover.get_init_signal().shape[-1]],
        threads=1,
        nb_principal_components=50,
        notch_filter=notch_filter,
        quality_factor=150,
        frequency_peaks=30,
    )
    results = sol.analyse(compute_signal_error=False)
    sol.plot(signals=True, fft=True, singular_values=notch_filter == False, stack_batch=False, show_analysis=False)
