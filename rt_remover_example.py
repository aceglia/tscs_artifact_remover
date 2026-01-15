from artifact_remover.rt_automatic_remover import RtArtefactRemover
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Data need to be either a path to a file or a numpy array
    # If data is a numpy array, it should be of shape (n_batch, n_channels, n_samples)
    path_file = r"D:\Documents\Programmation\tscs_artifact_remover\004_TN-SCI_002.txt"
    path_file = r'D:\Downloads\T1_008_arm_sa_002.mat'
    # path_file = r"test001.txt"
    notch_filter=False
    artefact_remover = RtArtefactRemover(data=path_file, chunk_size=20, window_size=500)
    output = artefact_remover.process_all_data(
        hankel_size=90,
        randomized=False,
        # batch_idxs=[0],
        channel_idxs=[-1],
        data_window=[artefact_remover.get_init_signal().shape[-1] - 8000, artefact_remover.get_init_signal().shape[-1]],
        threads=1,
        nb_principal_components=50,
        notch_filter=notch_filter,
        quality_factor=100,
        frequency_peaks=30,
    )

    plt.plot(artefact_remover.get_init_signal()[0, 0, :])
    plt.plot(output[0, 0, :])
    plt.show(block=True)
    # results = sol.analyse(compute_signal_error=False)
    # sol.plot(signals=True, fft=True, singular_values=notch_filter == False, stack_batch=False, show_analysis=False)
