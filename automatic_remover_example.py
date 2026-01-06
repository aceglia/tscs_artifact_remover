from artifact_remover.automatic_remover import ArtefactRemover


if __name__ == "__main__":
    # Data need to be either a path to a file or a numpy array
    # If data is a numpy array, it should be of shape (n_batch, n_channels, n_samples)
    path_file = r"D:\Documents\Programmation\tscs_artifact_remover\004_TN-SCI_002.txt"
    artefact_remover = ArtefactRemover(data=path_file)
    sol = artefact_remover.process(
        hankel_size=100,
        randomized=True,
        # batch_idxs=[0, 1],
        # channel_idxs=[0],
        # data_window=[0, 5000],
        threads=6,
    )
