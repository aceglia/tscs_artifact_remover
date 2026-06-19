"""
This file is an example of how to use the ArtifactRemover class from the star_emg library.
It demonstrates how to process data and analyze the results. The data used in this example is a file located at 'data\example.mat'.
"""

from star_emg.automatic_remover import ArtifactRemover

if __name__ == "__main__":
    # Data need to be either a path to a file or a numpy array
    # If data is a numpy array, it should be of shape (n_batch, n_channels, n_samples)
    path_file = r"data\test001.txt"
    artifact_remover = ArtifactRemover(data=path_file, signal_filter=True, center=True, cutoff=[10, 500])

    sol = artifact_remover.process(
        hankel_size=450,
        randomized=False,
        channel_idxs=None,
        first_peak=40,
        threads=4,
        nb_principal_components=None,
        post_filter=False,
        notch_filter=True,
        quality_factor=10,
        frequency_peaks=80,
        hankel_delay=1,
        process_window=5000,
        freq_bounds=[10, 300],
        factor=0.35,
    )
    results = sol.analyse()
    sol.plot(signals=True, fft=True, singular_values=False, stack_batch=False, show_analysis=True)
