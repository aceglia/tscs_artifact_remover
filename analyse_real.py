from automatic_remover import ArtefactRemover
import os
import numpy as np
from biosiglive import load, save


if __name__ == '__main__':
    path_file = "test001.txt"
    #data = load(path_file)
    frame_idx = list(range(1, 7))
    idx_to_remove = 0
    results = {}
    artefact_remover = ArtefactRemover(data=path_file, plot_figure=True)
    for i in frame_idx:
        txt_file = f"test_emg_{i}_optimized.txt" 
        json_win_path = f'test_emg_{i}.json'
        # size = 1500
        if os.path.exists(txt_file):
            with open(txt_file) as f:
                data = f.readlines()
                keys = [d.split(",")[0] for d in data]
                sizes = [float(d.split(",")[1]) for d in data][-1]
        print("Working on frame: ", i)
        size = int(sizes)
        signal_to_remove = artefact_remover.init_data[i, :, idx_to_remove]
        artefact_remover.signal_decomposition(signal_to_remove,
                                            hankel_size=size, artefactless_signal=None,
                                             threshold=None)
        artefact_remover.compute_signal_error(
            original_signal=signal_to_remove,
            reduced_signal=artefact_remover.signal_reduced, artefactless_signal=None,
            json_path=json_win_path)
        artefact_remover.compute_frequency_analysis(
            original_signal=signal_to_remove,
            reduced_signal=artefact_remover.signal_reduced,
            artefactless_signal=None)
        results[i] = {"reduced_ratio": artefact_remover.ratio, 
                            "original_ratio": artefact_remover.initial_ratio,
                            # "artifactfree_ratio": artefact_remover.artefactless_ratio,
                            # "pearson": artefact_remover.pearson,
                            "original_mdf": artefact_remover.mdfs[0],
                            "reduced_mdf": artefact_remover.mdfs[1],
                            # "artifactfree_mdf": artefact_remover.mdfs[-1],
                            }
        # save(results, path_file.replace(".bio", "_results.bio"), add_data=True)
        
        artefact_remover.plot()
        import matplotlib.pyplot as plt
        plt.show()

