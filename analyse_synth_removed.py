from automatic_remover import ArtefactRemover
import os
import numpy as np
from biosiglive import load, save


if __name__ == '__main__':
    path_file = "synth_data_with_artifact_all.bio"
    txt_file = 'result_optim_synth_data_with_artifact_all.txt'
    json_win_path = 'synth_data_window.json'
    #data = load(path_file)
    frame_idx = 0
    idx_to_remove = 0
    txt_file = "result_optim_synth_data_with_artifact_all.txt" 
    # size = 1500
    if os.path.exists(txt_file):
        with open(txt_file) as f:
            data = f.readlines()
            keys = [d.split(",")[0] for d in data]
            sizes = [float(d.split(",")[1]) for d in data]
    results = {}
    artefact_remover = ArtefactRemover(data=path_file, plot_figure=True)
    channel_ref = np.array([[c] * 5 for c in artefact_remover.chanels_names if "_0_" in c]).flatten()
    all_channels = [c for c in artefact_remover.chanels_names if "_0_"  not in c and "15_hz" not in c]
    for c, channel in enumerate(all_channels):
        print("Working on channel: ", channel)
        ref = channel_ref[c]
        size = int(sizes[keys.index(channel)])
        artefactless_signal = artefact_remover.init_data[0, :, artefact_remover.chanels_names.index(ref)]
        signal_to_remove = artefact_remover.init_data[0, :, artefact_remover.chanels_names.index(channel)]
        artefact_remover.signal_decomposition(signal_to_remove,
                                            hankel_size=size, artefactless_signal=artefactless_signal,
                                             threshold=None)
        artefact_remover.compute_signal_error(
            original_signal=signal_to_remove,
            reduced_signal=artefact_remover.signal_reduced, artefactless_signal=artefactless_signal,
            json_path=json_win_path)
        artefact_remover.compute_frequency_analysis(
            original_signal=signal_to_remove,
            reduced_signal=artefact_remover.signal_reduced,
            artefactless_signal=artefactless_signal)
        results[channel] = {"reduced_ratio": artefact_remover.ratio, 
                            "original_ratio": artefact_remover.initial_ratio,
                            "artifactfree_ratio": artefact_remover.artefactless_ratio,
                            "pearson": artefact_remover.pearson,
                            "original_pearson": artefact_remover.initial_pearson,
                            "original_mdf": artefact_remover.mdfs[0],
                            "reduced_mdf": artefact_remover.mdfs[1],
                            "artifactfree_mdf": artefact_remover.mdfs[-1],
                            }
        save(results, path_file.replace(".bio", "_results_final.bio"), add_data=True)
        
        #artefact_remover.plot()
