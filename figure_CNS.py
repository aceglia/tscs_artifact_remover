import matplotlib.pyplot as plt
import os
from automatic_remover import ArtefactRemover
import numpy as np

def get_fft(signal, rate=2000):
    n = signal.shape[0]
    freq = np.fft.fftfreq(n, 1 / rate)
    fft = np.fft.fft(signal)
    amp = np.abs(fft[freq > 0])
    energy = amp ** 2
    energy_cumsum = np.cumsum(energy)
    mdfs = freq[np.where(energy_cumsum > np.max(energy_cumsum) / 2)[0][0]]
    freq_pos = freq[freq > 0]
    return freq_pos, np.abs(fft[freq > 0]), int(mdfs)

if __name__ == '__main__':
    synth = True
    # path_file = "synth_stim_artifact.bio" if synth else r"test_stim_artifact.txt"
    path_file = "synth_stim_artifact_new.bio" if synth else r"test001.txt"
    frame_idx = 0 if synth else 24  # index of the frame to process
    # json_path = 'windows_synth_signal.json' if synth else 'windows_test_signal.json'
    json_path = 'windows_synth_signal_new.json' if synth else f'test_emg_{frame_idx}.json'
    idx_to_remove = 2 if synth else 0  # index of the channel to remove

    artefact_remover = ArtefactRemover(path=path_file, plot_figure=True)
    signal_to_remove = artefact_remover.init_data[frame_idx, :, idx_to_remove]
    artefactless_signal = artefact_remover.init_data[frame_idx, :, 0] if synth else None
    txt_file = json_path.replace(".json", "_optimized.txt")
    size = 1500
    if os.path.exists(txt_file):
        with open(json_path.replace(".json", "_optimized.txt")) as f:
            data = f.readlines()
            size = int(float(data[-1].split(",")[1]))
    artefact_remover.signal_decomposition(signal_to_remove,
                                          hankel_size=size, artefactless_signal=artefactless_signal,
                                          backward_pass=False, threshold=None)

    artefact_remover.compute_signal_error(
        original_signal=signal_to_remove,
        reduced_signal=artefact_remover.signal_reduced, artefactless_signal=artefactless_signal,
        json_path=json_path)
    artefact_remover.compute_frequency_analysis(
        original_signal=signal_to_remove,
        reduced_signal=artefact_remover.signal_reduced,
        artefactless_signal=artefactless_signal)
    
    signal_clean = artefact_remover.signal_reduced
    signal_init = artefact_remover.init_data[frame_idx, :, idx_to_remove]
    pure_emg = artefact_remover.init_data[frame_idx, :, 0]
    freq, fft_signal, mdfs = get_fft(signal_init)
    freq_clean, fft_clean, mdfs_clean = get_fft(signal_clean)
    freq_emg, fft_emg, mdfs_emg = get_fft(pure_emg)
    plt.figure("fft with artefact")
    plt.plot(freq, fft_signal, label="signal with artefact")
    plt.axvline(mdfs, color='r', linestyle='--', label="mdfs")
    plt.figure("fft without artefact")
    plt.plot(freq_clean, fft_clean, label="signal without artefact")
    plt.axvline(mdfs_clean, color='g', linestyle='--', label="mdfs clean")
    plt.figure("pure emg")
    plt.plot(freq_emg, fft_emg, label="pure emg")
    plt.axvline(mdfs_emg, color='b', linestyle='--', label="mdfs emg")
    plt.legend()
    plt.show()

    artefact_remover.plot()
