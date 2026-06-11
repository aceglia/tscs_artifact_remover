import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import rfft, rfftfreq

from artifact_remover.generator import ArtifactGenerator
from artifact_remover.io_utils import DataLoader
from artifact_remover.generator_utils import Modulator

from biosiglive import load, save

if __name__ == "__main__":
    generator = ArtifactGenerator()
    data_path = r"D:\Documents\Programmation\tscs_artifact_remover\test_data\clean_walk_test.mat"
    data_path = r"D:\Documents\Programmation\tscs_artifact_remover\test_data\test001.txt"
    data_loader = DataLoader(data=data_path, signal_filter=False, center=True, cutoff=[10, 500])
    data = data_loader.init_data
    signal = np.asarray(data[:, 0, 0:9629], np.float64)
    signal = np.hstack([signal[0]]*5)
    # plt.plot(signal)
    # plt.show(block=True)
    freq = [50]
    mod = ["steps"]
    for m in mod:
        modulator = Modulator(modulation_type=m, min=0, max=1.2, step_inc=None, step_length=9629)
        for f in freq:
            data_with_artifacts = generator.apply_artifact_to_signal(
                signal,
                artifact_duration=[0.0055, 0.006],
                stimulation_frequency=f,
                sampling_rate=data_loader.data_rate,
                delay_1=[0.1, 0.15],
                delay_2=[0.22, 0.27],
                num=[1],
                den=[[0.02, 0.025], [0.4, 0.5], [14, 18]],
                amplitude=np.max(signal),
                phase_inversion=False,
                factors = [[0.95, 1.05], [1.95, 2.05], [0.95, 1.05]],
                modulator=modulator,
            )
            # plt.plot(data_with_artifacts)
            # plt.show(block=True)
            generator.save(data_path.replace('.txt', f"with_artifacts_synth_{f}_{m}.bio"))
            data = load(data_path.replace(".txt", f"with_artifacts_synth_{f}_{m}.bio"))
            data["values"] = data["signal_with_artifacts"]
            data["data_rate"] = data_loader.data_rate
            save(data, data_path.replace(".txt", f"with_artifacts_synth_{f}_{m}.bio"), safe=False)
    # time = np.linspace(0, len(data_with_artifacts[0]) / generator.fs, len(data_with_artifacts[0]))
    # plt.plot(time, data_with_artifacts[0])

    # plt.figure("fft")
    # plt.plot(rfftfreq(len(data_with_artifacts[0]), 1 / generator.fs), abs(rfft(data_with_artifacts[0])))
    # plt.show(block=True)
