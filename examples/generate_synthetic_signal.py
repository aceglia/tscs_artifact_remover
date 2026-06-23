import numpy as np

from star_emg.generator import ArtifactGenerator
from star_emg.io_utils import DataLoader
from star_emg.generator_utils import Modulator

from biosiglive import load, save

if __name__ == "__main__":
    generator = ArtifactGenerator()
    data_path = r"data\clean_emg.txt"
    data_loader = DataLoader(data=data_path, signal_filter=False, center=True, cutoff=[10, 500])
    data = data_loader.init_data
    signal = np.asarray(data[:, 0, :], np.float64)
    signal = np.hstack([signal[0]] * 5)
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
                factors=[[0.95, 1.05], [1.95, 2.05], [0.95, 1.05]],
                modulator=modulator,
            )
            generator.save(data_path.replace(".txt", f"with_artifacts_synth_{f}_{m}.bio"))
