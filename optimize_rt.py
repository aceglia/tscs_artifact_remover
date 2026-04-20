from functools import partial

import numpy as np
from artifact_remover.rt_automatic_remover import RtArtefactRemover
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
from artifact_remover.solution import Solution
from biosiglive import LivePlot, PlotType
from artifact_remover.processing_utils import Quality

if __name__ == "__main__":
    # Data need to be either a path to a file or a numpy array
    # If data is a numpy array, it should be of shape (n_batch, n_channels, n_samples)
    path_file = r"D:\Documents\Programmation\tscs_artifact_remover\004_TN-SCI_002.txt"
    path_file = r"D:\Downloads\T1_008_arm_sa_002.mat"
    # path_file = r"test001.txt"
    # path_file = r'D:\Documents\Programmation\tscs_artifact_remover\test_data\007Loc_sa_20_Avec000.mat'
    # path_file =  r"D:\Documents\Programmation\tscs_artifact_remover\test_data\clean_walk_testwith_artifacts_synth_80_steps.bio"

    notch_filter = False
    process_window = 700
    h_delay = 2
    h_size = int((process_window / 4) / h_delay)

    artefact_remover = RtArtefactRemover(
        data=path_file, chunk_size=20, window_size=process_window, center=True, filter=False, cutoff=[10, 500]
    )
    current_idx = (artefact_remover.get_init_signal().shape[-1] - 10000) // artefact_remover.streamer.chunk_size
    current_idx = artefact_remover.get_init_signal().shape[-1] - 10000
    # current_idx = 40000

    artefact_remover.streamer.current_index = current_idx
    is_data = True
    n_principal_component = None

    shape_1 = n_principal_component if n_principal_component else h_size
    all_data = np.zeros((100, process_window))
    out = np.zeros((100, process_window))
    quality = Quality(shape=(1, 1, 600))
    fft_freqs = rfftfreq(process_window, 1 / artefact_remover.streamer.data_loader.data_rate)
    fft_freqs_quality = rfftfreq(1000, 1 / artefact_remover.streamer.data_loader.data_rate)
    quality_fct = partial(
        quality.compute_quality,
        ground_truth=None,
        channel=0,
        idx=0,
        fs=artefact_remover.streamer.data_loader.data_rate,
        kw=20,
        fft_freqs=fft_freqs_quality,
    )
    plot_curve = LivePlot(
        name="curve",
        rate=60,
        plot_type=PlotType.Curve,
        nb_subplots=3,
        channel_names=["Raw", "Cleaned", "quality"],
    )
    plot_curve.init(plot_windows=10000, y_labels=["Volts", "Volts", "NA"])
    c = 0
    toc = 0
    import time
    while is_data:
        is_data, chunk = artefact_remover.streamer.get_next_chunk(artefact_remover.streamer.chunk_size)
        if chunk is None:
            break
        chunk = chunk[2:3, :][None]
        if not artefact_remover.buffer.full:
            artefact_remover.buffer.append(chunk)
            artefact_remover.to_evaluate_buffer.append(np.hstack([chunk, np.zeros_like(chunk)]))
        else:
            artefact_remover.buffer.append(chunk)
            data = artefact_remover.buffer.get()
            data_proc = artefact_remover.streamer.data_loader.apply_filtering(data)
            dict = artefact_remover._remove_artifact_from_windows(
                data=data_proc[0, 0, :],
                hankel_size=h_size,
                randomized=False,
                nb_principal_components=n_principal_component,
                filter=False,
                epsilon=None,
                return_dict=False,
                # n_reconstruct=artefact_remover.streamer.chunk_size,
                n_reconstruct=None,
                hankel_delay=h_delay,
                offline=False,
                data_rate=artefact_remover.streamer.data_loader.data_rate,
                freq_bounds=[10, 150],
                factor=0.2,
                fft_freqs=fft_freqs,
                notch_filter=notch_filter,
                quality_factor=15,
                frequency_peaks=30,
                first_peak=45,
            )
            artefact_remover.to_evaluate_buffer.append(
                np.hstack([chunk, dict[None, None, -artefact_remover.streamer.chunk_size :]])
            )
            if artefact_remover.to_evaluate_buffer.full:
                data_to_eval = artefact_remover.to_evaluate_buffer.get()
                
                quality_fct(raw=None, processed=data_to_eval[0, 1], analysis=[1, 2, 3])
                # fft_data = abs(rfft(data_to_eval[0, :], axis=-1))
                # plt.plot(fft_freqs_quality, fft_data.T)
                # plt.show(block=True)
                quality = quality_fct(raw=data_to_eval[0, 0], processed=None, analysis=[3])
                ll = np.sum(np.abs(np.diff(data_to_eval[0, 1])), axis=-1) / np.std(data_to_eval[0, 1])
                line_length = quality[1][1].item() / 100  # min
                kurt_proc = np.max(np.abs(data_to_eval[0, 1])) / np.mean(np.abs(data_to_eval[0, 1])) / 10
                # med_freq = abs(80 - quality[1][2].item()) / 100  # min
                med_freq = abs(1 - (quality[1][2].item() / 80))
                max_freq = abs(1 - (quality[1][3].item() / quality[0][3].item()))
                # max_freq = np.sqrt(np.mean((quality[0][3].item() - quality[1][3].item()) ** 2))  # min
                criteria = np.array([np.nansum([kurt_proc, line_length, med_freq, max_freq])])
            else:
                criteria = np.array([0])
                max_freq, kurt_proc, line_length, med_freq = 0, 0, 0, 0
            plot_curve.update(
                [
                    np.concatenate((
                        data_proc[0, 0:1, -artefact_remover.streamer.chunk_size :],
                        dict[None, -artefact_remover.streamer.chunk_size :]
                    ), axis=0),
                    dict[None, -artefact_remover.streamer.chunk_size :],
                    np.concatenate(([np.repeat(np.array([max_freq])[None, 0:1], 20, axis=-1),
                            #    np.repeat(np.array([line_length])[None, 0:1], 20, axis=-1),
                               np.repeat(np.array([med_freq])[None, 0:1], 20, axis=-1),
                               np.repeat(np.array([kurt_proc])[None, 0:1], 20, axis=-1),
                               ]), axis=0)
                ]
            )
            c += 1
            if c == 10000:
                break

    result_dict = {
        "output": out.flatten(),
        "unfiltered_signal": out.flatten(),
        "data": all_data.flatten(),
    }
    sol = Solution(data_rate=artefact_remover.streamer.data_loader.data_rate)
    sol.from_signal_decomposition(result_dict, (100, 1, process_window))
    sol.plot(signals=True, fft=True, singular_values=notch_filter == False, stack_batch=False, show_analysis=False)
