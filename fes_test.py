from pyomeca import Analogs
from automatic_remover import ArtefactRemover
import matplotlib.pyplot as plt
from biosiglive import save, load


if __name__ == '__main__':
    # load data
    # emg_names = ['bic_d.IM EMG1', 'trap_sup_d.IM EMG10', 'dors_d.IM EMG11',
    #    'pec_d.IM EMG12', 'delt_med_g.IM EMG13', 'trap_sup_g.IM EMG14',
    #    'dors_g.IM EMG15', 'pec_g.IM EMG16', 'tric_d.IM EMG2',
    #    'delt_ant_d.IM EMG3', 'delt_post.IM EMG4', 'bic_g.IM EMG5',
    #    'tri_g.IM EMG6', 'delt_ant_g.IM EMG7', 'delt_post_g.IM EMG8',
    #    'delt_med_d.IM EMG9']
    # emg_rate = 2000
    # emg = Analogs.from_c3d("D:\Downloads\endurance trial.c3d", usecols=emg_names)
    # emg_data = emg.values
    # emg_data = emg_data[:, 150000:200000]
    # save({"emg": emg_data, "emg_names": emg_names}, "test_fes.bio")
    emg_data = load("test_fes.bio")
    emg = emg_data['emg'][:, :25000]
    emg_names = emg_data['emg_names']
    channel_to_clean = 0
    artefact_remover = ArtefactRemover(path=None, plot_figure=True)
    artefact_remover.is_data_loaded = True
    artefact_remover.signal_decomposition(emg[channel_to_clean, :], hankel_size=500,
                                          threshold=None)
    signal_to_remove = emg[channel_to_clean, :]
    # artefact_remover.compute_signal_error(
    #     original_signal=signal_to_remove,
    #     reduced_signal=artefact_remover.signal_reduced)
    artefact_remover.compute_frequency_analysis(
        original_signal=signal_to_remove,
        reduced_signal=artefact_remover.signal_reduced)
    artefact_remover.plot()
