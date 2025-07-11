import numpy as np

from automatic_remover import ArtefactRemover
from scipy.optimize import minimize_scalar, minimize
from biosiglive import load
from numba import njit

class Optimizer:
    def __init__(self, data_path):
        self.artefact_remover = ArtefactRemover(data_path, plot_figure=False)

    @staticmethod
    def _cost_function(param, artefact_remover, channel_idx=-1, json_path=None, frame=0):
        if isinstance(param, list):
            param = param[0]
        signal_to_remove = artefact_remover.init_data[frame, :, channel_idx]
        artefact_remover.signal_decomposition(signal_to_remove,
                                                   hankel_size=param)
        artefact_remover.compute_signal_error(
            original_signal=signal_to_remove,
            reduced_signal=artefact_remover.signal_reduced,
            json_path=json_path)
        artefact_remover.compute_frequency_analysis(
            original_signal=signal_to_remove,
            reduced_signal=artefact_remover.signal_reduced)
        
        mdf = artefact_remover.mdfs[1] / 100
        ratio = 1 / artefact_remover.ratio
        return mdf + ratio

    def optimize_single(self, channel_idx=-1, json_window_path=None, json_path=None, frame=0, save_in_file=False, process_signal=False):
        if json_path is None:
            json_path = ".json"
        json_file_path = json_path.split('.')[0] + '_optimized.txt'
        f = lambda x: self._cost_function(x, self.artefact_remover, channel_idx, json_window_path, frame=frame)
        res = minimize_scalar(f, bounds=(10, 2000), method='bounded')
        signal_cleaned = None
        if save_in_file:
            # add line to a text file
            with open(json_file_path, 'a') as f:
                f.write(f"{frame},{res.x},{ res.fun}\n")
        if process_signal:
            self.artefact_remover.signal_decomposition(self.artefact_remover.init_data[frame, :, channel_idx],
                                                       hankel_size=res.x, backward_pass=False)
            signal_cleaned = self.artefact_remover.signal_reduced[0, :]
        return res.x, res.fun, signal_cleaned

    def optimize(self, channel_idx=-1, json_path=None, process_signal=False, save_in_file=False):
        x_list = []
        fun_list = []
        signal_cleaned = []
        for f in range(self.artefact_remover.init_data.shape[0]):
            x, fun, _ = self.optimize_single(channel_idx, json_path, f, save_in_file=save_in_file, process_signal=process_signal)
            x_list.append(x)
            fun_list.append(fun)
            if process_signal:
                signal_cleaned.append(self.artefact_remover.signal_reduced[0, :])
        return x_list, fun_list, signal_cleaned
    
    



if __name__ == '__main__':
    path_file = "synth_data_with_artifact_all.bio"
    txt_file = 'result_optim_synth_data_with_artifact_all.txt'
    json_cin_path = 'synth_data_window.json'
    data = load(path_file)
    keys = ["data_with_ratio_0.3_15_hz"]
    import time
    for key in keys: 
        if "ratio_0_" in key:
            continue
        tic = time.time()
        x, fun, signal_cleaned = Optimizer(data[key][None, :, None]).optimize_single(0, json_cin_path, None, 0, save_in_file=False)
        with open(txt_file, 'a') as f:
            f.write(f"{key},{x},{fun},{time.time()-tic}\n")



