from functools import partial
from scipy.fft import rfftfreq
import multiprocessing as mp
import numpy as np
from ..automatic_remover import ArtifactRemover
from ..processing_utils import Quality, ensure_list
from typing import Union

try:
    import cma
except ImportError:
    cma = None


class Optimizer:
    def __init__(self, star_emg: ArtifactRemover, n_processes: int = 1):
        self.n_processes = n_processes
        self.star_emg = star_emg
        self.quality = Quality()

    @staticmethod
    def get_cost_from_quality(quality: float) -> float:
        kurt_proc = quality[1][0].item()
        # line_length = quality[1][1].item()
        # med_freq = abs(80 - quality[1][2].item()) / 100# min
        # max_freq = np.sqrt(np.mean((quality[0][3].item() - quality[1][3].item())**2)) * 10
        med_freq = abs(1 - (quality[1][2].item() / 80))
        max_freq = abs(1 - (quality[1][3].item() / quality[0][3].item()))
        return kurt_proc + med_freq + max_freq

    @staticmethod
    def compute_cost(x, fct: partial, data: np.ndarray, quality_fct: partial) -> float:
        data_processed = fct(data=data, hankel_size=int(x[0] * 1000), freqs_bounds=[10, x[1] * 1000], factor=x[2])
        quality_fct(raw=None, processed=data_processed)
        quality = quality_fct(raw=data, processed=None, analysis=[3])
        return Optimizer.get_cost_from_quality(quality)

    @staticmethod
    def _optimize_single(idx: int, data, fct: partial, quality_fct: partial, process_window: int) -> tuple:
        # bounds = [[0, 1] for _ in range(4)]
        lower_bounds = [0.2, 0.3, 0]
        upper_bounds = [0.8, 1, 1.5]
        init = [0.4, 0.45, 0.35]
        if cma is None:
            raise ImportError("cma module is required for optimization")
        res, es = cma.fmin2(
            lambda x: Optimizer.compute_cost(x, fct, data[:10000], quality_fct),
            init,
            0.2,
            options={
                "bounds": [lower_bounds, upper_bounds],
                "tolfun": 1e-3,
                "verb_log": 0,
                "maxiter": 80,
                "popsize": 8,
            },
        )
        print(res)
        # n_windows = data.shape[-1] // process_window
        # params_list = []
        # es = cma.CMAEvolutionStrategy(init, 0.2, options={'bounds': [lower_bounds, upper_bounds],'verb_disp': 1, 'tolfun': 1e-3, 'verb_log': 0,
        #     'popsize': 15})
        # max_iter = 50
        # es.logger.disp_header()
        # for i in range(n_windows + 1):
        #     es.stop().clear()
        #     if i == n_windows and data.shape[-1] % process_window != 0:
        #         data_tmp = data[-process_window:]
        #     else:
        #         data_tmp = data[i * process_window: (i + 1) * process_window]

        #     for _ in range(max_iter if i == 0 else 20):
        #         X = es.ask()
        #         f = [
        #             Optimizer.compute_cost(x, fct, data_tmp, quality_fct)
        #             for x in X
        #         ]
        #         es.tell(X, f)
        #         print(es.best.x, es.best.f)
        #         # es.disp()
        #         if es.stop():
        #             break
        #     # res, es = cma.fmin2(lambda x: Optimizer.compute_cost(x, fct, data_tmp, quality_fct), init, 0.2, options={'bounds': [lower_bounds, upper_bounds], 'tolfun': 1e-3, 'verb_log': 0, 'maxiter': 80,
        #     # 'popsize': 8})
        #     params_list.append(es.best.x)
        return idx, params_list

    def optimize(self, process_window: int = 5000, channels: Union[list, int] = None, epochs: Union[list, int] = None):
        """
        Optimizes the artifact remover for the given channels and epochs.

        Parameters:
        -----------
        process_window: int
            The window size for processing.
        channels: Union[list, int]
            The channels to optimize. If None, all channels are optimized.
        epochs: Union[list, int]
            The epochs to optimize. If None, all epochs are optimized.
        """
        data, data_rate = self._get_data_to_optimize(channels, epochs)
        freqs = rfftfreq(data.shape[-1], d=1 / data_rate)
        self.quality.init_shape((1, 1, data.shape[-1]))
        self.process_window = process_window

        fct = partial(
            self.star_emg.perform_window_process,
            notch_filter=False,
            window=process_window,
            return_dict=False,
            data_rate=data_rate,
            freqs=freqs,
        )

        quality_part = partial(
            self.quality.compute_quality, ground_truth=None, channel=0, idx=0, fs=data_rate, kw=100, fft_freqs=freqs
        )

        if self.n_processes == 1 or data.shape[0] == 1:
            return self._optimize_singleproc(fct, quality_part, data)
        else:
            return self._optimize_multiproc(fct, quality_part, data)

    def _optimize_multiproc(self, fct: partial, quality_fct: partial, data: np.ndarray):
        args = [(i, data[i], fct, quality_fct) for i in range(data.shape[0])]
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=self.n_processes) as pool:
            list_results = pool.map(self._optimize_single, args)
        return list_results

    def _optimize_singleproc(self, fct: partial, quality_fct: partial, data: np.ndarray):
        list_results = []
        for i in range(data.shape[0]):
            list_results.append(self._optimize_single(i, data[i], fct, quality_fct, self.process_window))
        return list_results

    def _get_data_to_optimize(self, channels: Union[list, int] = None, epochs: Union[list, int] = None) -> tuple:
        """
        Gets the data to optimize for the given channels and epochs.
        Parameters:
        -----------
        channels: Union[list, int]
            The channels to optimize. If None, all channels are optimized.
        epochs: Union[list, int]
            The epochs to optimize. If None, all epochs are optimized.

        Returns:
        --------
        tuple: (data, data_rate, process_window)
            The data to optimize, the data rate and the process window.
        """
        data = self.star_emg.data_loader.init_data
        j = ensure_list(epochs) if epochs is not None else slice(None)
        k = ensure_list(channels) if channels is not None else slice(None)
        self.init_data_shape = data.shape
        self.star_emg.data_loader.init_data = data[j][:, k, ...]
        self.star_emg.data_loader._apply_stack_epochs()
        # data = self.star_emg.data_loader.flatten_data(data[j, k, ...])
        data_rate = self.star_emg.data_loader.data_rate
        return self.star_emg.data_loader.init_data, data_rate

    def save(self, filename: str):
        pass
