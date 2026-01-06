from typing import Union, List
import multiprocessing as mp

from artifact_remover.io_utils import DataLoader
from artifact_remover.decomposition_utils import (
    compute_svd,
    remove_singular_values,
    get_signal_from_hankel,
)
from artifact_remover.solution import Solution
from artifact_remover.processing_utils import filter_data


class ArtefactRemover:
    def __init__(self, data: Union[str, List[str]] = None, **data_loader_kwargs):
        self.ratio = None
        self.transformer = None
        self.is_txt_file = False
        self.is_data_loaded = False

        self.solution = Solution()
        if data is not None:
            self.load_data(data, data_loader_kwargs)

    def load_data(self, data, data_loader_kwargs):
        self.data_loader = DataLoader(data, **data_loader_kwargs)
        self.is_data_loaded = True

    def process(
        self,
        hankel_size=300,
        threshold=None,
        randomized=True,
        post_filter=True,
        threads=1,
        batch_idxs=None,
        channel_idxs=None,
        data_window=None,
    ):
        print("Processing signals, this might take a while...")
        data = self.data_loader.init_data
        if batch_idxs:
            if not isinstance(batch_idxs, list):
                batch_idxs = [batch_idxs]
            data = data[batch_idxs, ...]
        if channel_idxs:
            if not isinstance(channel_idxs, list):
                channel_idxs = [channel_idxs]
            data = data[:, channel_idxs, :]
        if data_window:
            data = data[..., data_window[0] : data_window[1]]

        data = self.data_loader.flatten_data(data)

        list_results = []
        if threads == 1:
            for d in range(data.shape[0]):
                list_results.append(
                    self._perform_decomposition(data[d], hankel_size, threshold, randomized, post_filter)
                )
        else:
            args = [
                (
                    data[b],
                    hankel_size,
                    threshold,
                    randomized,
                    filter,
                )
                for b in range(data.shape[0])
            ]
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=threads) as pool:
                list_results = pool.map(self.worker, args)

        self.solution.from_signal_decomposition(list_results, initial_data_shape=self.data_loader._data_shape)

    @staticmethod
    def worker(args):
        data, hankel_size, threshold, randomized, filter = args
        return ArtefactRemover()._perform_decomposition(data, hankel_size, threshold, randomized, filter)

    @staticmethod
    def _perform_decomposition(data, hankel_size=None, threshold=None, randomized=True, filter=True):
        u, s, v, hankel_matrix = compute_svd(data, n_rows=hankel_size, hankel=None, randomized=randomized)
        s_reduced = remove_singular_values(v, s, threshold=threshold, n_points=50)
        signal_reduced = get_signal_from_hankel((u * s_reduced) @ v)
        unfiltered_signal = signal_reduced
        if filter:
            signal_reduced = filter_data(signal_reduced[None, None, :])[0, 0, :]
        out_dict = {
            "data": data,
            "unfiltered_signal": unfiltered_signal,
            "output": signal_reduced,
            "u": u,
            "s": s,
            "v": v,
            "s_reduced": s_reduced,
        }
        return out_dict

    def get_process_signal(self, filtered=True):
        return self.solution.get("signal_reduced") if filtered else self.solution.get("unfiltered_signal")

    def get_init_signal(self):
        return self.data_loader.init_data

    def get_singular_values(self, processed=False):
        return self.solution.get(["s"]) if not processed else self.solution.get(["s_reduced"])

    def get_data_rate(self):
        return self.data_loader.data_rate

    def get_channel_names(self):
        return self.data_loader.channel_names
