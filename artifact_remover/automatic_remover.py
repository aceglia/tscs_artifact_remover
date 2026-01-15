from typing import Union, List
import multiprocessing as mp

import numpy as np
import scipy

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
        nb_principal_components=50,
        notch_filter=False,
        quality_factor=150,
        frequency_peaks=30,
    ) -> Solution:
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

        if data.shape[-1] > 10000:
            raise RuntimeError(
                "Data length too large, please split your data into smaller windows or consider using the moving window module."
            )

        data = self.data_loader.flatten_data(data)

        list_results = []
        if threads == 1:
            for d in range(data.shape[0]):
                if notch_filter:
                    list_results.append(
                        self._perform_notch_filter(
                            frequency_peaks=frequency_peaks,
                            data=data[d],
                            fs=self.data_loader.data_rate,
                            quality_factor=quality_factor,
                        )
                    )
                else:
                    list_results.append(
                        self._perform_decomposition(
                            data[d], hankel_size, threshold, randomized, post_filter, nb_principal_components
                        )
                    )
        else:
            args = [
                (
                    data[b],
                    hankel_size,
                    threshold,
                    randomized,
                    post_filter,
                    nb_principal_components,
                    notch_filter,
                    quality_factor,
                    frequency_peaks,
                )
                for b in range(data.shape[0])
            ]
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=threads) as pool:
                list_results = pool.map(self.worker, args)
        fct = self.solution.from_notch_filter if notch_filter else self.solution.from_signal_decomposition
        fct(list_results, initial_data_shape=self.data_loader._data_shape)
        return self.solution

    @staticmethod
    def worker(args):
        (
            data,
            hankel_size,
            threshold,
            randomized,
            post_filter,
            nb_principal_components,
            notch_filter,
            quality_factor,
            frequency_peaks,
        ) = args
        if notch_filter:
            return ArtefactRemover()._perform_notch_filter(
                frequency_peaks=frequency_peaks,
                data=data,
                fs=2000,
                quality_factor=quality_factor,
            )
        return ArtefactRemover()._perform_decomposition(
            data, hankel_size, threshold, randomized, post_filter, nb_principal_components
        )

    @staticmethod
    def _perform_decomposition(
        data, hankel_size=None, threshold=None, randomized=True, filter=True, nb_principal_components=50
    ):
        u, s, v, hankel_matrix = compute_svd(
            data,
            n_rows=hankel_size,
            hankel=None,
            randomized=randomized,
            nb_principal_components=nb_principal_components,
        )
        s_reduced = remove_singular_values(v, s.copy(), threshold=threshold)
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

    @staticmethod
    def _perform_notch_filter(frequency_peaks, data, fs=2000, quality_factor=150):
        filtered_signal = data.copy()
        freqs = np.fft.fftfreq(data.shape[0], 1 / fs)
        harmonics = [(frequency_peaks * i) for i in range(1, int((fs / 2) / frequency_peaks) + 1)]
        for p in harmonics:
            w0 = p
            Q = quality_factor
            b, a = scipy.signal.iirnotch(w0, Q, fs=fs)

            # Apply the filter to the signal
            filtered_signal = scipy.signal.filtfilt(b, a, filtered_signal)

        out_dict = {
            "data": data,
            "unfiltered_signal": filtered_signal,
            "output": filtered_signal,
            "u": None,
            "s": None,
            "v": None,
            "s_reduced": None,
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
