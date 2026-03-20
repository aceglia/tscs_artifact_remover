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
from artifact_remover.processing_utils import filter_data, merge_dict


class ArtefactRemover:
    def __init__(self, data: Union[str, List[str]] = None, **data_loader_kwargs):
        self.ratio = None
        self.transformer = None
        self.is_txt_file = False
        self.is_data_loaded = False

        if data is not None:
            self.load_data(data, data_loader_kwargs)

    def load_data(self, data, data_loader_kwargs):
        self.data_loader = DataLoader(data, **data_loader_kwargs)
        self.is_data_loaded = True

    def process(
        self,
        hankel_size=300,
        threshold=None,
        randomized=False,
        post_filter=False,
        threads=1,
        batch_idxs=None,
        channel_idxs=None,
        data_window=None,
        nb_principal_components=None,
        epsilon=None,
        notch_filter=False,
        quality_factor=150,
        frequency_peaks=30,
        hankel_delay=1,
        process_window=None,
        freq_bounds=[10, 450],
        factor=0.5, 
    ) -> Solution:
        self.solution = Solution(self.data_loader.data_rate)

        print("Processing signals, this might take a while...")
        data = self.data_loader.init_data
        if batch_idxs and not self.data_loader.stack_batch:
            if not isinstance(batch_idxs, list):
                batch_idxs = [batch_idxs]
            data = data[batch_idxs, ...]

        if channel_idxs:
            if not isinstance(channel_idxs, list):
                channel_idxs = [channel_idxs]
            data = data[:, channel_idxs, :]

        if data_window:
            data = data[..., data_window[0] : min(data.shape[-1], data_window[1])]
        process_window = process_window if process_window is not None else data.shape[-1]
        process_window = min(process_window, 10000)

        data = self.data_loader.flatten_data(data)
        data_rate = self.data_loader.data_rate
        list_results = []
        fft_freqs = np.fft.rfftfreq(process_window - (hankel_size - 1) * hankel_delay, 1/data_rate)
        if threads == 1:
            for d in range(data.shape[0]):
                if notch_filter:
                    list_results.append(
                        self._perform_window_notch(
                            frequency_peaks=frequency_peaks,
                            data=data[d],
                            fs=data_rate,
                            quality_factor=quality_factor,
                            window=process_window
                        )
                    )
                else:
                    list_results.append(
                        self._perform_window_decomposition(
                            data[d],
                            hankel_size,
                            randomized,
                            post_filter,
                            nb_principal_components,
                            None,
                            epsilon,
                            True,
                            hankel_delay,
                            True,
                            window=process_window, 
                            data_rate=data_rate, 
                            freq_bounds=freq_bounds,
                            factor=factor, 
                            fft_freqs=fft_freqs,
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
                    epsilon,
                    notch_filter,
                    quality_factor,
                    frequency_peaks,
                    hankel_delay,
                    process_window, 
                    data_rate
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
            randomized,
            post_filter,
            nb_principal_components,
            epsilon,
            notch_filter,
            quality_factor,
            frequency_peaks,
            hankel_delay,
            window, 
            data_rate, 
            freq_bounds,
            factor, 
            fft_freqs    
        ) = args
        if notch_filter:
            return ArtefactRemover()._perform_window_notch(
                frequency_peaks=frequency_peaks,
                data=data,
                fs=2000,
                quality_factor=quality_factor,
                window=window
            )
        return ArtefactRemover()._perform_window_decomposition(
        data,
        hankel_size,
        randomized,
        post_filter,
        nb_principal_components,
        None,
        epsilon,
        True,
        hankel_delay,
        True,
        window=window, 
        data_rate=data_rate, 
        freq_bounds=freq_bounds,
        factor=factor, 
        fft_freqs=fft_freqs,
        )

    @staticmethod
    def _perform_window_decomposition(
        data,
        hankel_size=None,
        randomized=False,
        filter=True,
        nb_principal_components=None,
        n_reconstruct=None,
        epsilon=None,
        offline=True,
        hankel_delay=1,
        return_dict=True,
        window=5000,
        data_rate=None, 
        freq_bounds=[10, 450],
        factor=0.5, 
        fft_freqs=None    
    ):
        if return_dict:
            init_data = data.copy()
        count = 0
        len_d = len(data)
        break_at_end = False
        res = None
        while True:
            overlap = 0
            if count != 0:
                available_wind = min(len_d - count, window)
                if available_wind < window:
                    overlap = window - available_wind
                    data_tmp = np.concatenate((data[count - overlap : count], data[count : count + available_wind]))
                    break_at_end = True
                else:
                    data_tmp = data[count : count + window]
            else:
                data_tmp = data[count : count + window]
            res_tmp = ArtefactRemover()._perform_decomposition(
                data_tmp,
                hankel_size=hankel_size,
                randomized=randomized,
                filter=filter,
                nb_principal_components=nb_principal_components,
                n_reconstruct=n_reconstruct,
                epsilon=epsilon,
                offline=offline,
                hankel_delay=hankel_delay,
                return_dict=return_dict,
                data_rate=data_rate,
                freq_bounds=freq_bounds,
                factor=factor,
                fft_freqs=fft_freqs
            )
            if return_dict:
                if overlap != 0:
                    res_tmp['output'] = res_tmp['output'][overlap:]
                res = merge_dict(res, res_tmp) 
            else:
                res = np.concatenate((res, res_tmp[overlap:])) if res is not None else res_tmp
            
            count += window
            if break_at_end or count >= len_d:
                break

        sig_to_filt = res if not return_dict else res['output'].copy()
        if filter:
            signal_filtered = filter_data(sig_to_filt[None, None, :])[0, 0, :]
            if return_dict:
                res["output"] = signal_filtered
        if return_dict:
            res['data'] = init_data
            res['unfiltered_signal'] = sig_to_filt
        return res

    def _perform_window_notch(self, frequency_peaks, data, fs=2000, quality_factor=150, return_dict=True, window=10000):
        if return_dict:
            init_data = data.copy()
        count = 0
        len_d = len(data)
        break_at_end = False
        res = None
        while True:
            overlap = 0
            if count != 0:
                available_wind = min(len_d - count, window)
                if available_wind < window:
                    overlap = window - available_wind
                    data_tmp = np.concatenate((data[count - overlap : count], data[count : count + available_wind]))
                    break_at_end = True
                else:
                    data_tmp = data[count : count + window]
            else:
                data_tmp = data[count : count + window]
            res_tmp = ArtefactRemover()._perform_notch_filter(
                frequency_peaks=frequency_peaks,
                data=data_tmp,
                fs=fs,
                quality_factor=quality_factor,
            )
            if return_dict:
                if overlap != 0:
                    res_tmp['output'] = res_tmp['output'][overlap:]
                res = merge_dict(res, res_tmp) 
            else:
                res = np.concatenate((res, res_tmp[overlap:])) if res is not None else res_tmp
            
            count += window
            if break_at_end or count >= len_d:
                break

        sig_to_filt = res if not return_dict else res['output'].copy()
        if filter:
            signal_filtered = filter_data(sig_to_filt[None, None, :])[0, 0, :]
            if return_dict:
                res["output"] = signal_filtered
        if return_dict:
            res['data'] = init_data
            res['unfiltered_signal'] = sig_to_filt
        return res

    @staticmethod
    def _perform_decomposition(
        data,
        hankel_size=None,
        randomized=False,
        filter=True,
        nb_principal_components=None,
        n_reconstruct=None,
        epsilon=None,
        offline=True,
        hankel_delay=1,
        return_dict=True,
        data_rate= None, 
        freq_bounds=[10, 450],
        factor=0.5, 
        fft_freqs=None    
    ):
        if return_dict:
            out_dict = {
                "data": data,
            }
        u, s, v, hankel_matrix = compute_svd(
            data,
            n_rows=hankel_size,
            hankel=None,
            randomized=randomized,
            nb_principal_components=nb_principal_components,
            epsilon=epsilon,
            hankel_delay=hankel_delay,
        )

        if return_dict:
            out_dict.update({"s": s.copy()})
        # rem_fct = remove_singular_values_offline if offline else remove_singular_values
        s_reduced, v, u = remove_singular_values(v, s, u, data_rate=data_rate, freq_bounds=freq_bounds, factor=factor, fft_freqs=fft_freqs)

        if n_reconstruct is not None:
            signal_reduced = get_signal_from_hankel(u @ (v[:, -n_reconstruct:] * s_reduced[:, None]), hankel_delay)
        else:
            signal_reduced = get_signal_from_hankel(u @ (v * s_reduced[:, None]), hankel_delay)
        if return_dict:
            out_dict["unfiltered_signal"] = signal_reduced.copy()
        if filter:
            signal_reduced = filter_data(signal_reduced[None, None, :])[0, 0, :]
        if return_dict:
            out_dict["output"] = signal_reduced
            out_dict["u"] = u
            out_dict["v"] = v
            out_dict["s_reduced"] = s_reduced
            return out_dict
        else:
            return signal_reduced

    @staticmethod
    def _perform_notch_filter(frequency_peaks, data, fs=2000, quality_factor=150, return_dict=True):
        if return_dict:
            out_dict = {"data": data.copy()}
        harmonics = [(frequency_peaks * i) for i in range(1, int((fs / 2) / frequency_peaks) + 1)]
        for p, ha in enumerate(harmonics):
            try:
                b, a = scipy.signal.iirnotch(ha, quality_factor, fs=fs)
                filtered_signal = (
                    scipy.signal.filtfilt(b, a, data) if p == 0 else scipy.signal.filtfilt(b, a, filtered_signal)
                )
            except:
                continue

        if return_dict:
            out_dict.update(
                {
                    "unfiltered_signal": filtered_signal,
                    "output": filtered_signal,
                    "u": None,
                    "s": None,
                    "v": None,
                    "s_reduced": None,
                }
            )
            return out_dict
        else:
            return filtered_signal

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
