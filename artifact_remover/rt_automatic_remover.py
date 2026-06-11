from functools import partial

from artifact_remover.streaming_utils import DataStreamer
from biosiglive.streaming.utils import CircularBuffer
from artifact_remover.automatic_remover import ArtefactRemover
from artifact_remover.solution import Solution
from artifact_remover.processing_utils import Quality
import numpy as np

class RtArtefactRemover(ArtefactRemover):
    def __init__(self, data=None, window_size=2000, **data_loader_kwargs):
        super().__init__(None, **data_loader_kwargs)
        self.offline = False if data is None else True
        self.window_size = window_size
        self.output = None
        self.evaluation_wind = 1000

        self.to_evaluate_buffer = CircularBuffer(2, int(self.evaluation_wind))
        self.buffer = CircularBuffer(1, window_size)

        self.last_rejected = None
        if data is not None:
            self.streamer = DataStreamer(data=data, offline=self.offline, **data_loader_kwargs)

        self.solution = Solution()

        self.idx = 0
        self.quality = Quality(shape=(1, 1, self.evaluation_wind))

    def get_init_signal(self):
        return self.streamer.init_data

    def process_chunck(self, data, **process_kwargs):
        if not self.buffer.full:
            self.buffer.append(data)
            self.to_evaluate_buffer.append(np.hstack([data, np.zeros_like(data)]))
        else:
            self.buffer.append(data)
            data, _ = self.buffer.get()
            data_tmp = self.streamer.data_loader.apply_filtering(data, offline=False)
            if process_kwargs["notch_filter"]:
                process_kwargs["offline"] = False
            process_kwargs["rejected_idx"] = None if self.idx % self.update_svd_every == 0 else self.last_rejected
            output = self._remove_artifact_from_windows(data_tmp, **process_kwargs)
            # if process_kwargs["notch_filter"]:
                # output = output[0]
            output = output[0]
            # self.to_evaluate_buffer.append(np.hstack([data, output[None, None]]))
            if self.offline:
                self.output[:, self.idx : self.idx + self.streamer.chunk_size] = output[-self.streamer.chunk_size :][
                    None, :
                ]
            # if self.to_evaluate_buffer.full:
            #     self._stream_evaluation(self.to_evaluate_buffer.get())

    def _stream_evaluation(self, data):
        data_weighted = np.concatenate([data[:, :, :-100], data[:, :, -100:] *10], axis=-1)
        self.quality_fct(raw=None, processed=data[0, 1], analysis=[0, 1, 2, 3])
        quality = self.quality_fct(raw=data[0, 0], processed=None, analysis=[0, 1, 2, 3])
        data_fft = np.abs(rfft(data_weighted[0, 1]))
        self.quality_fct(raw=data_fft, processed=None, analysis=[1])
        from artifact_remover.processing_utils import line_length, kurtosis_value, robust_max_percentile, median_frequency
        np.cumsum(np.abs(rfft(data[0, 0])))[-1]
        np.cumsum(np.abs(rfft(data[0, 1])))[-1]
        np.sum(np.abs(np.diff(data[0, 1])), axis=-1)  
        line_length(data[0, 0])
        line_length(data[0, 1])
        spike_score(data[0, 0], 10)
        spike_score(data[0, 1], 10)
        def spike_score(x, k=20):
            threshold = np.percentile(np.abs(x), 95)
            score = np.mean(np.abs(x)[np.abs(x) > threshold])
            # topk = np.mean(ax[np.argpartition(ax, -k)[-k:]]) * k
            return score
        kurtosis_value(data[0, 0], 50)
        kurtosis_value(data[0, 1], 50)
        robust_max_percentile(np.abs(rfft(data[0, 0]))[rfftfreq(600, 1 / self.streamer.data_loader.data_rate) < 150], 99.2)
        robust_max_percentile(np.abs(rfft(data[0, 1]))[rfftfreq(600, 1 / self.streamer.data_loader.data_rate) < 150], 99.9)
        median_frequency(data[0, 0], self.streamer.data_loader.data_rate)
        median_frequency(data[0, 1], self.streamer.data_loader.data_rate)
        
        line_length_fft = line_length(np.abs(rfft(data_weighted[0, 0])))
        from artifact_remover.processing_utils import rfft, rfftfreq
        import matplotlib.pyplot as plt
        plt.plot(rfftfreq(600, 1 / self.streamer.data_loader.data_rate), np.abs(rfft(data_weighted[0, 0, :])))
        plt.plot(rfftfreq(600, 1 / self.streamer.data_loader.data_rate), np.abs(rfft(data_weighted[0, 1, :])))
        # plt.plot(data_weighted[0, 0])
        # plt.plot(data_weighted[0, 1])
        plt.show(block=True)

    def process_all_data(
        self, chunk_size=None, data_window=None, channel_idxs=None, update_svd_every=1, **process_kwargs
    ):
        if data_window is not None or channel_idxs is not None:
            data = self.get_init_signal()
            if not isinstance(channel_idxs, list):
                channel_idxs = [channel_idxs]
            data = data[channel_idxs, :] if channel_idxs is not None else data
            data = data[:, data_window[0] : data_window[1]] if data_window is not None else data
            self.streamer.data_loader.init_data = data
            # self.streamer.init_data = data
        self.output = np.zeros_like(self.streamer.init_data)
        self.streamer.chunk_size = chunk_size if chunk_size else self.streamer.chunk_size
        import time

        self.update_svd_every = update_svd_every

        if not process_kwargs["notch_filter"]:
            fft_freqs = np.fft.rfftfreq(
                self.window_size - (process_kwargs["hankel_size"] - 1) * process_kwargs.get("hankel_delay", 1),
                1 / self.streamer.data_loader.data_rate,
            )
        else:
            fft_freqs = None
        process_kwargs["fft_freqs"] = fft_freqs
        process_kwargs["offline"] = False
        self.quality_fct = partial(
            self.quality.compute_quality,
            ground_truth=None,
            channel=0,
            idx=0,
            fs=self.streamer.data_loader.data_rate,
            kw=20,
            fft_freqs=fft_freqs,
        )
        tic = time.time()
        count = 0
        for i in range(self.streamer.num_chunks):
            _, data_chunk = self.streamer.get_next_chunk(self.streamer.chunk_size)
            if data_chunk is None:
                break
            self.process_chunck(data_chunk, **process_kwargs)
            self.idx += self.streamer.chunk_size
            count += 1

        print(
            "Total time to process data:",
            time.time() - tic,
            "its around: ",
            np.round(((time.time() - tic) / count) * 1000, 2),
            "ms",
            np.round(1 / ((time.time() - tic) / count), 2),
            "FPS",
        )
        return self.output

    def process_stream(self, **process_kwargs):
        pass

    def _remove_artifact_from_windows(
        self,
        data,
        hankel_size=None,
        randomized=False,
        nb_principal_components=None,
        epsilon=None,
        notch_filter=False,
        quality_factor=30,
        frequency_peaks=30,
        first_peak=30,
        hankel_delay=1,
        freq_bounds=[10, 450],
        factor=0.5,
        fft_freqs=None,
        rejected_idx=None,
        data_rate=None,
        **kwargs
    ):
        data_rate = data_rate if data_rate is not None else self.streamer.data_loader.data_rate
        data = data.flatten()
        if notch_filter:
            output = self._perform_notch_filter(
                frequency_peaks,
                data,
                data_rate,
                quality_factor,
                return_dict=False,
                first_peak=first_peak,
                offline=False,
            )
        else:
            output = self._perform_decomposition(
                data,
                hankel_size,
                randomized,
                False,
                nb_principal_components,
                None,
                epsilon,
                False,
                hankel_delay,
                False,
                data_rate=data_rate,
                freq_bounds=freq_bounds,
                factor=factor,
                fft_freqs=fft_freqs,
                rejected_idx=rejected_idx,
            )
            self.last_rejected = rejected_idx
        return output
        # if self.offline:
        #     self.output[:, 0, self.idx : self.idx + self.streamer.chunk_size] = output[-self.streamer.chunk_size :][
        #         None, :
        #     ]
        #     return self.output[:, 0, self.idx : self.idx + self.streamer.chunk_size]
