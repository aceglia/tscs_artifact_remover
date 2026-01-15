from artifact_remover.streaming_utils import DataStreamer, CircularBuffer
from artifact_remover.automatic_remover import ArtefactRemover
from artifact_remover.solution import Solution
import numpy as np

class RtArtefactRemover(ArtefactRemover):
    def __init__(self, data=None, window_size=2000,  **data_loader_kwargs):
        super().__init__(None, **data_loader_kwargs)
        self.offline = False if data is None else True
        self.streamer = DataStreamer(data=data, offline=self.offline, **data_loader_kwargs)
        self.solution = Solution()
        self.window_size = window_size
        self.buffer = CircularBuffer(self.streamer.init_data.shape[1], window_size)
        self.output = None
        self.idx = 0

    def get_init_signal(self):
        return self.streamer.init_data

    def process_chunck(self, data, **process_kwargs):    
        if not self.buffer.full:
            self.buffer.append(data)
        else:
            self.buffer.append(data)
            self._remove_artifact_from_windows(self.buffer.get(), **process_kwargs)

    def process_all_data(self, chunk_size=None, data_window=None, channel_idxs=None, **process_kwargs):
        if data_window is not None or channel_idxs is not None:
            data = self.get_init_signal()
            if not isinstance(channel_idxs, list):
                channel_idxs = [channel_idxs]
            data = data[:, channel_idxs, :] if channel_idxs is not None else data
            data = data[:, :, data_window[0]:data_window[1]] if data_window is not None else data
            self.streamer.init_data = data
        self.output = np.zeros_like(self.streamer.init_data)
        self.streamer.chunk_size = chunk_size if chunk_size else self.streamer.chunk_size
        import time
        tic = time.time()
        for i in range(self.streamer.num_chunks):
            data_chunk = self.streamer.get_next_chunk(self.streamer.chunk_size)
            self.process_chunck(data_chunk, **process_kwargs)
            self.idx += self.streamer.chunk_size
        print('Total time to process data:', time.time()-tic, 'its around: ', (time.time()-tic) / self.streamer.num_chunks, 'per iteration')
        return self.output

    def process_stream(self, **process_kwargs):
        pass

    def _remove_artifact_from_windows(self,
                                       data,
        hankel_size=300,
        randomized=True,
        nb_principal_components=50,
        notch_filter=False,
        quality_factor=150,
        frequency_peaks=30, **kwargs):
        data = data[0, 0, :]
        if notch_filter:
            output = self._perform_notch_filter(frequency_peaks, data, self.streamer.data_loader.data_rate, quality_factor, return_dict=False)
        else:
            output = self._perform_decomposition(data, hankel_size, None, randomized, False, nb_principal_components, None, return_dict=False,
             n_reconstruct=self.streamer.chunk_size)
        
        if self.offline:
            self.output[:, 0, self.idx:self.idx+self.streamer.chunk_size] = output[-self.streamer.chunk_size:][None, :]
        