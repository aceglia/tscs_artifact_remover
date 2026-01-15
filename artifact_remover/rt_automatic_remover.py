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
        self._process_data_buffer = np.empty((1, self.streamer.init_data.shape[1], self.window_size))
        self._buffer_full = False
        self.buffer = CircularBuffer(self.streamer.init_data.shape[1], window_size)
        self.output = np.zeros_like(self.streamer.init_data) if data is not None else None

    def process_chunck(self, data, **process_kwargs):    
        if not self.buffer.full:
            self.buffer.append(data)
        else:
            self._remove_artifact_from_windows(self.buffer.get(), **process_kwargs)

    def process_all_data(self, **process_kwargs):
        for i in range(self.streamer.num_chunks):
            data_chunk = self.streamer.get_next_chunk(self.streamer.chunk_size)
            self.process_chunck(data_chunk, **process_kwargs)

    def _remove_artifact_from_windows(self,
                                       data,
        hankel_size=300,
        randomized=True,
        channel_idxs=None,
        nb_principal_components=50,
        notch_filter=False,
        quality_factor=150,
        frequency_peaks=30):
        pass
        