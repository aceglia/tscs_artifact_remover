import numpy as np
from artifact_remover.io_utils import DataLoader


import numpy as np


class CircularBuffer:
    def __init__(self, n, W, dtype=np.float32):
        self.W = W
        self.ring = np.zeros((1, n, W), dtype=dtype)
        self.linear = np.zeros((1, n, W), dtype=dtype)
        self.idx = 0
        self.full = False

    def append(self, x):
        """
        x shape: (1, n, w)
        """
        w = x.shape[-1]
        end = self.idx + w

        if end <= self.W:
            self.ring[:, :, self.idx:end] = x
        else:
            first = self.W - self.idx
            self.ring[:, :, self.idx:] = x[:, :, :first]
            self.ring[:, :, :w - first] = x[:, :, first:]

        self.idx = end % self.W
        self.full |= end >= self.W

    def get(self):
        if self.full:
            k = self.idx
            self.linear[:, :, :self.W-k] = self.ring[:, :, k:]
            self.linear[:, :, self.W-k:] = self.ring[:, :, :k]
        else:
            self.linear[:, :, :self.idx] = self.ring[:, :, :self.idx]
        return self.linear




class DataStreamer:
    def __init__(self, data=None, offline=True, chunk_size=None, **data_loader_kwargs):
        self.current_index = 0
        self.num_chunks = None
        self.offline = offline
        self.chunk_size = chunk_size
        if self.offline and not data:
            raise ValueError("Offline mode requires initial data.")
        if data is not None:
            self.load_data(data, data_loader_kwargs)

    def load_data(self, data, data_loader_kwargs):
        self.data_loader = DataLoader(data, stack_batch=True, **data_loader_kwargs)
        self.num_chunks = np.ceil(self.data_loader.init_data.shape[-1] / self.chunk_size).astype(int)
        self.init_data = self.data_loader.init_data
        self.is_data_loaded = True

    def get_next_chunk(self, chunk_size):
        if self.init_data is None:
            raise ValueError("No data loaded.")

        start_index = self.current_index
        end_index = min(self.current_index + chunk_size, self.init_data.shape[-1])
        chunk = self.init_data[..., start_index:end_index]
        if chunk.shape[-1] < chunk_size:
            chunk = np.concatenate([chunk, np.ones((chunk.shape[0], chunk_size - chunk.shape[-1])) * np.nan], axis=-1)
        # self.current_index = 0 if end_index == self.init_data.shape[-1] else self.current_index + chunk_size
        self.current_index = end_index % self.data.shape[-1]  # Wrap around if needed
        return chunk
