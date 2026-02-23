import numpy as np
from artifact_remover.io_utils import DataLoader


import numpy as np


class CircularBuffer:
    def __init__(self, n, W, n_batch=1, dtype=np.float32):
        self.W = W
        self.ring = np.zeros((n_batch, n, W), dtype=dtype)
        self.linear = np.zeros((n_batch, n, W), dtype=dtype)
        self.idx = 0
        self.full = False

    @property
    def shape(self):
        return self.linear.shape

    def append(self, x):
        """
        x shape: (n_batch, n, w)
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

import numpy as np
from collections import deque

class RealTimeHankel:
    def __init__(self, n_rows, n_cols, dtype=np.float64):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.hankel = np.zeros((n_rows, n_cols), dtype=dtype)
        self.buffer = CircularBuffer(n_cols, )
        
    def update(self, new_sample):
        self.buffer.append(new_sample)
        if len(self.buffer) >= self.n_rows + self.n_cols - 1:
            # Shift matrix left and update last column
            self.hankel[:, :-1] = self.hankel[:, 1:]
            # Fill last column with new data
            start_idx = len(self.buffer) - self.n_rows
            self.hankel[:, -1] = list(self.buffer)[start_idx:]
            
    def get_matrix(self):
        return self.hankel.copy()


class DataStreamer:
    def __init__(self, data=None, offline=True, chunk_size=None, **data_loader_kwargs):
        self.current_index = 0
        self.offline = offline
        self.chunk_size = chunk_size
        if self.offline and not data:
            raise ValueError("Offline mode requires initial data.")
        if data is not None:
            self.load_data(data, data_loader_kwargs)

    def load_data(self, data, data_loader_kwargs):
        self.data_loader = DataLoader(data, stack_batch=True, **data_loader_kwargs)
        self.init_data = self.data_loader.init_data
        self.is_data_loaded = True

    @property
    def num_chunks(self):
        return np.ceil(self.init_data.shape[-1] / self.chunk_size).astype(int)
    
    @property
    def data_rate(self):
        return self.data_loader.data_rate

    def get_next_chunk(self, chunk_size):
        if self.init_data is None:
            raise ValueError("No data loaded.")

        start_index = self.current_index
        if self.current_index + chunk_size > self.init_data.shape[-1]:
            return False, None
        
        end_index = min(self.current_index + chunk_size, self.init_data.shape[-1])
        chunk = self.init_data[..., start_index:end_index]
        if chunk.shape[-1] < chunk_size:
            chunk = np.concatenate([chunk, np.ones((chunk.shape[0], chunk_size - chunk.shape[-1])) * np.nan], axis=-1)
        self.current_index = end_index
        # self.current_index = 0 if end_index == self.init_data.shape[-1] else self.current_index + chunk_size
        # self.current_index = end_index % self.init_data.shape[-1]  # Wrap around if needed
        return True, chunk
