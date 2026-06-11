import numpy as np
from artifact_remover.io_utils import DataLoader


import numpy as np


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
        self.data_loader = DataLoader(data,ignore_filtering=True, **data_loader_kwargs)
        self.data_loader._apply_stack_batch()
        # self.init_data = self.data_loader.init_data
        self.is_data_loaded = True

    @property
    def init_data(self):
        if not self.is_data_loaded:
            raise ValueError("Data not loaded.")
        return self.data_loader.init_data

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
