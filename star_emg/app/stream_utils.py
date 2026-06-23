import numpy as np
from multiprocessing import RawArray, RawValue, Queue
import time


class CustomQueue:
    """
    A custom queue class that extends the built-in Queue class and adds additional functionality.
    """

    def __init__(self, name: str = None):
        self.queue = Queue()
        self.name = name

    def clear(self):
        """
        Clear the queue.
        """
        while True:
            try:
                self.queue.get_nowait()
            except Exception:
                break

    def get(self, timeout: float = None):
        """
        Get an item from the queue, with an optional timeout.
        """
        return self.queue.get(timeout=timeout)

    def put_nowait(self, obj):
        """
        Put an item into the queue without blocking.
        """
        try:
            self.queue.put_nowait(obj)
        except Exception:
            pass

    def get_stacked(self) -> dict | None:
        """
        Get all items from the queue and stack them by channel. This is used to get the data for all channels at once for plotting.

        Returns:
            dict | None: A dictionary where the keys are the channel indices and the values are tuples containing the data and timestamps for that channel.
        """
        data_chunks = []
        time_chunks = []
        idx_chunks = []
        while True:
            try:
                d, t, idx = self.queue.get_nowait()
                data_chunks.append(d)
                time_chunks.append(t)
                idx_chunks.append(idx)
            except Exception:
                break
        if not data_chunks:
            return None
        data_all = np.concatenate(data_chunks, axis=-1)
        t_all = np.concatenate(time_chunks)
        idx_all = idx_chunks[0]
        out = {}
        for ch in np.unique(idx_all):
            mask = idx_all == ch
            d_ch = data_all[mask]
            d_ch = d_ch[None] if d_ch.ndim == 1 else d_ch
            out[ch] = (d_ch, t_all)
        return out

    def get_nowait(self):
        try:
            data = self.queue.get_nowait()
            return data
        except Exception:
            return None


def dispatch_queue(results):
    data, t, total_samples, idx = results
    return data, t, total_samples, idx


def get_config_by_idx(configs, idx):
    return configs[idx]


def empty_queue(queue, timeout=0.01):
    all_data = []
    tic = time.perf_counter()
    while time.perf_counter() - tic < timeout:
        try:
            all_data.append(queue.get_nowait())
        except Exception:
            break
    return all_data


class SharedArray:
    def __init__(self, size, dtype=np.float64):
        self.size = size
        self.raw_array = RawArray(dtype, (size[0] + 1, size[1] * 5))  # increase the size for safety if needed

        self.array = np.frombuffer(self.raw_array, dtype=dtype)

        self.last_len = RawValue("i", 0)
        self.version = RawValue("i", 0)  # for consistency

    def set(self, data: np.ndarray, t: np.ndarray = None, idx: int = None):
        n = len(data)
        self.version.value = 1
        if idx == 0:
            raise ValueError("idx cannot be 0 as it is reserved for timestamps.")

        self.array[0, :n] = t if t is not None else 0
        if idx is None:
            self.array[1:, :n] = data
        else:
            self.array[idx, :n] = data
            self.array[idx, n:] = 0

        self.last_len.value = n
        self.version.value = 0

    def get(self, idx: int = None):
        while True:
            v1 = self.version.value
            if v1 == 1:
                continue
            n = self.last_len.value
            if idx is None:
                data = self.array[1:, :n].copy()
            else:
                data = self.array[idx, :n].copy()
            return data, self.array[0, :n].copy()

    @classmethod
    def from_existing(cls, raw_array, last_len, version, shape, dtype=np.float64):
        obj = cls.__new__(cls)

        obj.raw_array = raw_array
        obj.array = np.frombuffer(raw_array, dtype=dtype).reshape(shape)

        obj.last_len = last_len
        obj.version = version

        return obj

    def export(self):
        return (self.raw_array, self.last_len, self.version, self.size, self.array.dtype)
