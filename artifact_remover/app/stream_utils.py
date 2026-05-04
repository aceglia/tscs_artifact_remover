import queue

import numpy as np
from multiprocessing import RawArray, RawValue, Queue
import time


class ClearableQueue:
    def __init__(self, maxwrite=1000, name=None):
        self.queue = Queue()
        self.maxwrite = maxwrite
        self.total_written = 0
        self.name = name
    
    def clear(self):
        print(f"Queue {self.name} reached maxwrite limit, clearing the queue.")
        while True:
            try:
                self.queue.get_nowait()
            except Exception:
                break
        self.total_written = 0
        
    def put_nowait(self, obj):
        # if self.total_written >= self.maxwrite:
        #     self.clear()
        try:
            # if self.name == 'plot':
            #     print(f"Putting data into queue {self.name}, data shape: {obj[0].shape} for channel {obj[2]}")
            self.queue.put_nowait(obj)
            # self.total_written += 1
        except Exception:
            pass
        # print(f"Queue {self.name} size: {self.total_written}")

    def get_stacked(self):
        data_chunks = []
        time_chunks = []
        idx_chunks = []
        while True:
            try:
                d, t, idx = self.queue.get_nowait()
                self.total_written -= 1
                # if self.name == 'plot':
                    # print(f"Getting data from queue {self.name}, data shape: {d.shape} for channel {idx}")
                data_chunks.append(d)  
                time_chunks.append(t) 
                idx_chunks.append(idx)
            except Exception:
                break
        if not data_chunks:
            return None
        # print(f"Queue {self.name} size: {self.total_written}")
        # self.total_written = max(self.total_written, 0)
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
            self.total_written -= 1
            self.total_written = max(0, self.total_written)
            return data
        except Exception:
            return None

def dispatch_queue(results):
    data, t, total_samples, idx = results
    return data, t, total_samples, idx


def get_config_by_idx(configs, idx):
    return


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

    def set(self, data: np.ndarray, t: np.ndarray = None, idx: int =None):
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
