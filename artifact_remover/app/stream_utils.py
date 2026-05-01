import queue

import numpy as np
from multiprocessing import RawArray, RawValue, Queue
import time


class ClearableQueue(Queue):
    def __init__(self, maxsize=1000):
        super().__init__(maxsize=maxsize)
    
    def clear(self):
        while not self.empty():
            try:
                self.get_nowait()
            except Exception:
                break
        
    def put_nowait(self, obj):
        try:
            super().put_nowait(obj)
        except self.Full:
            self.clear()
            super().put_nowait(obj)

    def get_stacked(self):
        data, time, idx = [], [], []
        n_read = 0
        while True:
            try:
                data_tmp, time_tmp, idx_tmp = self.get_nowait()
                data.append(data_tmp)
                time.append(time_tmp)
                idx.append(idx_tmp)
                n_read += 1
            except queue.Empty:
                break
        if n_read == 0:
            return None
        data_stacked = {}
        unique_idx = np.unique(idx)
        for i in unique_idx:
            idx_tmp = np.argwhere(idx == i)
            data_i = np.hstack(np.array(data)[idx_tmp])
            time_i = np.hstack(np.array(time)[idx_tmp])
            data_stacked[i] = (data_i, time_i)
        return data_stacked

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
