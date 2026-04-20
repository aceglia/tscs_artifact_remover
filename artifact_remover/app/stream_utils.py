import numpy as np
from multiprocessing import RawArray, RawValue


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
