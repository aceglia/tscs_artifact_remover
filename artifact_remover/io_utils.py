import csv
import numpy as np

from biosiglive import load
from artifact_remover.processing_utils import filter_data


def handle_init_data(
    data, center=True, signal_filter=True, cutoff=450.0, fs=2000, order=2
):
    if center:
        data -= np.mean(data, axis=-1, keepdims=True)
    if signal_filter:
        filter_type = "band" if isinstance(cutoff, list) else "low"
        data = filter_data(data, cutoff, order, fs, filter_type)
    return data


def load_txt_file(path, delimiter="\t"):
    frames = []
    with open(path, "r") as file:
        reader = csv.reader(file, delimiter=delimiter)
        rows = []
        len_row = -1
        for row in reader:
            len_row = len(row) if len_row == -1 else len_row
            if len(row) != len_row:
                frames.append(rows)
                rows = []
                continue
            rows.append(row)
    all_len = [len(row) for row in frames]
    channel_names = frames[0][:1][0][1:]
    frames = [row[1: min(all_len)] for row in frames]
    array = np.array(frames).astype(float)
    array = np.swapaxes(array, 1, 2)
    array = array[:, 1:, :]
    frames = np.arange(0, len(frames))
    return array, channel_names, frames


def load_bio_file(data, channel_names=None):
    array, frames = None, None
    data = load(data)
    frames = list(data.keys())
    for key in data.keys():
        if isinstance(data[key], np.ndarray):
            array = data[key] if array is None else np.vstack((array, data[key]))
        else:
            pass
    array = array.T[None, ...]
    if channel_names is None:
        channel_names = [f"chanel_{i}" for i in range(array.shape[-1])]
    return array, channel_names, frames


class DataLoader:
    """
    Docstring for DataLoader
    """

    def __init__(self, data, **kwargs):
        """
        Docstring for __init__

        :param self: Description
        :param data: Description
        :param kwargs: Description
            kwargs might contains filters kwargs and loader kwargs as follows:
            delimiter, center, signal_filter, cutoff, fs, order, from_ced_signal, channel_names, data_rate
        """
        self.path = data if isinstance(data, str) else None
        self.data = data if isinstance(data, np.ndarray) else None
        if self.path is None and self.data is None:
            raise RuntimeError("Data format is not recognized")

        self.get_data_params(**kwargs)
        self.get_filtering_params(**kwargs)
        self.load_data()

    def get_filtering_params(self, **kwargs):
        filtering_params = ["cutoff", "fs", "order", "center", "signal_filter"]
        default = [450.0, 2000, 2, True, True]
        for k, key in enumerate(filtering_params):
            self.__dict__[key] = kwargs.get(key, default[k])

    def get_data_params(self, **kwargs):
        data_params = ["from_ced_signal", "delimiter", "channel_names", "data_rate"]
        default = [False, "\t", None, 2000]
        for k, key in enumerate(data_params):
            self.__dict__[key] = kwargs.get(key, default[k])

    def _load_files(self):
        if self.path.endswith(".txt"):
            self.data, self.channel_names, self.frames = load_txt_file(
                self.path, self.delimiter
            )
        elif self.path.endswith(".bio"):
            self.data, self.channel_names, self.frames = load_bio_file(
                self.path, self.channel_names
            )
        else:
            raise ValueError("File format not supported")

    def load_data(self):
        if self.path is not None:
            self._load_files()
        if self.signal_filter and self.fs != self.data_rate:
            print(
                "WARNING: Data rate and filter frequency are different, filtering data with filter frequency"
            )

        self.init_data = handle_init_data(
            self.data,
            center=self.center,
            signal_filter=self.signal_filter,
            cutoff=self.cutoff,
            fs=self.fs,
            order=self.order,
        )
        self.is_data_loaded = True

    def flatten_data(self, data):
        self._data_shape = data.shape
        return data.reshape(-1, data.shape[-1])
    
    def unflatten_data(self, data, data_shape=None):
        data_shape = data_shape or self._data_shape
        return data.reshape(data_shape)


