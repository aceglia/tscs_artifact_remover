import csv
import numpy as np
from scipy.io import loadmat

from biosiglive import load
from artifact_remover.processing_utils import filter_data


def handle_init_data(data, center=True, signal_filter=True, cutoff=450.0, fs=2000, order=2):
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
    frames = [row[1 : min(all_len)] for row in frames]
    array = np.array(frames).astype(float)
    array = np.swapaxes(array, 1, 2)
    data_rate = 1 / np.mean(np.diff(array[:, 0, :], axis=-1))
    array = array[:, 1:, :]
    frames = np.arange(0, len(frames))
    return array, channel_names, frames, data_rate

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
    return array, channel_names, frames, None


def _load_mat_spike(data_dict):
    channels = [key for key in data_dict.keys() if not key.startswith("__") and key != "file"]
    all_chans = [data_dict[key][0, 0] for key in channels]
    channel_content = list(all_chans[0].dtype.fields.keys())
    chanel_names = [str(chan[channel_content.index("title")][0]) for chan in all_chans]
    data_rate = 1 / float(all_chans[0][channel_content.index("interval")][0][0])
    frames = [0]
    array = np.hstack([chan[channel_content.index("values")] for chan in all_chans]).T[None, ...]
    return array, chanel_names, frames, data_rate


def _get_chan_names(chaninfo):
    return [str(chaninfo[i][1][0]) for i in range(chaninfo.shape[0])]


def _load_from_wave_data(wave_data):
    items = list(wave_data[0][0].dtype.fields.keys())
    chanel_names = _get_chan_names(wave_data[0][0][items.index("chaninfo")].reshape(-1))
    frames = list(range(wave_data[0][0][items.index("frames")][0][0]))
    array = wave_data[0][0][items.index("values")]
    array = np.swapaxes(array, 0, -1)
    return array, chanel_names, frames


def load_mat_file(path):
    try:
        mat_file = loadmat(path)
    except:
        raise ValueError("Not able to load the .mat file. Try exporting in version 6 or lower of matlab.")
    if "file" in mat_file.keys():
        file_name = str(mat_file["file"][0, 0][0][0])
        if file_name.endswith(".smrx"):
            return _load_mat_spike(mat_file)
        
    wave_data = [key for key in mat_file.keys() if "wave_data" in key][0]

    if len(wave_data) > 0:
        return _load_from_wave_data(mat_file[wave_data])
    
    else:
        raise ValueError("No recognized data found in the .mat file.")

class DataLoader:
    """
    Docstring for DataLoader
    """

    def __init__(self, data, stack_batch=False, **kwargs):
        """
        Docstring for __init__

        :param self: Description
        :param data: Description
        :param kwargs: Description
            kwargs might contains filters kwargs and loader kwargs as follows:
            delimiter, center, signal_filter, cutoff, fs, order, from_ced_signal, channel_names, data_rate, data_window
        """
        self.path = data if isinstance(data, str) else None
        self.data = data if isinstance(data, np.ndarray) else None
        if self.path is None and self.data is None:
            raise RuntimeError("Data format is not recognized")

        self.get_data_params(**kwargs)
        self.get_filtering_params(**kwargs)
        self.load_data()

        if stack_batch and self.init_data.shape[0] > 1:
            self._apply_stack_batch()

    def _apply_stack_batch(self):
        self._unstack_shape = self.init_data.shape
        self.init_data = np.swapaxes(self.init_data, 0, 1)
        self.init_data = self.init_data.reshape(self.init_data.shape[0], -1)[None, ...]
    
    def get_unstacked_data(self):
        if hasattr(self, "_unstack_shape"):
            data = self.init_data[0, ...].reshape(self._unstack_shape[1], self._unstack_shape[0], self._unstack_shape[2])
            data = np.swapaxes(data, 0, 1)
            return data
        else:
            return self.init_data

    def get_filtering_params(self, **kwargs):
        filtering_params = ["cutoff", "order", "center", "signal_filter"]
        default = [450.0, 2, True, True]
        for k, key in enumerate(filtering_params):
            self.__dict__[key] = kwargs.get(key, default[k])

    def get_data_params(self, **kwargs):
        data_params = ["delimiter", "channel_names", "data_rate", 'data_window']
        default = ["\t", None, 2000, None]
        for k, key in enumerate(data_params):
            self.__dict__[key] = kwargs.get(key, default[k])

    def _load_files(self):
        if self.path.endswith(".txt"):
            self.data, self.channel_names, self.frames, self.data_rate = load_txt_file(self.path, self.delimiter)
        elif self.path.endswith(".bio"):
            self.data, self.channel_names, self.frames, self.data_rate = load_bio_file(self.path, self.channel_names)
        elif self.path.endswith(".mat"):
            self.data, self.channel_names, self.frames, self.data_rate = load_mat_file(self.path)
        else:
            raise ValueError("File format not supported")

    def load_data(self):
        if self.path is not None:
            self._load_files()
        if self.data_rate is None:
            raise ValueError("Data rate must be provided if not loaded from file")
        self.init_data = handle_init_data(
            self.data,
            center=self.center,
            signal_filter=self.signal_filter,
            cutoff=self.cutoff,
            fs=self.data_rate,
            order=self.order,
        )
        self.is_data_loaded = True

    def flatten_data(self, data):
        self._data_shape = data.shape
        return data.reshape(-1, data.shape[-1])

    def unflatten_data(self, data, data_shape=None):
        data_shape = data_shape or self._data_shape
        return data.reshape(data_shape)
