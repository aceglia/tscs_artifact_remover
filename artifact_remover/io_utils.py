import csv
import numpy as np
from itertools import zip_longest
from scipy.io import loadmat

from biosiglive import load
from artifact_remover.processing_utils import filter_data


def center_and_filter(data, center=True, signal_filter=True, cutoff=450.0, fs=2000, order=2):
    if center:
        data -= np.nanmean(data, axis=-1, keepdims=True)
    if signal_filter:
        filter_type = "band" if isinstance(cutoff, list) else "low"
        raise_error = False
        if filter_type == "low" and (cutoff / fs) >= 0.5:
            raise_error = True
        elif filter_type == "band" and (cutoff[1] / fs) >= 0.5:
            raise_error = True
        if raise_error:
            raise RuntimeError(
                f"Filter cutoff ({cutoff}) is not suitable for the frequency ({fs}). Try reduce the cutoff."
            )
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


def load_csv_file(path, delimiter="\t"):
    """
    CSV file must contain the channel names in the first row the delimiter should be tab (\t). 
    An optional first column can contain the reccoring times, if so the frame rate will be computed from the difference between two consecutive reccoring times.
    """
    rows = []
    with open(path, newline='') as file:
        reader = csv.reader(file, delimiter=delimiter)
        headers = next(reader)
        for row in reader:
            rows.append(row)
    array = np.array(rows).astype(float).T
    array = array[None]
    channel_names = headers
    frames = 1
    if headers[0] == 'time':
        data_rate = 1 / np.mean(np.diff(array[0, :, 0]))
        channel_names = headers[1:]
        array = array[:, 1:, :]
    array[array == None] = np.nan
    return array, channel_names, frames, data_rate


def ensure_array_dim(array):
    if array.ndim == 1:
        array = array[None, None]
    elif array.ndim == 2 :
        array = array[None, :]
    elif array.ndim == 3:
        pass
    else:
        raise RuntimeError('Shape of the data must be (epochs, n_channels, n_frames) or (n_channels, n_frames) or (n_frames).')
    return array


def load_from_dict(data_dic):
    array = ensure_array_dim(data_dic["values"])
    frames = array.shape[0]
    if "channel_names" in data_dic.keys():
        channel_names = data_dic["channel_names"]
    elif channel_names is None:
        channel_names = [f"chanel_{i}" for i in range(array.shape[1])]
    else:
        channel_names = channel_names
    data_rate = None
    if "data_rate" in data_dic.keys():
        data_rate = data_dic["data_rate"]
    return array, channel_names, frames, data_rate


def load_bio_file(data, channel_names=None):
    data = load(data)
    data['channel_names'] = channel_names
    return load_from_dict(data)


def _load_mat_spike(data_dict):
    channels = [key for key in data_dict.keys() if not key.startswith("__") and key != "file"]
    all_chans = [data_dict[key][0, 0] for key in channels]
    channel_content = list(all_chans[0].dtype.fields.keys())
    chanel_names = [str(chan[channel_content.index("title")][0]) for chan in all_chans]
    data_rate = 1 / float(all_chans[0][channel_content.index("interval")][0][0])
    frames = [0]
    idx_keyboard = [i for i, name in enumerate(chanel_names) if "keyboard" in name.lower()]
    if len(idx_keyboard) != 0:
        idx_keyboard = idx_keyboard[0]
        chanel_names.pop(idx_keyboard)
        all_chans.pop(idx_keyboard)
    list_array = [np.array(chan[channel_content.index("values")]).flatten() for chan in all_chans]
    arr_filled = [list(tpl) for tpl in zip(*zip_longest(*list_array))]
    array = np.vstack(arr_filled)[None, ...]
    # change None with nan
    array[array == None] = np.nan
    return array, chanel_names, frames, data_rate


def _get_chan_names(chaninfo):
    return [str(chaninfo[i][1][0]) for i in range(chaninfo.shape[0])]


def _load_from_wave_data(wave_data):
    items = list(wave_data[0][0].dtype.fields.keys())
    chanel_names = _get_chan_names(wave_data[0][0][items.index("chaninfo")].reshape(-1))
    frames = list(range(wave_data[0][0][items.index("frames")][0][0]))
    data_rate = 1 / wave_data[0][0][items.index("interval")][0][0]
    array = wave_data[0][0][items.index("values")]
    array = np.swapaxes(array, 0, -1)
    return array, chanel_names, frames, data_rate


def load_mat_file(path):
    try:
        mat_file = loadmat(path)
    except:
        raise ValueError("Not able to load the .mat file. Try exporting in version 6 or lower of matlab.")
    if "file" in mat_file.keys():
        file_name = str(mat_file["file"][0, 0][0][0])
        if "smr" in file_name.split(".")[-1]:
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

    def __init__(self, data, stack_batch=False, ignore_filtering=False, **kwargs):
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
        self.time = None
        if self.path is None and self.data is None:
            raise RuntimeError("Data format is not recognized")

        self.get_data_params(**kwargs)
        self.get_filtering_params(**kwargs)
        self.load_data(ignore_filtering=ignore_filtering)
        self.stack_batch = stack_batch
        if stack_batch and self.init_data.shape[0] > 1:
            self._apply_stack_batch()

    def _apply_stack_batch(self):
        self._unstack_shape = self.init_data.shape
        self.init_data = np.swapaxes(self.init_data, 0, 1)
        self.init_data = self.init_data.reshape(self.init_data.shape[0], -1)[None, ...]

    def get_unstacked_data(self):
        if hasattr(self, "_unstack_shape"):
            data = self.init_data[0, ...].reshape(
                self._unstack_shape[1], self._unstack_shape[0], self._unstack_shape[2]
            )
            data = np.swapaxes(data, 0, 1)
            return data
        else:
            return self.init_data

    def get_filtering_params(self, **kwargs):
        filtering_params = ["cutoff", "order", "center", "signal_filter"]
        default = [450.0, 2, True, True]
        self.filtering_params = {}
        for k, key in enumerate(filtering_params):
            self.filtering_params[key] = kwargs.get(key, default[k])
        self.__dict__.update(self.filtering_params)

    def get_data_params(self, **kwargs):
        data_params = ["delimiter", "channel_names", "data_rate", "data_window"]
        default = ["\t", None, None, None]
        self.loading_params = {}
        for k, key in enumerate(data_params):
            self.loading_params[key] = kwargs.get(key, default[k])
        self.__dict__.update(self.loading_params)

    def _load_files(self):
        if self.path.endswith(".txt"):
            self.data, self.channel_names, self.frames, self.data_rate = load_txt_file(self.path, self.delimiter)
        elif self.path.endswith(".bio"):
            self.data, self.channel_names, self.frames, self.data_rate = load_bio_file(self.path, self.channel_names)
        elif self.path.endswith(".mat"):
            self.data, self.channel_names, self.frames, self.data_rate = load_mat_file(self.path)
        elif self.path.endswith(".csv"):
            self.data, self.channel_names, self.frames, self.data_rate = load_csv_file(self.path, self.delimiter)
        else:
            raise ValueError("File format not supported")
        if self.data_rate is None and self.loading_params["data_rate"] is not None: 
            self.data_rate = self.loading_params["data_rate"]
        elif self.data_rate is None:
            raise ValueError("Data rate must be provided if not loaded from file")

    def load_data(self, ignore_filtering=False):
        if self.path is not None:
            self._load_files()
        if self.data_rate is None:
            raise ValueError("Data rate must be provided if not loaded from file")
        self.init_data = self.apply_filtering() if not ignore_filtering else self.data
        self.time = np.repeat(
            (np.arange(0, self.init_data.shape[-1]) / self.data_rate)[None], self.init_data.shape[0], axis=0
        )
        self.is_data_loaded = True

    def apply_filtering(self, data=None):
        to_filter = data if data is not None else self.data
        filtered = center_and_filter(
            to_filter,
            center=self.center,
            signal_filter=self.signal_filter,
            cutoff=self.cutoff,
            fs=self.data_rate,
            order=self.order,
        )
        return filtered

    def flatten_data(self, data):
        self._data_shape = data.shape
        return data.reshape(-1, data.shape[-1])

    def unflatten_data(self, data, data_shape=None):
        data_shape = data_shape or self._data_shape
        return data.reshape(data_shape)


def export_csv(path, raw_data, processed_data_svd, processed_data_notch, channels, rate):
    final_mat = None
    for data in [raw_data, processed_data_notch, processed_data_svd]:
        data_stacked = data.transpose(1, 0, -1)
        data_stacked = data_stacked[:, :, :-1].reshape(raw_data.shape[1], -1)
        final_mat = data_stacked if final_mat is None else np.vstack((final_mat, data_stacked))
    channels.extend(sum([], [c + "_notch_processed" for c in channels] + [c + "_svd_processed" for c in channels]))
    comments = f"Data rate: {rate} | {raw_data.shape[0]} frames | {raw_data.shape[-1]} samples per frame\n"
    headers = channels
    data_to_write = final_mat.T
    np.savetxt(
        path.replace(".mat", ".txt"),
        data_to_write,
        header=comments + "\t".join(headers),
        delimiter="\t",
        comments="",  # removes the default '#' before header
    )
