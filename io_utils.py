import csv
import numpy as np
from biosiglive import load
from processing_utils import filter_data


def handle_init_data(data, center=True, signal_filter=True, **kwargs):
    if center:
        data -= np.mean(data, axis=1)[:, np.newaxis, :]
    if signal_filter:
        cutoff = kwargs.get('cutoff', 450.0)
        fs = kwargs.get('fs', 2000)
        filter_type = kwargs.get('filter_type', 'low')
        order = kwargs.get('order', 5)
        data = filter_data(data, cutoff, order, fs, filter_type)
    return data


def load_txt_file(path, delimiter, center=True, signal_filter=False, **kwargs):
    frames = []
    with open(path, 'r') as file:
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
    frames = [row[:min(all_len)] for row in frames]
    array = np.array(frames)
    chanel_names = frames[0][0][1:]
    array = array[:, 1:, 1:].astype(float)
    if center or signal_filter:
        array = handle_init_data(array, center=center, signal_filter=signal_filter, **kwargs)
    array = array[:, :, :]
    return array, chanel_names


def load_bio_file(path, center=True, signal_filter=True, **kwargs):
    array, frames = None, None
    data = load(path)
    frames = list(data.keys())
    for key in data.keys():
        if isinstance(data[key], np.ndarray):
            array = data[key] if array is None else np.vstack((array, data[key]))
        else:
            pass
    array = array.T[None, ...]
    if center or signal_filter:
        array = handle_init_data(array, center=center, signal_filter=signal_filter, **kwargs)
    return array, frames


