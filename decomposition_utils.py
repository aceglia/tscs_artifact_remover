import numpy as np
import scipy
import matplotlib.pyplot as plt

def compute_svd(emg_signal, n_rows=800, hankel=None):
    if hankel is None or n_rows != hankel.shape[0]:
        # hankel = np.column_stack([emg_signal[i: i + n_rows] for i in range(len(emg_signal) - n_rows)])
        hankel = scipy.linalg.hankel(emg_signal[:int(n_rows)], emg_signal[int(n_rows - 1):])
    U, S, Vh = scipy.linalg.svd(hankel, full_matrices=False,
                                check_finite=False,
                                overwrite_a=True)
    return U, S, Vh, hankel

def remove_singular_values(v, s, threshold=2, n_points=50):
    # s_reduced = s.copy()
    # diff = s[:-1] - s[1:]
    # threshold = 1
    # idxs = np.argwhere(diff[:n_points] > diff[:n_points].mean() + threshold * diff[:n_points].std())
    # if idxs.shape[0] > 0:
    #     max_idx = int(idxs.max()) + 1
    #     s_reduced[:max_idx] = 0
    s_reduced = s.copy()
    all_fft = np.abs(np.fft.fft(v, axis=1))
    fft_max = all_fft.max(axis=1)
    all_values = -np.sort(-fft_max)
    std = all_values.std()
    mean = all_values.mean()
    thres = threshold if threshold is not None else mean
    # print(f"mean: {mean}, std: {mean + std}")
    s_reduced[fft_max > thres] = 0
    return s_reduced

