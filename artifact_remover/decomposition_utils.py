import numpy as np
import scipy
from scipy.fft import rfft, rfftfreq
from fbpca import pca
from sklearn.utils.extmath import randomized_svd
from numpy.lib.stride_tricks import as_strided
from artifact_remover.processing_utils import robust_max_percentile

# transformer = ipca(n_components=40, batch_size=100)
# transformer = tsvd(n_components=50, n_iter=2)

# def get_signal_from_hankel(hankel):
#     max_row, max_col = hankel.shape
#     flip_matrix = np.fliplr(hankel)
#     offsets = np.arange(max_col- 1, -max_row, -1)
#     reconstructed_signal = np.array([np.mean(flip_matrix.diagonal(offset=offset)) for offset in offsets])
#     return reconstructed_signal

import numpy as np


def get_signal_from_hankel(hankel, tau=1):
    rows, cols = hankel.shape
    # Create index grid for anti-diagonals: m = i + j
    i = np.arange(rows)[:, np.newaxis]
    j = np.arange(cols)
    idx = i * tau + j  # Shape: (rows, cols)

    # Flatten for bincount
    vals = hankel.ravel()
    idx_flat = idx.ravel()

    # Compute sums and counts per m (anti-diagonal)
    total_length = (rows - 1) * tau + cols
    sums = np.bincount(idx_flat, weights=vals, minlength=total_length)
    counts = np.bincount(idx_flat, minlength=total_length)

    reconstructed_signal = sums / counts
    return reconstructed_signal


def hankel_delay_fast(x, L, tau):
    # x = np.asarray(x)
    N = len(x)

    K = N - (L - 1) * tau
    if K <= 0:
        raise ValueError("L and tau too large")

    stride = x.strides[0]

    return as_strided(
        x,
        shape=(L, K),
        strides=(tau * stride, stride)
    )

def compute_svd(emg_signal, n_rows=800, hankel=None, randomized=True, nb_principal_components=None, epsilon=None, hankel_delay=1):
    if hankel is None or n_rows != hankel.shape[0]:
        hankel = hankel_delay_fast(emg_signal, n_rows, hankel_delay)

    if randomized:
        nb_principal_components = hankel.shape[0] if nb_principal_components is None else nb_principal_components
        U, S, Vh = pca(hankel, k=nb_principal_components, raw=True, n_iter=2)

    else:
        U, S, Vh = scipy.linalg.svd(
            hankel,
            full_matrices=False,
            check_finite=False,
            overwrite_a=True,
        )
        # rank = np.sum(S > 1e-5)
        # print('hankel ranek:', rank)
        if nb_principal_components is not None:
            U, S, Vh = U[:, :nb_principal_components], S[:nb_principal_components], Vh[:nb_principal_components]
        if epsilon is not None:
            idx = np.argwhere(S > epsilon).flatten()
            U, S, Vh = U[idx], S[idx], Vh[idx]
    # if np.mean(S[-10:]) > 1e-5:
    #     print("WARNING: Singular values do not drop close to zero, you might want to increase the number of principal components.", 
    #           "The current value is: ", np.mean(S[-10:]))
    return U, S, Vh, hankel

import numpy as np

def mean_around_row_max(X, w):
    N, M = X.shape
    idx = np.argmax(X, axis=1)
    offsets = np.arange(-w, w+1)
    cols = idx[:, None] + offsets[None, :]
    cols = np.clip(cols, 0, M-1)
    values = X[np.arange(N)[:, None], cols]
    return values.mean(axis=1)

def peak_energy_ratio(X):
    P = np.abs(X)**2
    max_idx = np.argmax(P, axis=-1)
    return np.max(P, axis=-1) / mean_around_row_max(P, 100)

def peak_energy_ratio_log(X):
    P = np.abs(X)**2
    return 10*np.log(np.max(P, axis=-1) / np.mean(P, axis=-1))


def bound_freq(freq, lower_bound, upper_bound):
    return (freq >= lower_bound) & (freq <= upper_bound)

def absolute_max(max_value, threshold):
    return max_value < threshold

def remove_singular_values(v, s, u, data_rate=None, freq_bounds=[10, 450], factor=0.5, fft_freqs=None):
    # all_fft = (np.abs(rfft(v, axis=-1))**2).max(axis=-1)
    # freqs=np.fft.rfftfreq(v.shape[-1], d=1/2000)
    # import matplotlib.pyplot as plt
    # for i in range(5):
    #     fig, axes = plt.subplots(5, 2, num=i)
    #     ax=axes.flatten()
    #     count = 0
    #     for k in range(i*10, (i+1)*10):
    #         ax[count].plot(rfftfreq(v.shape[-1], d=1/2000), np.abs(rfft(v, axis=-1))[k])
    #         count += 1
    # plt.show(block=True)
    freq = scipy.fft.rfftfreq(v.shape[1], 1/data_rate) if fft_freqs is None else fft_freqs
    all_fft_max = (np.abs(rfft(v, axis=-1)))
    peak_ratio = peak_energy_ratio(all_fft_max)
    ratio = peak_ratio / np.max(peak_ratio)
    freqs = freq[np.argwhere(all_fft_max == all_fft_max.max(axis=-1, keepdims=True))[:, 1]]
    to_reject = [np.argwhere((freqs <= freq_bounds[0]) | (freqs >= freq_bounds[1]))[:, 0], 
             np.argwhere((ratio > np.median(ratio) + factor*np.std(ratio)))[:, 0]
             ]
    s[np.unique(np.hstack(to_reject))] = 0
    return s, v, u


def remove_singular_values_offline(v, s, u, threshold=2, weight_matrix=None, data_rate=None):
    # all_fft_max = (np.abs(rfft(v, axis=-1)))
    # import matplotlib.pyplot as plt
    # freq = scipy.fft.rfftfreq(v.shape[1], 1/data_rate)

    # c='b'
    # for i in range(4):
    #     fig, axes = plt.subplots(20, 2)
    #     axes = axes.flatten()
    #     start = i * 40
    #     for j in range(40):
    #         # if ratio[start + j] > 0.4:
    #         #     c = 'b'
    #         axes[j].plot(freq, all_fft_max[start + j], c)
    # plt.show(block=True)
    # plt.scatter(np.arange(0, len(all_fft_max)), peak_energy_ratio(all_fft_max) / np.max(peak_energy_ratio(all_fft_max)))


    # all_fft_max = (np.abs(rfft(v, axis=-1))).max(axis=-1)
    # fft_mean = all_fft_max.mean()
    # freq = rfftfreq(v.shape[-1], d=1/1948)
    # sorted_fft = -np.sort(-all_fft_max)
    # cum_sum = np.cumsum(sorted_fft)
    # cum_sum = np.cumsum((np.abs(rfft(v, axis=-1))**2), axis=-1) / np.cumsum((np.abs(rfft(v, axis=-1))**2), axis=-1)[:, -1:]
    # dcum_sum = np.diff(cum_sum, axis=-1)
    # sorted_dcum_sum = -np.sort(-dcum_sum.max(axis=-1), axis=0)
    # mean_dcum_sum = np.mean(sorted_dcum_sum[sorted_dcum_sum > 0.1])
    # import matplotlib.pyplot as plt
    # iqr = np.percentile(dcum_sum.max(axis=-1), 95)
    # mean = np.median(dcum_sum.max(axis=-1)) + 1 * np.std(dcum_sum.max(axis=-1))
    # thres = mean
    freq = scipy.fft.rfftfreq(v.shape[1], 1/data_rate)
    # print(np.mean(s[-10:]))
    all_fft_max = (np.abs(rfft(v, axis=-1)))
    # peak_energy = peak_energy_ratio(all_fft_max)
    # plt.plot(peak_energy)
    # plt.show(block=True)
    ratio = peak_energy_ratio(all_fft_max) / np.max(peak_energy_ratio(all_fft_max))

    freqs = freq[np.argwhere(all_fft_max == all_fft_max.max(axis=-1, keepdims=True))[:, 1]]
    to_reject = [np.argwhere((freqs <= 10) | (freqs >= 450))[:, 0], 
                #  np.argwhere(peak_energy > 0.3)[:, 0]
             np.argwhere((ratio > np.median(ratio) + 0.2*np.std(ratio)))[:, 0]
             ]

    # import matplotlib.pyplot as plt
    # # plt.plot(np.sort(robust_max_percentile(all_fft_max, 95)))
    # plt.scatter(np.arange(0, len(all_fft_max)), peak_energy_ratio(all_fft_max))

    # plt.hlines(peak_energy_ratio(all_fft_max).mean() + peak_energy_ratio(all_fft_max).std()*2, 0, 300)
    # plt.show(block=True)

    to_reject = np.unique(np.hstack(to_reject))
    
    # to_reject = np.intersect1d(to_reject[0], to_reject[1])
    # s_copy = np.zeros_like(s)
    # s_copy[to_reject] = s[to_reject]
    # s = s_copy
    s[to_reject] = 0
    # fig, axes = plt.subplots(20, 1)
    # y = 0
    # for i in range(15, 100): #len(all_fft_max)):
    #     color = 'r' if dcum_sum.max(axis=-1)[i] > iqr else 'b'
    #     # plt.plot(freq, cum_sum[i] / cum_sum[i, -1], color=color)
    #     plt.plot(freq[:-1], dcum_sum[i] + y, color=color)
    #     # plt.hlines(iqr, freq[0], freq[-1], colors='green', linestyles='dashed')
    #     y += (dcum_sum[i].max() * 0.6)
    #     # plt.set_axis_off()
    # plt.show()
    # plt.plot(np.cumsum(sorted_fft))
    # ax = plt.gca()
    # ax.hlines(fft_mean, 0, len(all_fft_max), colors='red', linestyles='dashed')
    # ax_twin = ax.twinx()
    # ax_twin.plot(s, color='orange')
    # plt.show(block=True)

    # thres = threshold if threshold is not None else fft_mean * 1.4
    # thres = max(thres, absolute_threshold)
    # plt.plot(-np.sort(-dcum_sum.max(axis=-1)))
    # plt.show(block=True)
    # # if np.percentile(dcum_sum.max(axis=-1), 95) > 0.25:
    # thres = 0.47
    # s[dcum_sum.max(axis=-1) > thres] = 0
    # s[80:] = 0
    # idxs = np.where(all_fft_max > thres)[0]

    # idxs = np.where(all_fft_max > thres)[0]

    # x = (idxs - idxs.min()) / (idxs.max() - idxs.min() + 1e-12)

    # gamma = 2.0
    # weights = np.exp(gamma * x)
    # weights /= weights.max()   # scale to [0, 1]

    # s_test = s.copy()
    # s_test[idxs] *= weights
    # plt.plot(s_test, label='original s')
    # plt.show(block=True)
    # s = s_test
    # s[dcum_sum.max(axis=-1) > thres] = 0
    # s[:32] = 0
    # s[200:] = 0
    # s[:35] = 0
    # s[100:] = 0
    # u = u[:, :35]

    # s[dcum_sum.max(axis=-1) < 0.1] = 0

    return s, v, u