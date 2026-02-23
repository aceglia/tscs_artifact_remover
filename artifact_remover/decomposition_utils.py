import numpy as np
import scipy
from scipy.fft import rfft, rfftfreq
from fbpca import pca
from sklearn.utils.extmath import randomized_svd
from numpy.lib.stride_tricks import as_strided

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
        nb_principal_components = 50 if nb_principal_components is None else nb_principal_components
        # U, S, Vh = randomized_svd(hankel, n_components=nb_principal_components, n_iter=1)
        U, S, Vh = pca(hankel, k=nb_principal_components, raw=True, n_iter=2)
        # U, S, Vh = transformer.fit(hankel)
        # transformer.fit(hankel)
        # U = transformer.components_
        # S = transformer.singular_values_
        # Vh = transformer.components_
        # U, S, Vh = pca(hankel, k=nb_principal_components, raw=True, n_iter=1)

        # U, S, Vh = scipy.sparse.linalg.svds(hankel, k=nb_principal_components, solver='arpack',
        #                                     return_singular_vectors=True, tol=1e-6)
        # idx = np.argsort(S)[::-1]
        # U, S, Vh = U[:, idx], S[idx], Vh[idx]

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
    return U, S, Vh, hankel


def remove_singular_values(v, s, u, threshold=2, weight_matrix=None):
    # s_copy = s.copy()
    # FFT based thresholding
    # all_fft_max = np.abs(np.fft.fft(v, axis=1)).max(axis=1)
    all_fft_max = (np.abs(rfft(v, axis=-1))).max(axis=-1)
    v_fft = np.abs(scipy.fft.rfft(v))[0:2, :]
    freqs = np.fft.rfftfreq(v.shape[1], d=1 / 2000)
    # normalize between 0 and 1
    v_fft = v_fft / np.max(v_fft)

    # X = np.linspace(0, v_fft.shape[1], v_fft.shape[1])
    Y = np.linspace(0, v_fft.shape[0], v_fft.shape[0])
    # X, Y = np.meshgrid(freqs, Y)
    # plot contour
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.contourf(X, Y, v_fft, 100, cmap='viridis')
    # plt.colorbar()
    # plt.show()
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
    max_freqs = rfftfreq(v.shape[-1], d=1/2000)[(np.argmax(rfft(v, axis=-1), axis=-1))]
    idx_to_reject = np.argwhere((all_fft_max > all_fft_max.mean()))
    # idx_to_reject = np.concatenate((idx_to_reject, np.argwhere((max_freqs > 250))))
    # idx_to_reject = np.concatenate((idx_to_reject, np.argwhere((max_freqs < 50))))
    # idx_to_reject = np.unique(idx_to_reject).flatten()

    # idx_to_reject = []
    # for i in range(len(s)):
    #     v_i = v[i]
    #     a = np.dot(v_i[:-1], v_i[1:]) / np.dot(v_i[:-1], v_i[:-1])
    #     err = np.linalg.norm(v_i[1:] - a*v_i[:-1]) / np.linalg.norm(v_i[1:])
    #     lags = [1, 2, 3, 4, 5]
    #     rho = [abs(np.dot(v_i[:-t], v_i[t:]) / np.dot(v_i, v_i)) for t in lags]
    #     rho_mean = np.mean(rho)
    #     std_rho = np.std(rho)

    #     K=4
    #     local_flatness_i = np.std(s[i:i+K]) / np.mean(s[i:i+K])
    #     criteria = int(err < 0.03) + int(rho_mean > 0.8 and std_rho < 0.05) + int(local_flatness_i < 0.15)
    #     if criteria < 1:
    #         idx_to_reject.append(True)
    #     else:
    #         idx_to_reject.append(False)

    # u = u[:, idx_to_reject]          # shape (L, r_a)
    s[idx_to_reject] = 0
    # s[:25] = 0
    # s[100:] = 0
    # P_art = u @ u.T 
    # weight_matrix = weight_matrix if weight_matrix is not None else np.ones(s.shape)

    # fft_mean = all_fft_max.mean() 
    # s[s > s.mean()] = 0
    # all_values = -np.sort(-fft_max)
    # mean = fft_max.mean()
    # thres = threshold if threshold is not None else fft_mean
    # s[all_fft_max > fft_mean] = 0
    # s[:8] = 0
    # s = s[all_fft_max < fft_mean]
    # v = v[all_fft_max < fft_mean, :]
    # u = u[:, all_fft_max < fft_mean]

    # if weight_matrix is None:
    #     weight_matrix = np.ones(s.shape)
    #     weight_matrix[fft_max > thres] = 0

    # s = s * weight_matrix

    # plt.scatter(np.arange(0, len(all_values)), all_values)
    # plt.axhline(y=thres, color='r', linestyle='-')
    # plt.axhline(y=mean + all_values.std(), color='r', linestyle='-')
    # plt.show()
    # import matplotlib.pyplot as plt
    # plt.plot(all_fft_max)
    # plt.hlines(all_fft_max.mean(), 0, 60)
    # plt.plot(s)  
    # plt.plot(s_copy)
    # plt.show(block=True)
    return s, v, u


def remove_singular_values_offline(v, s, u, threshold=2, weight_matrix=None):
    all_fft_max = (np.abs(rfft(v, axis=-1))).max(axis=-1)
    fft_mean = all_fft_max.mean()
    freq = rfftfreq(v.shape[-1], d=1/1948)
    sorted_fft = -np.sort(-all_fft_max)
    cum_sum = np.cumsum((np.abs(rfft(v, axis=-1))**2), axis=-1) / np.cumsum((np.abs(rfft(v, axis=-1))**2), axis=-1)[:, -1:]
    dcum_sum = np.diff(cum_sum, axis=-1)
    sorted_dcum_sum = -np.sort(-dcum_sum.max(axis=-1), axis=0)
    mean_dcum_sum = np.mean(sorted_dcum_sum[sorted_dcum_sum > 0.1])
    import matplotlib.pyplot as plt
    iqr = np.percentile(dcum_sum.max(axis=-1), 70)
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

    thres = threshold if threshold is not None else fft_mean
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
    # s[all_fft_max > thres] = 0
    s[:32] = 0
    # s[200:] = 0
    # s[:35] = 0
    # s[100:] = 0
    # u = u[:, :35]

    # s[dcum_sum.max(axis=-1) < 0.1] = 0

    return s, v, u