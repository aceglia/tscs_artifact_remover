import numpy as np
import scipy
from scipy.fft import rfft
from fbpca import pca

# transformer = ipca(n_components=40, batch_size=100)
# transformer = tsvd(n_components=50, n_iter=2)

# def get_signal_from_hankel(hankel):
#     max_row, max_col = hankel.shape
#     flip_matrix = np.fliplr(hankel)
#     offsets = np.arange(max_col- 1, -max_row, -1)
#     reconstructed_signal = np.array([np.mean(flip_matrix.diagonal(offset=offset)) for offset in offsets])
#     return reconstructed_signal

import numpy as np


def get_signal_from_hankel(hankel):
    rows, cols = hankel.shape
    # Create index grid for anti-diagonals: m = i + j
    i = np.arange(rows)[:, np.newaxis]
    j = np.arange(cols)
    idx = i + j  # Shape: (rows, cols)

    # Flatten for bincount
    vals = hankel.ravel()
    idx_flat = idx.ravel()

    # Compute sums and counts per m (anti-diagonal)
    total_length = rows + cols - 1
    sums = np.bincount(idx_flat, weights=vals, minlength=total_length)
    counts = np.bincount(idx_flat, minlength=total_length)

    reconstructed_signal = sums / counts
    return reconstructed_signal


def compute_svd(emg_signal, n_rows=800, hankel=None, randomized=True, nb_principal_components=50):
    if hankel is None or n_rows != hankel.shape[0]:
        # hankel = scipy.linalg.hankel(emg_signal[: int(n_rows)], emg_signal[int(n_rows - 1) :])
        hankel = scipy.linalg.hankel(emg_signal[: int(n_rows)], emg_signal[int(n_rows - 1) :])


    if randomized:
        # U, S, Vh = randomized_svd(hankel, n_components=40, n_iter='auto', random_state=None)
        # U, S, Vh = pca(hankel, k=50, raw=True, n_iter=2)
        # U, S, Vh = transformer.fit(hankel)
        # transformer.fit(hankel)
        # U = transformer.components_
        # S = transformer.singular_values_
        # Vh = transformer.components_
        U, S, Vh = pca(hankel, k=nb_principal_components, raw=True, n_iter=1)

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
        if nb_principal_components is not None:
            U, S, Vh = U[:, :nb_principal_components], S[:nb_principal_components], Vh[:nb_principal_components]
    return U, S, Vh, hankel


def remove_singular_values(v, s, threshold=2, weight_matrix=None):

    # Singular value diff thresholding
    # diff = s[:-1] - s[1:]
    # threshold = 1
    # # plt.plot(s)
    # idxs = np.argwhere(diff[:n_points] > diff[:n_points].mean() + threshold * diff[:n_points].std())
    # if idxs.shape[0] > 0:
    #     max_idx = int(idxs.max()) + 1
    #     s[:max_idx] = 0
    # plt.plot(s)
    # plt.show()

    # FFT based thresholding
    # all_fft_max = np.abs(np.fft.fft(v, axis=1)).max(axis=1)
    all_fft_max = (np.abs(rfft(v, axis=1))**2).max(axis=1)
    # for i in range(5):
    #     fig, axes = plt.subplots(5, 2, num=i)
    #     ax=axes.flatten()
    #     count = 0
    #     for k in range(i*10, (i+1)*10):
    #         ax[count].plot(all_fft[k])
    #         count += 1
    # plt.show()
    # weight_matrix = weight_matrix if weight_matrix is not None else np.ones(s.shape)

    fft_mean = all_fft_max.mean() * 0.868
    # all_values = -np.sort(-fft_max)
    # mean = fft_max.mean()
    # thres = threshold if threshold is not None else fft_mean
    s[all_fft_max > fft_mean] = 0
    # if weight_matrix is None:
    #     weight_matrix = np.ones(s.shape)
    #     weight_matrix[fft_max > thres] = 0

    # s = s * weight_matrix

    # plt.scatter(np.arange(0, len(all_values)), all_values)
    # plt.axhline(y=thres, color='r', linestyle='-')
    # plt.axhline(y=mean + all_values.std(), color='r', linestyle='-')
    # plt.show()
    return s
