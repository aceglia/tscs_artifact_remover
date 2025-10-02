import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd


def compute_svd(emg_signal, n_rows=800, hankel=None, randomized=True):
    if hankel is None or n_rows != hankel.shape[0]:
        hankel = scipy.linalg.hankel(emg_signal[:int(n_rows)], emg_signal[int(n_rows - 1):])
    
    if randomized:
        U, S, Vh = randomized_svd(hankel, n_components=100, n_iter='auto', random_state=None)
        # U, S, Vh = scipy.sparse.linalg.svds(hankel, k=50, solver='arpack', which='LM',
                                            #  maxiter=10, return_singular_vectors=True)
    else:
        U, S, Vh = scipy.linalg.svd(hankel, full_matrices=False,
                                    check_finite=False,
                                    overwrite_a=True)
        
    return U, S, Vh, hankel

def remove_singular_values(v, s, threshold=2, n_points=50):

    # Singular value diff thresholding
    # diff = s[:-1] - s[1:]
    # threshold = 2
    # idxs = np.argwhere(diff[:n_points] > diff[:n_points].mean() + threshold * diff[:n_points].std())
    # if idxs.shape[0] > 0:
    #     max_idx = int(idxs.max()) + 1
    #     s[:max_idx] = 0

    # FFT based thresholding
    all_fft = np.abs(np.fft.fft(v, axis=1))
    fft_max = all_fft.max(axis=1)
    # all_values = -np.sort(-fft_max)
    mean = fft_max.mean()
    std = fft_max.std()
    thres = threshold if threshold is not None else mean
    s[fft_max > thres] = 0
    # plt.scatter(np.arange(0, len(all_values)), all_values)
    # plt.axhline(y=thres, color='r', linestyle='-')
    # plt.axhline(y=mean + all_values.std(), color='r', linestyle='-')
    # plt.show()
    return s

