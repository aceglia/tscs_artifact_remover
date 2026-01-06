import numpy as np
import scipy
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import IncrementalPCA as ipca
from sklearn.decomposition import TruncatedSVD as tsvd
from fbpca import pca

# transformer = ipca(n_components=40, batch_size=100)
# transformer = tsvd(n_components=50, n_iter=2)

def get_signal_from_hankel(hankel):
    max_row, max_col = hankel.shape
    flip_matrix = np.fliplr(hankel)
    offsets = np.arange(max_col- 1, -max_row, -1)
    reconstructed_signal = np.array([np.mean(flip_matrix.diagonal(offset=offset)) for offset in offsets])
    return reconstructed_signal

def compute_svd(emg_signal, n_rows=800, hankel=None, randomized=True):
    if hankel is None or n_rows != hankel.shape[0]:
        hankel = scipy.linalg.hankel(emg_signal[:int(n_rows)], emg_signal[int(n_rows - 1):])
    
    if randomized:
        # U, S, Vh = randomized_svd(hankel, n_components=40, n_iter='auto', random_state=None)
        # U, S, Vh = pca(hankel, k=50, raw=True, n_iter=2)
        # U, S, Vh = transformer.fit(hankel)
        # transformer.fit(hankel)
        # U = transformer.components_
        # S = transformer.singular_values_
        # Vh = transformer.components_
        U, S, Vh = pca(hankel, k=50, raw=True, n_iter=2)

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
    # threshold = 1
    # # plt.plot(s)
    # idxs = np.argwhere(diff[:n_points] > diff[:n_points].mean() + threshold * diff[:n_points].std())
    # if idxs.shape[0] > 0:
    #     max_idx = int(idxs.max()) + 1
    #     s[:max_idx] = 0
    # plt.plot(s)
    # plt.show()


    # FFT based thresholding
    all_fft = np.abs(np.fft.fft(v, axis=1))
    # for i in range(5):
    #     fig, axes = plt.subplots(5, 2, num=i)
    #     ax=axes.flatten()
    #     count = 0
    #     for k in range(i*10, (i+1)*10):
    #         ax[count].plot(all_fft[k])
    #         count += 1
    # plt.show()

    fft_max = all_fft.max(axis=1)
    # all_values = -np.sort(-fft_max)
    mean = fft_max.mean()
    std = fft_max.std()
    thres = threshold if threshold is not None else mean 
    # s[fft_max > thres] = 0
    s[fft_max < thres] = 0


    # plt.scatter(np.arange(0, len(all_values)), all_values)
    # plt.axhline(y=thres, color='r', linestyle='-')
    # plt.axhline(y=mean + all_values.std(), color='r', linestyle='-')
    # plt.show()
    return s

