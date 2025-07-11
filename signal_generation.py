import scipy
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import csv
from biosiglive import load, save


# fit curve to exponential
# expo = np.polyfit(np.arange(0,  36), expo, 1)
def func(x, a, b, c):
    return a * np.exp(-b * x) + c


def load_from_file(file_path):
    # data = loadmat(file_path)
    # open csv
    frames = []
    with open(file_path, 'r') as file:
        # reader = csv.reader(file, delimiter=',')
        reader = csv.reader(file, delimiter='\t')
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
    header = frames[0][0]
    data = array[:, 1:, :].astype(float)
    return header, data


def get_artifact_template(artifact_frequency, n_data_points, rate=2000, white_noise=True):
    artifact_template = np.array([0.00061, -0.240173, -0.358582, 0.002289, 0.108032,
                                  0.093231, 0.077515, 0.068512, 0.059204, 0.051727, 0.046692, 0.040283,
                                  0.0354, 0.031433, 0.027466, 0.023041, 0.022125, 0.019226, 0.016174,
                                  0.015564, 0.013275, 0.011139, 0.007019])

    artifact_duration = 1 / artifact_frequency
    n_points_artifact = int(artifact_duration * rate)
    if n_points_artifact < artifact_template.shape[0]:
        x = np.arange(0, artifact_duration, 1 / rate)
        f = scipy.interpolate.interp1d(x, artifact_template, kind='linear')
        artifact_template = f(np.arange(0, artifact_duration, 1 / rate))
    n_zeros = n_points_artifact - artifact_template.shape[0]
    artifact = np.concatenate((artifact_template, np.zeros(n_zeros)))
    expo = artifact[np.argmax(artifact):]
    x = np.arange(0, expo.shape[0])
    popt, pcov = scipy.optimize.curve_fit(func, x, expo)
    new_data = func(x, *popt)
    artifact[np.argmax(artifact):] = new_data
    artifact = np.concatenate((-artifact, artifact))
    number_of_artifact = int(np.ceil(n_data_points / artifact.shape[0]))
    artifact = np.tile(artifact, number_of_artifact)
    artifact = artifact[:n_data_points]
    if white_noise:
        for data in [artifact]:
            white_noise = np.random.normal(0, 0.003, data.shape[0])
            data += white_noise
    return artifact


def create_signal(original_emg, artifact_ratio, artifact_frequency, white_noise=True, rate=2000, save_file=False, save_path=''):
    final_dic = {}
    max = np.max(original_emg)
    for f in artifact_frequency:
        artifact = get_artifact_template(f, original_emg.shape[0], rate, white_noise)
        plt.plot(artifact)
        plt.show()
        max_artifact = np.max(artifact)
        for r in artifact_ratio:
            ratio_max = (max * r) / max_artifact
            final_dic[f"data_with_ratio_{r}_{f}_hz"] = (original_emg + ratio_max * artifact)
    if save_file:
        save(final_dic, save_path, safe=False)
    return final_dic


if __name__ == '__main__':
    file_path = r'D:\Documents\Udem\Postdoctorat\ameddeo\005tSCS\Gait\test_stim_artifact.txt'
    h_2, data_2 = load_from_file(file_path)
    data_2 = data_2[0, :, -1]
    emg_data = r"D:\Documents\Programmation\biosiglive\examples\abd.bio"
    emg_data = load(emg_data)["emg"][2, 2000:] * 100
    artifact_dic = create_signal(emg_data, [0, 0.3, 0.5, 0.8, 1, 1.2, 1.5, 1.8], [15, 30, 50, 80],
                                  white_noise=True, save_file=True,
                                 save_path="synth_data_with_artifact_all.bio")
    alpha = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1]
    count = 0
    for key in artifact_dic:
        if key == f"data_with_ratio_1.2_80_hz":
            plt.plot(artifact_dic[key])
        count += 1
    plt.show()
