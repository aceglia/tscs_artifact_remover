import matplotlib.pyplot as plt
import numpy as np
from biosiglive import load

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

if __name__ == '__main__':
    file_path = "synth_data_with_artifact_all_results_final.bio"
    data = load(file_path)
    # plot color map for each parameter with freqency in x-axis and ratio in y-axis and value in color
    ratios = [0.3, 0.5, 0.8, 1, 1.2]
    freqs = [30, 50, 80, -1]
    pearson_mat = np.zeros((len(ratios), len(freqs)))
    mdf_mat = np.zeros((len(ratios), len(freqs)))
    ratio_mat = np.zeros((len(ratios), len(freqs)))
    keys = data.keys()
    def clean_dict(d):
        final_dict = {}
        for key in list(d.keys()):
            final_dict[key] = {}
            for key2 in list(d[key].keys()):
                if isinstance(d[key][key2], (list, np.ndarray)):
                    final_dict[key][key2] = d[key][key2][0]
                else:
                    final_dict[key][key2] = d[key][key2]
        return final_dict
    data = clean_dict(data)
    base_mdf = None
    base_ratio = None
    keys = [k for k in keys if "15_hz" not in k]
    for param in keys:
        if base_mdf is None:
            base_mdf = data[param]["artifactfree_mdf"]
            base_ratio = data[param]["artifactfree_ratio"]
        ratio = [r for r in ratios if "_" + str(r) + "_" in param][0]
        freq = [f for f in freqs if "_" + str(f) + "_" in param][0]
        pearson_mat[ratios.index(ratio), freqs.index(freq)] = data[param]["pearson"]
        mdf_mat[ratios.index(ratio), freqs.index(freq)] =data[param]["reduced_mdf"] # rmse(data[param]["artifactfree_mdf"], data[param]["reduced_mdf"])
        ratio_mat[ratios.index(ratio), freqs.index(freq)] = data[param]["reduced_ratio"] #rmse(data[param]["artifactfree_ratio"], data[param]["reduced_ratio"])
        if freq == 80:
            pearson_mat[ratios.index(ratio), freqs.index(freq) + 1] = data[param]["original_pearson"]
            mdf_mat[ratios.index(ratio), freqs.index(freq) + 1] =data[param]["original_mdf"] # rmse(data[param]["artifactfree_mdf"], data[param]["reduced_mdf"])
            ratio_mat[ratios.index(ratio), freqs.index(freq) + 1] = data[param]["original_ratio"]

    import pandas as pd
    def create_pd(mat):
        col = [f"{f} Hz" for f in freqs]
        row = [f"{r} ratio" for r in ratios]
        return pd.DataFrame(mat, index=row, columns=col)
    
    colors = ["#333333ff", "#0000ffff", "#800080ff", "#c83637ff"]
    ticks_font = 20
    axis_font = 24
    title_font = 26
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    pearson_pd = create_pd(pearson_mat)
    ax = axes[2]
    pearson_pd.plot(ax=ax,kind='bar', stacked=False, rot=0, color=colors)

    xmin, xmax = ax.get_xlim()

    ax.set_title("Pearson correlation", fontsize=title_font)
    ax.set_xticks(np.arange(len(ratios)))
    ax.set_xticklabels([f"{r}" for r in ratios], fontsize=ticks_font)
    ax.set_xlabel("Level of artifact (ratio)", fontsize=axis_font)
    ax.set_ylabel("Coefficent", fontsize=axis_font)
    ax.tick_params(axis='y', labelsize=ticks_font)
    mdf_pd = create_pd(mdf_mat)
    ax = axes[1]
    mdf_pd.plot(ax=ax, kind='bar', stacked=False, rot=0, color=colors)
    ax.hlines(y=base_mdf, xmin=xmin, xmax=xmax, color="#c83637ff", linestyle='--', linewidth=3)
    ax.tick_params(axis='y', labelsize=ticks_font)
    # add text for base mdf

    ax.set_title("Median frequency", fontsize=title_font)
    ax.set_xticks(np.arange(len(ratios)))
    ax.set_xticklabels([f"{r}" for r in ratios], fontsize=ticks_font)
    # set yticks label font
    ax.tick_params(axis='y', labelsize=ticks_font)
    ax.set_xlabel("Level of artifact (ratio)", fontsize=axis_font)
    ax.set_ylabel("Frequency (Hz)", fontsize=axis_font)

    ratio_mat = create_pd(ratio_mat)
    ax = axes[0]
    ratio_mat.plot(ax=ax, kind='bar', stacked=False, rot=0, color=colors)
    ax.hlines(y=base_ratio, xmin=xmin, xmax=xmax, color="#c83637ff", linestyle='--', linewidth=3)
    
    ax.set_title("EMG/Baseline ratio", fontsize=title_font)
    ax.set_xticks(np.arange(len(ratios)))
    ax.set_xticklabels([f"{r}" for r in ratios], fontsize=ticks_font)
    ax.set_xlabel("Level of artifact (ratio)", fontsize=axis_font)
    ax.set_ylabel("Ratio", fontsize=axis_font)
    ax.tick_params(axis='y', labelsize=ticks_font)

    plt.show()
        