import matplotlib.pyplot as plt
import numpy as np
from biosiglive import load

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

if __name__ == '__main__':
    file_path = "test001_results.bio"
    data = load(file_path)
    # plot color map for each parameter with freqency in x-axis and ratio in y-axis and value in color
    ratios = [1, 2, 3, 4, 5, 6]
    freqs = [30, -1]
    mdf_mat = np.zeros((len(ratios), len(freqs)))
    ratio_mat = np.zeros((len(ratios), len(freqs)))
    keys = data.keys()
    keys = [k for k in keys]
    for param in keys:
        ratio = param 
        freq = 30
        for i in range(2):
            if i==0:
                mdf_mat[ratios.index(ratio), i] =data[param]["reduced_mdf"] # rmse(data[param]["artifactfree_mdf"], data[param]["reduced_mdf"])
                ratio_mat[ratios.index(ratio), i] = data[param]["reduced_ratio"] #rmse(data[param]["artifactfree_ratio"], data[param]["reduced_ratio"])
            else:
                mdf_mat[ratios.index(ratio), i] = data[param]["original_mdf"]
                ratio_mat[ratios.index(ratio), i] = data[param]["original_ratio"]
    print(mdf_mat)
    print(ratio_mat)
    print('mean RMSE of MDF:', np.mean(mdf_mat))
    print('mean RMSE of ratio:', np.mean(ratio_mat))

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Artifact level")
    import pandas as pd
    def create_pd(mat):
        col = [f"{f} Hz" for f in freqs]
        row = [f"{r} ratio" for r in ratios]
        return pd.DataFrame(mat, index=row, columns=col)
    
    colors = ["#333333ff", "#c83637ff"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    mdf_pd = create_pd(mdf_mat)
    ax = axes[1]

    mdf_pd.plot(ax=ax, kind='bar', stacked=False, rot=0, color=colors)
    ratios = [5, 10, 15, 20, 25, 30]
    ticks_font = 20
    axis_font = 24
    title_font = 26
    ax.set_title("Median frequency", fontsize=title_font)
    ax.set_xticks(np.arange(len(ratios)))
    ax.set_xticklabels([f"{r}" for r in ratios], fontsize=ticks_font)
    ax.set_xlabel("Stimulation intensity (mA)", fontsize=axis_font)
    ax.set_ylabel("Frequency (Hz)", fontsize=axis_font)
    ax.tick_params(axis='y', labelsize=ticks_font)
    ratio_mat = create_pd(ratio_mat)
    ax = axes[0]
    ratio_mat.plot(ax=ax, kind='bar', stacked=False, rot=0, color=colors)


    ax.set_title("EMG/baseline ratio", fontsize=title_font)
    ax.set_xticks(np.arange(len(ratios)))
    ax.set_xticklabels([f"{r}" for r in ratios], fontsize=ticks_font)
    ax.set_xlabel("Stimulation intensity (mA)", fontsize=axis_font)
    ax.set_ylabel("Ratio", fontsize=axis_font)
    ax.tick_params(axis='y', labelsize=ticks_font)
    plt.show()
        
