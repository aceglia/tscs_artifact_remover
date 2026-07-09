from datetime import time
from os import path
from pathlib import Path

import numpy as np
import zarr
from star_emg.app.save_utils import ZarrRecording


SAVE_PATH = Path("test_recording.zarr")


if __name__ == "__main__":
    # Create a ZarrRecording instance
    rec = ZarrRecording(
    "test_recording.zarr",
    n_channels=8,
    sampling_rate=2000,
    channel_names=[
        "TA",
        "SOL",
        "MG",
        "LG",
        "RF",
        "VL",
        "BF",
        "ST",
    ],
    )

    n = 10000

    raw = np.random.randn(8, n)
    processed = raw * 0.8
    time = np.arange(n) / 2000

    rec.append_signals(raw, processed, time)

    rec.append_config(
    timestamp=12.54,
    channel=[2, 1],
    parameters={
        "method": "SVD",
        "hankel_size": 80,
        "process_window": 40,
        "hankel_delay": 8,
        "rank": 4,
        "freq_bounds": [20, 450],
        "threshold": 0.85,
    },
    )
    rec.close()
    
    zarr_dataset = ZarrRecording.load_dataset(SAVE_PATH)
    print(zarr_dataset.root['signals']['raw'].shape)
    print(zarr_dataset.get_size())