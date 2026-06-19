import json

signal_preprocessing = {"cutoff": [10, 450], "order": 2, "center": True, "signal_filter": True}

signal_evaluation = {
    "clean_percentile": 99.9,
    "artifacts_percentile": 99.6,
    "time_domain_windows": 20,
}
