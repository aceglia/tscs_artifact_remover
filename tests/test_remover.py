import numpy as np
from unittest.mock import MagicMock, patch
import pytest
from star_emg.automatic_remover import ArtifactRemover

# =============================================================================
# Constructor
# =============================================================================


def test_init_without_data():
    remover = ArtifactRemover()

    assert remover.ratio is None
    assert remover.transformer is None
    assert remover.solution is None
    assert remover.is_data_loaded is False


@patch("star_emg.automatic_remover.DataLoader")
def test_load_data(mock_loader):
    remover = ArtifactRemover()

    remover.load_data("dummy_path", {})

    mock_loader.assert_called_once_with("dummy_path")
    assert remover.is_data_loaded is True


# =============================================================================
# Notch filter
# =============================================================================


def test_notch_filter_preserves_shape():
    fs = 2000
    t = np.arange(0, 2, 1 / fs)

    signal = np.sin(2 * np.pi * 50 * t)

    result, _, _ = ArtifactRemover._perform_notch_filter(
        frequency_peaks=50,
        data=signal,
        fs=fs,
        return_dict=False,
    )

    assert result.shape == signal.shape


def test_notch_filter_dict_output():
    fs = 2000
    t = np.arange(0, 1, 1 / fs)

    signal = np.sin(2 * np.pi * 50 * t)

    result, _, _ = ArtifactRemover._perform_notch_filter(
        frequency_peaks=50,
        data=signal,
        fs=fs,
        return_dict=True,
    )

    assert "output" in result
    assert "unfiltered_signal" in result
    assert result["output"].shape == signal.shape


# =============================================================================
# Window processing
# =============================================================================
def test_perform_window_process_notch():
    np.random.seed(0)
    signal = np.random.rand(4000)
    result = ArtifactRemover.perform_window_process(
        data=signal,
        notch_filter=True,
        return_dict=True,
        window=1000,
        frequency_peaks=30,
    )

    assert "output" in result
    assert len(result["output"]) == len(signal)
    target = np.array(
        [
            0.63281356,
            0.6210879,
            0.66459457,
            0.50677262,
            0.4273956,
            0.61661493,
            0.45623981,
            0.82049143,
            0.94850842,
            0.43063254,
            0.69057915,
            0.532716,
            0.59481495,
            0.89646775,
            0.12956648,
            0.10271445,
            0.10999502,
            0.78460095,
            0.8103814,
            0.83022528,
        ]
    )
    assert np.allclose(result["output"][:20], target, atol=1e-6)


# =============================================================================
# Decomposition
# =============================================================================
@pytest.mark.parametrize(
    "return_dict",
    [True, False],
)
def test_perform_decomposition(return_dict):
    np.random.seed(0)
    signal = np.random.rand(1000)

    result, rejected = ArtifactRemover._perform_decomposition(
        signal,
        hankel_size=100,
        return_dict=return_dict,
        data_rate=2000,
        freq_bounds=[10, 450],
        factor=0.5,
    )
    if return_dict:
        assert "output" in result
        assert "s" in result
        assert "s_reduced" in result
        output = result["output"]
    else:
        output = result
    assert np.allclose(
        output[:20],
        np.array(
            [
                0.08040893,
                0.12911394,
                0.11584106,
                0.07737867,
                0.09947618,
                0.10440376,
                0.14490228,
                0.20889045,
                0.16582877,
                0.07103342,
                0.13191337,
                0.23022345,
                0.27124749,
                0.11893307,
                -0.17491622,
                -0.28562535,
                -0.16239762,
                0.09343103,
                0.32206545,
                0.40023976,
            ]
        ),
        atol=1e-6,
    )
    assert np.allclose(
        rejected,
        np.array(
            [ 0,  1,  2,  5,  6,  8, 11, 15, 17, 18, 19, 20, 22, 23, 24, 25, 26,
       27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 41, 43, 44, 46, 47,
       48, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 64, 65, 67, 68, 69,
       71, 75, 76, 79, 80, 83, 84, 85, 86, 87, 88, 89, 90, 92, 93, 94, 95,
       96, 97, 98, 99]
        ),
        atol=1e-6,
    )


# =============================================================================
# Worker
# =============================================================================


@patch.object(ArtifactRemover, "perform_window_process")
def test_worker(mock_process):
    mock_process.return_value = {"output": np.array([1, 2, 3])}

    args = (
        np.array([1, 2, 3]),
        10,  # hankel_size
        False,  # randomized
        False,  # post_filter
        None,  # nb_pc
        None,  # epsilon
        False,  # notch_filter
        150,  # q
        30,  # frequency_peaks
        None,  # first_peak
        1,  # hankel_delay
        1000,  # window
        2000,  # data_rate
        [10, 450],  # freq_bounds
        0.5,  # factor
        np.array([1, 2, 3]),
    )

    ArtifactRemover.worker(args)

    mock_process.assert_called_once()


# =============================================================================
# Getters
# =============================================================================


def test_getters():
    remover = ArtifactRemover()

    remover.solution = MagicMock()
    remover.solution.get.return_value = np.array([1, 2, 3])

    assert remover.get_process_signal() is not None
    assert remover.get_singular_values() is not None


def test_get_data_rate():
    remover = ArtifactRemover()

    remover.data_loader = MagicMock()

    remover.data_loader.data_rate = 2000

    assert remover.get_data_rate() == 2000


def test_get_channel_names():
    remover = ArtifactRemover()

    remover.data_loader = MagicMock()
    remover.data_loader.channel_names = ["EMG1", "EMG2"]

    assert remover.get_channel_names() == ["EMG1", "EMG2"]
