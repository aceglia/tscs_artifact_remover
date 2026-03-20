import numpy as np

from ..automatic_remover import ArtefactRemover
from PyQt5.QtWidgets import (
    QWidget,
    QStackedWidget,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QCheckBox,
)
from PyQt5.QtCore import Qt
from .gui_utils import ChannelSelecter


class OptionWidget(QWidget):
    def __init__(self, type, parent=None):
        super().__init__()
        self.type = type
        self.channel_selecter = None
        self.parent = parent
        self.current_frame = 0
        self.process_window = 5000 if self.type != "notch" else 10000
        self.channels = []
        self.init_process_args = {
            "notch_filter": self.type == "notch",
            "quality_factor": 80,
            "frequency_peaks": 30,
            "hankel_size": 500,
            "hankel_delay": 1,
            "process_window": self.process_window,
            "factor": 0.5,
            "freq_bounds": [10, 500],
            "channel_idxs": None,
        }
        self.short_process_args = {}
        self.process_arguments = {}

        self.process_button = QPushButton("Process")
        self.process_button.setEnabled(False)
        self.process_button.clicked.connect(self.update_params)

        self.popup_button = QPushButton("Select channels to process")
        self.popup_button.clicked.connect(self.on_popup_button_clicked)

        self.input_wind = QLineEdit()
        self.input_wind.setText(str(self.process_window))
        self.input_wind.textChanged.connect(self.set_process_window)
        self.cancel_button = QPushButton("Cancel processing")
        self.cancel_button.clicked.connect(self.on_cancel_button_clicked)

    def init(self, channels, frames):
        self.channels = channels
        self.n_frames = frames
        self.channel_selecter = ChannelSelecter(self, self.channels, for_display=False)

    def get_options(self):
        return self.__dict__

    def set_options(self, **options):
        for key, value in options.items():
            setattr(self, key, value)

    def get_option(self, option):
        return getattr(self, option)

    def on_popup_button_clicked(self):
        if self.channel_selecter is None:
            self.channel_selecter = ChannelSelecter(self, self.channels, for_display=False)
        self.channel_selecter.show()

    def on_draw_clicked(self):
        self.channel_idxs = self.channel_selecter.get_channel_idxs()
        self.process_button.setEnabled(True)
        self.channel_selecter.quit()

    def update_params(self):
        process_arguments = self.get_process_arguments()
        self.parent.process(**process_arguments)
        list_empty = (
            [None] * len(self.channels)
            if f"Frame_{self.current_frame}" not in self.process_arguments
            else self.process_arguments[f"Frame_{self.current_frame}"]
        )
        channel_names = self.channel_selecter.get_channel_names()
        for c, ch in enumerate(self.channel_idxs):
            process_arguments["channel_name"] = channel_names[c]
            list_empty[ch] = process_arguments.copy()
        self.process_arguments[f"Frame_{self.current_frame}"] = list_empty

    def update_frame(self, frame_number):
        self.current_frame = frame_number

    def get_process_arguments(self, keys_item=None, value_item=None):
        args_item = self.init_process_args if keys_item is None else keys_item
        dic_item = value_item if value_item is not None else self.__dict__
        params_dict = {}
        for name, value in args_item.items():
            if name == "hankel_size" and self.type == "svd":
                params_dict[name] = self.hankel_size
                continue
            params_dict[name] = dic_item.get(name, args_item[name])
        return params_dict

    def load_config(self, config):
        self.process_arguments = config

    def get_short_config(self, value_item=None):
        return self.get_process_arguments(self.short_process_args, value_item)

    def get_args_by_idx(self, idx=None):
        if f"Frame_{self.current_frame}" not in self.process_arguments:
            return
        if idx:
            process_argument = self.process_arguments[f"Frame_{self.current_frame}"][idx]
            return self.get_short_config(value_item=process_argument)
        else:
            return self.process_arguments[f"Frame_{self.current_frame}"]

    def on_cancel_button_clicked(self):
        self.parent.cancel_processing()

    def disable(self):
        for item in self.findChildren(QWidget):
            item.setEnabled(False)
        
    def enable(self):
        for item in self.findChildren(QWidget):
            item.setEnabled(True)
        self.process_button.setEnabled(False)


class NotchOptions(OptionWidget):
    def __init__(self, parent=None):
        super().__init__("notch", parent)
        self.quality_factor = 80
        self.frequency_peaks = 30
        self.params_changed = True
        self._init_layout()
        self.short_process_args = {
            "quality_factor": 80,
            "frequency_peaks": 30,
            "process_window": self.process_window,
        }

    def _init_layout(self):
        layout = QGridLayout()
        layout.addWidget(QLabel("<b><font size=5>Notch Remover Options</font></b>"), 0, 0, 1, 2, Qt.AlignCenter)
        layout.addWidget(QLabel("Quality Factor:"), 1, 0, 1, 1)
        self.input_quality = QLineEdit()
        self.input_quality.setText(str(self.quality_factor))
        self.input_quality.textChanged.connect(self.set_quality_factor)
        layout.addWidget(self.input_quality, 1, 1, 1, 1)

        layout.addWidget(QLabel("Frequency Peaks:"), 2, 0, 1, 1)
        self.input_freq = QLineEdit()
        self.input_freq.setText(str(self.frequency_peaks))
        self.input_freq.textChanged.connect(self.set_frequency_peaks)
        layout.addWidget(self.input_freq, 2, 1, 1, 1)

        layout.addWidget(QLabel("Process window lenght:"), 3, 0, 1, 1)

        layout.addWidget(self.input_wind, 3, 1, 1, 2)

        layout.addWidget(self.popup_button, 4, 0, 1, 2)
        layout.addWidget(self.process_button, 5, 0, 1, 2)
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

    def set_quality_factor(self, text):
        if text == "":
            return
        self.quality_factor = float(text)

    def set_frequency_peaks(self, text):
        if text == "":
            return
        self.frequency_peaks = float(text)

    def set_process_window(self, text):
        if text == "":
            return
        self.process_window = int(text)


class SVDOptions(OptionWidget):
    def __init__(self, parent=None):
        super().__init__("svd", parent)
        self.hankel_delay = 1
        self._hankel_size = 500
        self.process_window = 1000
        self.overlap = 0
        self.nb_principal_components = None
        self.factor = 0.35
        self.freq_bounds = [10, 300]
        self.automatic_hsize = True
        self._init_layout()
        self.short_process_args = {
            "hankel_size": 500,
            "hankel_delay": 1,
            "process_window": self.process_window,
            "factor": 0.5,
            "freq_bounds": [10, 500],
        }

    def _init_layout(self):
        layout = QGridLayout()
        layout.addWidget(QLabel("<b><font size=5>SVD Remover Options</font></b>"), 0, 0, 1, 3, Qt.AlignCenter)
        layout.addWidget(QLabel("Process window lenght:"), 1, 0, 1, 2)

        layout.addWidget(self.input_wind, 1, 1, 1, 1)

        layout.addWidget(QLabel("Hankel size:"), 2, 0, 1, 1)
        self.input_hankel = QLineEdit()
        self.input_hankel.setText(str(self.hankel_size))
        self.input_hankel.textChanged.connect(self.set_hankel_size)
        self.check_auto_size = QCheckBox("Automatic size")
        self.check_auto_size.setChecked(self.automatic_hsize)
        self.check_auto_size.stateChanged.connect(self.set_automatic_hsize)
        self.set_automatic_hsize(Qt.Checked, update=False)
        layout.addWidget(self.input_hankel, 2, 1, 1, 1)
        layout.addWidget(self.check_auto_size, 2, 2, 1, 1)

        layout.addWidget(QLabel("Select threshold:"), 3, 0, 1, 2)
        self.input_factor = QLineEdit()
        self.input_factor.setText(str(self.factor))
        self.input_factor.textChanged.connect(self.set_factor)
        layout.addWidget(self.input_factor, 3, 1, 1, 1)

        layout.addWidget(QLabel("Select frequency bounds:"), 4, 0, 1, 1)
        self.input_low_freq = QLineEdit()
        self.input_low_freq.setText(str(self.freq_bounds[0]))
        self.input_low_freq.textChanged.connect(self.set_low_freq)
        layout.addWidget(self.input_low_freq, 4, 1, 1, 1)

        self.input_high_freq = QLineEdit()
        self.input_high_freq.setText(str(self.freq_bounds[1]))
        self.input_high_freq.textChanged.connect(self.set_high_freq)
        layout.addWidget(self.input_high_freq, 4, 2, 1, 1)

        layout.addWidget(self.popup_button, 5, 0, 1, 3)
        layout.addWidget(self.process_button, 6, 0, 1, 3)
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

    def set_process_window(self, text):
        if text == "":
            return
        self.process_window = int(text)
        if self.automatic_hsize:
            self.set_automatic_hsize(Qt.Checked)

    def set_hankel_size(self, text):
        if text == "":
            return
        self._hankel_size = int(text)

    def set_factor(self, text):
        if text == "":
            return
        self.factor = float(text)

    def set_low_freq(self, text):
        if text == "":
            return
        self.freq_bounds[0] = int(text)

    def set_high_freq(self, text):
        if text == "":
            return
        self.freq_bounds[1] = int(text)

    def set_automatic_hsize(self, state, update=True):
        self.automatic_hsize = state == Qt.Checked
        self.input_hankel.setEnabled(not self.automatic_hsize)
        self.input_hankel.setText(str(self.hankel_size_from_window()))

    def hankel_size_from_window(self, factor=8):
        if int((self.process_window / factor) / self.hankel_delay) < 500:
            return 500
        return int((self.process_window / factor) / self.hankel_delay)

    @property
    def hankel_size(self):
        if self.automatic_hsize:
            return self.hankel_size_from_window()
        return self._hankel_size


class Remover:
    def __init__(self, parent=None):
        self.remover = None
        self.parent = parent
        self.is_initialized = False
        self.is_results = False
        self.sol_notch = False
        self.sol_svd = False
        self.notch_options = NotchOptions(self.parent)
        self.svd_options = SVDOptions(self.parent)
        self.process_widgets = QStackedWidget()
        self.process_widgets.addWidget(self.notch_options)
        self.process_widgets.addWidget(self.svd_options)
        self.process_widgets.setCurrentIndex(0)
        self.current_filter = "notch"
        self.filter_list = ["notch", "svd"]

    def _init_remover(self, path_file, **kwargs):
        self.remover = ArtefactRemover(data=path_file, **kwargs)
        data_shape = self.remover.data_loader.init_data.shape
        window = 25_000
        if data_shape[-1] > window and data_shape[0] == 1:
            total_epochs = int(np.ceil(data_shape[-1] / 25000))
            rate = self.remover.data_loader.data_rate
            epochs_duration = window * (1/rate)
            self.parent.parent.show_split_windows(
                f"The data has {data_shape[-1]} points per channel. This may slowdown the programm."
                f"If you chose 'yes' your file will be splitted in {total_epochs} of {epochs_duration:.2f}s duration Frames. If the last frame is not full, it will be filled with 0."
            )
        if self.parent.parent._split == '&Yes':
            self._split_data(window, total_epochs, rate)
        elif self.parent.parent._split == 'Cancel':
            self.remover = None

        for options in [self.notch_options, self.svd_options]:
            options.init(self.remover.data_loader.channel_names, self.remover.data_loader.init_data.shape[0])

    def _split_data(self, window, epochs, rate):
        data = np.swapaxes(self.remover.data_loader.init_data, 0, 1)
        full_mat = np.zeros((data.shape[0], 1, int(epochs * window)))
        full_mat[..., :data.shape[-1]] = data
        full_mat = full_mat.reshape(data.shape[0], epochs, window)
        self.remover.data_loader.init_data = np.swapaxes(full_mat, 0, 1)
        time = np.linspace(0, int(epochs * window)/rate, int(epochs * window))
        self.remover.data_loader.time = time.reshape((epochs, window))
        
    def set_file(self, file_path, signal_filter=True, center=True, cutoff=[10, 450]):
        self.process_widgets.setCurrentIndex(0)
        self.current_filter = "notch"
        self._init_remover(file_path, signal_filter=signal_filter, center=center, cutoff=cutoff)
        self.enable()

    def get_all_data(self):
        return self.remover.data_loader.init_data

    def get_rate(self):
        return self.remover.data_loader.data_rate

    def get_channels(self):
        return self.remover.data_loader.channel_names

    def get_data(self, epochs, channel):
        pass

    def update_frame(self, frame_number):
        for option in [self.notch_options, self.svd_options]:
            option.update_frame(frame_number)

    def get_cleaned_data(self, epochs=None, channel=None):
        data = self.remover.solution.output
        if epochs is not None:
            data = data[epochs, :, :]
        if channel is not None:
            data = data[:, channel, :]
        return data

    def update_filter(self, name):
        self.current_filter = name
        self.process_widgets.setCurrentIndex(self.filter_list.index(name))

    def get_current_config(self, idx=None):
        filter = self.notch_options if self.current_filter == "notch" else self.svd_options
        return filter.get_args_by_idx(idx)
    
    def disable(self):
        self.svd_options.disable()
        self.notch_options.disable()

    def enable(self):
        self.svd_options.enable()
        self.notch_options.enable()