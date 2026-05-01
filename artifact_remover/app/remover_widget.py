import numpy as np
from PyQt5.QtWidgets import (
    QWidget,
    QStackedWidget,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QCheckBox,
    QPlainTextEdit,
    QMessageBox,
)
from PyQt5.QtCore import Qt

from ..automatic_remover import ArtefactRemover
from ..rt_automatic_remover import RtArtefactRemover
from .gui_utils import ChannelSelecter, ensure_list


class OptionWidget(QWidget):
    def __init__(self, type, parent=None):
        super().__init__()
        self.type = type
        self.channel_selecter = None
        self.parent = parent
        self.current_frame = 0
        self.channels = []
        self.event_log = None
        self.queue = None
        self.stream_mode = False
        self.init_process_args = {
            "notch_filter": self.type == "notch",
            "quality_factor": 80,
            "frequency_peaks": 30,
            "first_peak": None,
            "hankel_size": 500,
            "hankel_delay": 1,
            "process_window": 10000,
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

        self.process_shown_button = QPushButton("Select channels to process")
        self.process_shown_button.clicked.connect(self.on_process_shown_button)

        self.cancel_button = QPushButton("Cancel processing")
        self.cancel_button.clicked.connect(self.on_cancel_button_clicked)

        self.text_config = QPlainTextEdit()
        self.text_config.setReadOnly(True)
        self.text_config.setStyleSheet("background-color: transparent;border: 0;")

        # self.wind_to_proc_label = QLabel('Chose which part of the data to process.\n' \
        # '1) If nothing is writen it will process the whole data.' \
        # '2) If you want to process only a part of the data, you can write the start and end frame separated by a commainside brackets (e.g. [200, 1200]).\n' \
        # 'You can put as many brackets you want to process different parts of the data, if brackets overlap (e.g. [200, 400], [500, 700], ...).\n' \
        # '3) If you want to process repetitive parts of the data you can write the delay between the start of the windows (in ms), the first frame of the first windows and the windows lenght (e.g. [20, 0, 1000]).\n' \
        # 'The windows will be stack to be process all together by the same parameters.')
        # self.window_to_process_input = QLabel()
        # self.window_to_process_input.textChanged.connect(self.update_data_windows)

    def init(self, channels, frames, streaming=False, event_log=None, queue=None):
        self.channels = channels
        self.n_frames = frames
        self.streaming = streaming
        self.event_log = event_log
        self.queue = queue
        # self.channel_selecter = ChannelSelecter(self, self.channels, for_display=False)
        if self.type == "svd":
            self.set_automatic_hsize(Qt.Checked)

    def get_options(self):
        return self.__dict__

    def set_options(self, **options):
        for key, value in options.items():
            setattr(self, key, value)

    def get_option(self, option):
        return getattr(self, option)

    def on_process_shown_button(self):
        channels = self.parent.get_displayed_channels()
        self.channel_selecter.set_channels(channels)
        self.on_draw_clicked()

    def on_popup_button_clicked(self):
        if self.channel_selecter is None:
            self.channel_selecter = ChannelSelecter(self, self.channels, for_display=False)
        self.channel_selecter.show()

    def on_draw_clicked(self):
        idxs = self.channel_selecter.get_channel_idxs()
        if len(idxs) == 0:
            QMessageBox.warning(None, "No channel selected", "Please select at least one channel to process.")
            return
        self.channel_idxs = idxs
        self.process_button.setEnabled(True)
        self.channel_selecter.quit()

    def update_params(self):
        process_arguments = self.get_process_arguments()
        # if not self._check_config(process_arguments):
        #     self.parent.log_box.log("WARNING: Invalid configuration. Please check the parameters.")
        #     return
        if len(ensure_list(process_arguments["channel_idxs"])) == 0:
            self.parent.log_box.log("Unable to process data. Please select a channel to process first.")
            # self.process_button.setEnabled(False)
        if not self.streaming:
            self.parent.process(**process_arguments)
            
        list_empty = (
            [None] * len(self.channels)
            if f"Frame_{self.current_frame}" not in self.process_arguments
            else self.process_arguments[f"Frame_{self.current_frame}"]
        )
        channel_names = self.channel_selecter.get_channel_names()
        self.channel_idxs = self.channel_selecter.get_channel_idxs()
        for c, ch in enumerate(self.channel_idxs):
            process_arguments["channel_name"] = channel_names[c]
            list_empty[ch] = process_arguments.copy()
        self.process_arguments[f"Frame_{self.current_frame}"] = list_empty

    def update_frame(self, frame_number = 0):
        self.current_frame = frame_number

    @staticmethod
    def _check_empty(value):
        return None if value == "" else value

    def get_process_arguments(self, keys_item=None, value_item=None):
        args_item = self.init_process_args if keys_item is None else keys_item
        dic_item = value_item if value_item is not None else self.__dict__
        params_dict = {}
        for name, value in args_item.items():
            if name == "hankel_size" and self.type == "svd":
                params_dict[name] = self.hankel_size
                continue
            params_dict[name] = self._check_empty(dic_item.get(name, args_item[name]))
        return params_dict

    def load_config(self, config):
        self.process_arguments = config

    def get_short_config(self, value_item=None):
        return self.get_process_arguments(self.short_process_args, value_item)

    def get_args_by_idx(self, idx=None):
        if f"Frame_{self.current_frame}" not in self.process_arguments:
            return
        if idx is not None:
            process_argument = self.process_arguments[f"Frame_{self.current_frame}"][idx]
            if process_argument is None:
                return
            return self.get_short_config(value_item=process_argument)
        else:
            return self.process_arguments[f"Frame_{self.current_frame}"]

    def on_cancel_button_clicked(self):
        self.parent.cancel_processing()

    def disable(self):
        for item in self.findChildren(QWidget):
            item.setEnabled(False)

    def enable(self, all=False):
        for item in self.findChildren(QWidget):
            item.setEnabled(True)
        if not all:
            self.process_button.setEnabled(False)
        if self.type == "svd":
            self.input_hankel.setEnabled(not self.automatic_hsize)

    # def update_data_windows(self, text):
    #     windows = check_list(text)
    def show_config(self, text):
        self.text_config.setPlainText(text)

    def _check_config(self):
        return True

class NotchOptions(OptionWidget):
    def __init__(self, parent=None):
        super().__init__("notch", parent)
        self.process_window = 10000
        self.quality_factor = 80
        self.frequency_peaks = 30
        self.params_changed = True
        self.first_peak = ""
        self.init_process_args["process_window"] = self.process_window
        self._init_layout()
        self.short_process_args = {
            "quality_factor": 80,
            "frequency_peaks": 30,
            "process_window": self.process_window,
            "first_peak": None,
        }

    def _init_layout(self):
        layout = QGridLayout()
        layout.addWidget(QLabel("<b><font size=5>Notch Remover Options</font></b>"), 0, 0, 1, 2, Qt.AlignCenter)
        layout.addWidget(QLabel("Quality Factor:"), 1, 0, 1, 1)
        self.input_quality = QLineEdit()
        self.input_quality.setText(str(self.quality_factor))
        self.input_quality.textChanged.connect(self.set_quality_factor)
        layout.addWidget(self.input_quality, 1, 1, 1, 1)

        layout.addWidget(QLabel("Stimulation frequency:"), 2, 0, 1, 1)
        self.input_freq = QLineEdit()
        self.input_freq.setText(str(self.frequency_peaks))
        self.input_freq.textChanged.connect(self.set_frequency_peaks)
        layout.addWidget(self.input_freq, 2, 1, 1, 1)

        layout.addWidget(QLabel("First peak frequency:"), 3, 0, 1, 1)
        self.input_first = QLineEdit()
        self.input_first.setText(str(self.first_peak))
        self.input_first.textChanged.connect(self.set_first_peak)
        layout.addWidget(self.input_first, 3, 1, 1, 1)

        self.input_wind = QLineEdit()
        self.input_wind.setText(str(self.process_window))
        self.input_wind.textChanged.connect(self.set_process_window)
        layout.addWidget(QLabel("Process window lenght:"), 4, 0, 1, 1)
        layout.addWidget(self.input_wind, 5, 1, 1, 2)
        # layout.addWidget(self.wind_to_proc_label, 6, 0, 1, 2)
        # layout.addWidget(self.window_to_process_input, 6, 0, 1, 2)
        layout.addWidget(self.popup_button, 6, 0, 1, 2)
        # layout.addWidget(self.process_shown_button, 5, 1, 1, 1)
        layout.addWidget(self.process_button, 7, 0, 1, 2)
        layout.addWidget(self.text_config, 8, 0, 1, 2)
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

    def set_first_peak(self, text):
        if text == "":
            return
        self.first_peak = float(text)

    def set_process_window(self, text):
        if text == "":
            return
        self.process_window = int(text)


class SVDOptions(OptionWidget):
    def __init__(self, parent=None):
        super().__init__("svd", parent)
        self.hankel_delay = 1
        self._hankel_size = 500
        self.process_window = 5000
        self.overlap = 0
        self.nb_principal_components = None
        self.factor = 0.35
        self.freq_bounds = [10, 300]
        self.automatic_hsize = True
        self.init_process_args["process_window"] = self.process_window
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

        self.input_wind = QLineEdit()
        self.input_wind.setText(str(self.process_window))
        self.input_wind.textChanged.connect(self.set_process_window)
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

        layout.addWidget(QLabel("Hankel delay:"), 3, 0, 1, 1)
        self.input_delay = QLineEdit()
        self.input_delay.setText(str(self.hankel_delay))
        self.input_delay.textChanged.connect(self.set_hankel_delay)
        layout.addWidget(self.input_delay, 3, 1, 1, 1)


        layout.addWidget(QLabel("Select threshold:"), 4, 0, 1, 2)
        self.input_factor = QLineEdit()
        self.input_factor.setText(str(self.factor))
        self.input_factor.textChanged.connect(self.set_factor)
        layout.addWidget(self.input_factor, 4, 1, 1, 1)

        layout.addWidget(QLabel("Select frequency bounds:"), 5, 0, 1, 1)
        self.input_low_freq = QLineEdit()
        self.input_low_freq.setText(str(self.freq_bounds[0]))
        self.input_low_freq.textChanged.connect(self.set_low_freq)
        layout.addWidget(self.input_low_freq, 5, 1, 1, 1)

        self.input_high_freq = QLineEdit()
        self.input_high_freq.setText(str(self.freq_bounds[1]))
        self.input_high_freq.textChanged.connect(self.set_high_freq)
        layout.addWidget(self.input_high_freq, 5, 2, 1, 1)

        # layout.addWidget(self.wind_to_proc_label, 5, 0, 1, 2)
        # layout.addWidget(self.window_to_process_input, 6, 0, 1, 2)
        layout.addWidget(self.popup_button, 6, 0, 1, 3)
        layout.addWidget(self.process_button, 7, 0, 1, 3)
        layout.addWidget(self.text_config, 8, 0, 1, 3)

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

    def set_hankel_delay(self, text):
        if text == "":
            return
        self.hankel_delay = int(text)
        if self.automatic_hsize:
            self.set_automatic_hsize(Qt.Checked)

    def set_automatic_hsize(self, state, update=True):
        self.automatic_hsize = state == Qt.Checked
        self.input_hankel.setEnabled(not self.automatic_hsize)
        self.input_hankel.setText(str(self.hankel_size_from_window()))

    def hankel_size_from_window(self, factor=8):
        self.process_window = int(self.input_wind.text())
        return max(int((self.process_window / factor) / self.hankel_delay), 1)

    @property
    def hankel_size(self):
        if self.automatic_hsize:
            return self.hankel_size_from_window()
        return self._hankel_size
    
    def _check_config(self, process_arguments):
        return True


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

    def update_filter(self, name):
        self.current_filter = name
        self.process_widgets.setCurrentIndex(self.filter_list.index(name))
        filter = self.svd_options if self.current_filter == "svd" else self.notch_options
        if filter.channel_selecter is None:
            filter.process_button.setEnabled(False)

    def get_current_config(self, idx=None):
        filter = self.notch_options if self.current_filter == "notch" else self.svd_options
        return filter.get_args_by_idx(idx)

    def disable(self):
        self.svd_options.disable()
        self.notch_options.disable()

    def enable(self, all=False):
        self.svd_options.enable(all)
        self.notch_options.enable(all)

    def get_processed_channels(self):
        config = self.get_current_config()
        if config is None:
            return
        return [i for i, conf in enumerate(config) if conf is not None]

    def get_displayed_channels(self):
        return self.parent.display_options.channel_selecter.get_channel_idxs()

    def show_config(self, text):
        options = self.notch_options if self.current_filter == "notch" else self.svd_options
        options.show_config(text)

    def get_process_window(self):
        filter = self.svd_options if self.current_filter == "svd" else self.notch_options
        return filter.process_window


class OfflineRemover(Remover):
    def __init__(self, parent=None):
        super().__init__(parent)

    def _init_remover(self, path_file, **kwargs):
        self.remover = ArtefactRemover(data=path_file, **kwargs)
        data_shape = self.remover.data_loader.init_data.shape
        window = 25_000
        if data_shape[-1] > window and data_shape[0] == 1:
            total_epochs = int(np.ceil(data_shape[-1] / 25000))
            rate = self.remover.data_loader.data_rate
            epochs_duration = window * (1 / rate)
            self.parent.parent.show_split_windows(
                f"The data has {data_shape[-1]} points per channel. This may slowdown the programm."
                f"If you chose 'yes' your file will be splitted in {total_epochs} of {epochs_duration:.2f}s duration Frames. If the last frame is not full, it will be filled with 0."
            )
        if self.parent.parent._split == "&Yes":
            self._split_data(window, total_epochs, rate)
        elif self.parent.parent._split == "Cancel":
            self.remover = None

        for options in [self.notch_options, self.svd_options]:
            options.init(self.remover.data_loader.channel_names, self.remover.data_loader.init_data.shape[0])

    def _split_data(self, window, epochs, rate):
        data = np.swapaxes(self.remover.data_loader.init_data, 0, 1)
        full_mat = np.zeros((data.shape[0], 1, int(epochs * window)))
        full_mat[..., : data.shape[-1]] = data
        full_mat = full_mat.reshape(data.shape[0], epochs, window)
        self.remover.data_loader.init_data = np.swapaxes(full_mat, 0, 1)
        time = np.linspace(0, int(epochs * window) / rate, int(epochs * window))
        self.remover.data_loader.time = time.reshape((epochs, window))

    def set_file(self, file_path, signal_filter=False, center=True, cutoff=[10, 450], order=2):
        self.process_widgets.setCurrentIndex(0)
        self.current_filter = "notch"
        # for options in [self.notch_options, self.svd_options]:
        #     options.__init__(self.parent)
        self._init_remover(file_path, signal_filter=signal_filter, center=center, cutoff=cutoff, order=order)
        self.enable()

    def get_all_data(self):
        return self.remover.data_loader.init_data

    def get_rate(self):
        return self.remover.data_loader.data_rate

    def get_channels(self):
        return self.remover.data_loader.channel_names

    def get_data(self, epochs=None, channel=None):
        data = self.remover.data_loader.init_data
        if epochs is not None:
            data = data[epochs, :, :]
        if channel is not None:
            data = data[:, channel, :]
        return data

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


class StreamRemover(Remover):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._adjust_value_for_stream()
        self.remover = None
        
    def _adjust_value_for_stream(self):
        self.svd_options.init_process_args["process_window"] = 500
        self.svd_options.init_process_args["hankel_size"] = 100
        self.notch_options.init_process_args["process_window"] = 2000
        self.svd_options.input_hankel.setText(str(self.svd_options.init_process_args["hankel_size"]))
        self.svd_options.input_wind.setText(str(self.svd_options.init_process_args["process_window"]))
        self.notch_options.input_wind.setText(str(self.notch_options.init_process_args["process_window"]))

    def new_stream(self, channels, events, queue_args):
        self.process_widgets.setCurrentIndex(0)
        self.current_filter = "notch"
        self.channels = channels
        self.remover = RtArtefactRemover()
        self.enable()
        for options in [self.notch_options, self.svd_options]:
            options.init(channels, len(channels), streaming=True, events_log=events, queue=queue_args)
        
