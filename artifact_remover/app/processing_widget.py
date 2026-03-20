import json

from PyQt5.QtWidgets import QWidget, QSplitter, QVBoxLayout, QHBoxLayout, QMessageBox
from PyQt5.QtCore import Qt

import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d
from .remover_widget import Remover
from .display_options import DisplayWidget
from .plot_widget import Plotter
from .utils import ensure_list


class ProcessingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.remover_options = Remover(self)
        self.display_options = DisplayWidget(self)
        self.display_options.disable()
        self.remover_options.disable()
        self.plot = Plotter(self)
        self._init_layout()
        self.clean_notch = None
        self.clean_svd = None
        self.process_file_path = None

    def _init_layout(self):
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.remover_options.process_widgets)
        right_layout.addWidget(self.display_options)
        right_panel.setLayout(right_layout)
        right_panel.setMaximumWidth(500)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.plot)
        splitter.addWidget(right_panel)
        self.setLayout(QHBoxLayout())
        self.layout().addWidget(splitter)

    def update_filter(self, name="notch"):
        self.remover_options.update_filter(name)
        self.plot.update_filter(name)
        self.plot.update_config_button(self.remover_options.get_current_config())

    def update_frame(self, frame_number):
        self.plot.update_frame(frame_number, update_time=True)
        self.plot.update_config_button(self.remover_options.get_current_config())
        self.remover_options.update_frame(frame_number)

    def process(self, **kwargs):
        kwargs["batch_idxs"] = ensure_list(self.display_options.frame_number)
        self.parent.log_box.log("Processing data...")
        self.remover_options.remover.process(**kwargs)
        self.update_processed_plot(kwargs["batch_idxs"], kwargs["channel_idxs"])
        self.parent.log_box.log("Data processing done!")
        self.parent.saved_ok = False

    def set_file(self, file_list, process_data_file=None):
        self.file_path = file_list
        self.remover_options.set_file(file_list)
        if self.remover_options.remover is None:
            self.file_path = None
            return
        clean_svd, clean_notch = None, None
        if process_data_file is not None:
            clean_svd, clean_notch = self.get_process_data(process_data_file)
        # data_resampled = self._resample_data(self.remover_options.get_all_data(), self.remover_options.get_rate())
        # self.remover_options.remover.data_loader.init_data = data_resampled
        # self.remover_options.remover.data_loader.data_rate = 2000
        self.plot.initialize_data(
            self.remover_options.get_all_data(),
            self.remover_options.get_channels(),
            self.remover_options.remover.data_loader.time,
            cleaned_notch=clean_notch,
            cleaned_svd=clean_svd
        )
        self.display_options.set_file_params(
            self.remover_options.get_channels(), self.remover_options.get_all_data().shape[0]
        )
        self.parent.toolbar.radio_svd_filter_button.setEnabled(True)
        self.parent.toolbar.radio_notch_filter_button.setEnabled(False)
        self.update_filter("notch")

    def _resample_data(self, data, rate, target=2000):
        if rate != target:
            x_new = np.linspace(0, data.shape[-1] / rate, int(data.shape[-1] * target / rate))
            x_old = np.linspace(0, data.shape[-1] / rate, data.shape[-1])
            f_interp = interp1d(x_old, data, axis=-1)
            return f_interp(x_new)
            # return resample(data, int(data.shape[-1] * target / self.remover_options.get_rate()), axis=-1)
        return

    def update_processed_plot(self, batch_idxs, channel_idxs):
        cleaned_data = self.remover_options.get_cleaned_data()
        self.plot.update_data(
            cleaned_data, ensure_list(channel_idxs), ensure_list(batch_idxs), data_type="clean", auto_range=False
        )
        self.plot.enable_config_button(channel_idxs)

    def save_file(self, path):
        self.process_file_path = path
        dic_to_save = {
            "raw_data": self.plot.raw_data,
            "processed_data_svd": self.plot.clean_svd,
            "processed_data_notch": self.plot.clean_notch,
            "channels": self.remover_options.get_channels(),
            "rate": self.remover_options.get_rate(),
        }
        sio.savemat(path, dic_to_save)
        self.parent.saved_ok = True

    def load_config(self, path):
        if path == "":
            return
        with open(path, "r") as f:
            config_data = json.load(f)
        self.set_file(config_data["file_path"], config_data["process_file_path"])
        self.remover_options.svd_options.load_config(config_data["filters_params_svd"])
        self.remover_options.notch_options.load_config(config_data["filters_params_notch"])

    def get_process_data(self, file_path):
        if file_path is None:
            return
        data = sio.loadmat(file_path)
        return data["processed_data_svd"], data["processed_data_notch"]

    def save_config(self, path):
        config = {
            "file_path": self.file_path,
            "process_file_path": self.process_file_path,
            "preprocessing_params": self.remover_options.remover.data_loader.filtering_params,
            "filters_params_svd": self.remover_options.svd_options.process_arguments,
            "filters_params_notch": self.remover_options.notch_options.process_arguments,
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=4)

    def show_config(self, idx):
        config = self.remover_options.get_current_config(idx)
        self.popup_info(config)

    def popup_info(self, config):
        text = ""
        if config is None: 
            return
        for name, value in config.items():
            text += name + ": " + str(value) + "\n"
        wind = QMessageBox()
        wind.setWindowTitle("Filter configuration")
        wind.setText(text)
        wind.exec_()

    def update_mouse_pos(self, pos):
        self.display_options.update_mouse_pos(pos)
