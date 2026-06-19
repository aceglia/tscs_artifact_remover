import json

from PyQt5.QtWidgets import QWidget, QSplitter, QVBoxLayout, QHBoxLayout, QMessageBox
from PyQt5.QtCore import Qt

import numpy as np
import scipy.io as sio
from .display_options import DisplayWidget
from .plot_widget import Plotter
from .template_widget import Template
from .generator_utils import check_list
from ..io_utils import export_csv
from ..generator import ArtifactGenerator


class ArtifactWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.generator = ArtifactGenerator()
        self.data_shape = 10000
        self.template_options = Template(self)
        self.display_options = DisplayWidget(self)
        self.plot = Plotter(self)
        self._init_layout()
        self.process_file_path = None

    def _init_layout(self):
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.template_options.params_widget)
        right_layout.addWidget(self.display_options)
        right_panel.setLayout(right_layout)
        right_panel.setMaximumWidth(500)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.plot)
        splitter.addWidget(right_panel)
        self.setLayout(QHBoxLayout())
        self.layout().addWidget(splitter)
        self._init_template()

    def _init_template(self):
        self.update_template(**self.template_options.params_widget.get_short_config())
        self.template_options.params_widget.update_transfert_text()

    def update_template(self, **kwargs):
        self.template = self.generator._get_biphasic_response_template(**kwargs)
        self.sampled_template = self.generator._sample_to_frequency(
            self.template,
            duration=self.template_options.params_widget.duration,
            fs=self.template_options.params_widget.sampling_rate,
        )
        self.plot.update_template(self.template, self.sampled_template)
        self.train_artifact = self.generator.generate_artifact(
            stimulation_frequency=self.template_options.params_widget.stim_freq,
            sampling_rate=self.template_options.params_widget.sampling_rate,
            phase_inversion=self.display_options.phase_inversion,
            output_shape=self.data_shape,
            **self.template_options.params_widget.get_raw_values(),
        )
        train = (
            self.apply_white_noise(self.train_artifact)
            if self.display_options.white_noise_btn.isChecked()
            else self.train_artifact
        )
        self.plot.update_stim_train(train)

    def apply_white_noise(self, data):
        noise = np.random.normal(0, 1, data.shape) * 0.1
        return data + noise

    def add_white_noise(self):
        self.update_template(**self.template_options.params_widget.get_short_config())

    def save_file(self, path):
        self.process_file_path = path
        dic_to_save = {
            "raw_data": self.plot.raw_data,
            "processed_data_svd": self.plot.clean_svd,
            "processed_data_notch": self.plot.clean_notch,
            "channels": self.template_options.get_channels(),
            "rate": self.template_options.get_rate(),
        }
        sio.savemat(path, dic_to_save)
        export_csv(path, **dic_to_save)
        self.parent.log_box.log(
            "To use the processed file in signal you can import the txt file saved at " + path.replace(".mat", ".txt")
        )
        self.parent.set_saved_ok(True)

    def load_config(self, path):
        if path == "":
            return
        with open(path, "r") as f:
            config_data = json.load(f)
        self.set_file(config_data["file_path"], config_data["process_file_path"])
        self.template_options.load_config(config_data["template_config"])

    def save_config(self, path):
        config = {
            "file_path": self.file_path,
            "process_file_path": self.process_file_path,
            "preprocessing_params": self.template_options.remover.data_loader.filtering_params,
            "filters_params_svd": self.template_options.svd_options.process_arguments,
            "filters_params_notch": self.template_options.notch_options.process_arguments,
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=4)

    def update_mouse_pos(self, pos):
        self.display_options.update_mouse_pos(pos)
