import json

from PyQt5.QtWidgets import QWidget, QSplitter, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt

from .display_options import DisplayWidget
from .plot_widget import Plotter
from .template_widget import Template
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
        self.plot.update_stim_train(self.train_artifact)

    def load_config(self, path):
        if path == "":
            return
        with open(path, "r") as f:
            config_data = json.load(f)
        self.template_options.load_config(config_data)

    def save_config(self, path):
        if path == "":
            return
        options = self.template_options.get_config()
        with open(path, "w") as f:
            json.dump(options, f, indent=4)

    def update_mouse_pos(self, pos):
        self.display_options.update_mouse_pos(pos)
