from functools import partial

import numpy as np

from PyQt5.QtWidgets import (
    QWidget,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QPlainTextEdit,
)
from PyQt5.QtCore import Qt
from ..app.gui_utils import ChannelSelecter
from .generator_utils import check_list, get_from_range


class ParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.channel_selecter = None
        self.parent = parent
        self.channels = []
        self.short_process_args = {
            "amplitude": 1,
            "delay_1": 0.1,
            "delay_2": 0.2,
            "num": [1],
            "den": [0.02, 0.5, 12],
            "T": 1,
            "factors": [1, 2, 1],
        }
        self.init_template_args = self.short_process_args.copy()
        self.init_template_args.update({"sampling_rate": 2000, "duration": 0.007, "stim_freq": 30})
        self.set_options(self.init_template_args)
        self.template_arguments = {}
        self._init_layout()

    def _init_layout(self):
        layout = QGridLayout()
        layout.addWidget(QLabel("<b><font size=5>Artifact template parameters</font></b>"), 0, 0, 1, 4, Qt.AlignCenter)

        layout.addWidget(QLabel("Transfert function\n denominator:"), 1, 0, 1, 1)
        self.input_den = [QLineEdit() for _ in range(3)]
        self.template_arguments = self.get_template_arguments()
        [self.input_den[i].setText(str(self.template_arguments["den"][i])) for i in range(3)]
        [self.input_den[i].textChanged.connect(partial(self.set_denominator, i)) for i in range(3)]
        [layout.addWidget(self.input_den[i], 1, i + 1, 1, 1) for i in range(0, 3)]

        layout.addWidget(QLabel("Step delays:"), 2, 0, 1, 1)
        self.input_delay = [QLineEdit() for _ in range(2)]
        [self.input_delay[i].setText(str(self.template_arguments[f"delay_{i+1}"])) for i in range(2)]
        [self.input_delay[i].textChanged.connect(partial(self.set_delays, i)) for i in range(2)]
        [layout.addWidget(self.input_delay[i], 2, i + 1, 1, 1) for i in range(2)]

        layout.addWidget(QLabel("Amplitude (around 1):"), 3, 0, 1, 1)
        self.input_amplitude = QLineEdit()
        self.input_amplitude.setText(str(self.template_arguments["amplitude"]))
        self.input_amplitude.textChanged.connect(self.set_amplitude)
        layout.addWidget(self.input_amplitude, 3, 1, 1, 1)

        layout.addWidget(QLabel("Sum factors (+/-/+):"), 4, 0, 1, 1)
        self.input_factors = [QLineEdit() for _ in range(3)]
        [self.input_factors[i].setText(str(self.template_arguments["factors"][i])) for i in range(3)]
        [self.input_factors[i].textChanged.connect(partial(self.set_factors, i)) for i in range(3)]
        [layout.addWidget(self.input_factors[i], 4, i + 1, 1, 1) for i in range(0, 3)]

        layout.addWidget(QLabel("Transfert function: "), 5, 0, 1, 1)
        self.transfert_text = QPlainTextEdit()
        self.transfert_text.setReadOnly(True)
        self.transfert_text.setStyleSheet("background-color: transparent;border: 0;")
        layout.addWidget(self.transfert_text, 5, 2, 1, 3)

        layout.addWidget(QLabel("Stimulation frequency:"), 6, 0, 1, 1)
        self.input_freq = QLineEdit()
        self.input_freq.setText(str(self.template_arguments["stim_freq"]))
        self.input_freq.textChanged.connect(partial(self._set_attrib, key="stim_freq"))
        layout.addWidget(self.input_freq, 6, 1, 1, 1)

        layout.addWidget(QLabel("Sampling rate:"), 7, 0, 1, 1)
        self.input_rate = QLineEdit()
        self.input_rate.setText(str(self.template_arguments["sampling_rate"]))
        self.input_rate.textChanged.connect(partial(self._set_attrib, key="sampling_rate"))
        layout.addWidget(self.input_rate, 7, 1, 1, 1)

        layout.addWidget(QLabel("Template_duration:"), 8, 0, 1, 1)
        self.input_duration = QLineEdit()
        self.input_duration.setText(str(self.template_arguments["duration"]))
        self.input_duration.textChanged.connect(self.set_duration)

        self.apply_button = QPushButton("Apply to current channel")
        self.apply_button.clicked.connect(self.on_apply_button_clicked)
        self.apply_button.setEnabled(False)

        self.process_button = QPushButton("Generate template")
        self.process_button.clicked.connect(self.update_template)

        layout.addWidget(self.input_duration, 8, 1, 1, 2)
        layout.addWidget(self.apply_button, 10, 0, 1, 2)
        layout.addWidget(self.process_button, 9, 0, 1, 2)
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

    def update_transfert_text(self):
        tf = str(self.parent.generator._get_transfert_fct([1], self.den)).split("\n")
        text = tf[-3].replace("1", str(self.amplitude)) + "\n"
        text += "\n".join(tf[-2:])
        self.transfert_text.setPlainText(text)

    def set_amplitude(self, text):
        amplitude = check_list(text)
        if amplitude is not None:
            self.amplitude = amplitude if not isinstance(amplitude, list) else get_from_range(amplitude)

    def set_delays(self, i, text):
        delay = check_list(text)
        if delay is not None:
            setattr(self, f"delay_{i+1}", delay if not isinstance(delay, list) else get_from_range(delay))

    def set_denominator(self, i, text):
        den = check_list(text)
        if den is not None:
            self.den[i] = den if not isinstance(den, list) else get_from_range(den)

    def set_factors(self, i, text):
        factors = check_list(text)
        if factors is not None:
            self.factors[i] = factors if not isinstance(factors, list) else get_from_range(factors)

    def get_raw_values(self):
        return {
            "amplitude": check_list(self.input_amplitude.text()),
            "den": [check_list(self.input_den[i].text()) for i in range(3)],
            "delay_1": check_list(self.input_delay[0].text()),
            "delay_2": check_list(self.input_delay[1].text()),
            "num": [1],
            "factors": [check_list(self.input_factors[i].text()) for i in range(3)],
            "artifact_duration": check_list(self.input_duration.text()),
        }

    def _set_attrib(self, text, key):
        if text == "":
            return
        setattr(self, key, float(text))

    def set_stim_frequency(self, value):
        self.stim_freq = float(value)

    def set_sampling_rate(self, value):
        self.sampling_rate = float(value)

    def set_duration(self, value):
        duration = check_list(value)
        if duration is not None:
            self.duration = duration if not isinstance(duration, list) else get_from_range(duration)

    def get_options(self):
        return self.__dict__

    def set_options(self, options):
        [setattr(self, key, value) for key, value in options.items()]

    def get_option(self, option):
        return getattr(self, option)

    def on_apply_button_clicked(self):
        template_arguments = self.get_template_arguments()
        self.parent.apply_template(self.get_template_arguments())
        channel = self.parent.display_options.channel_selecter.get_channels()
        self.template_arguments[channel[1]] = template_arguments.copy()

    def update_random_param(self):
        self.set_amplitude(self.input_amplitude.text())
        [self.set_delays(i, self.input_delay[i].text()) for i in range(2)]
        [self.set_denominator(i, self.input_den[i].text()) for i in range(3)]

    def update_template(self):
        self.update_random_param()
        self.update_transfert_text()
        self.parent.update_template(**self.get_short_config())

    def get_template_arguments(self, keys_item=None, value_item=None):
        args_item = self.init_template_args if keys_item is None else keys_item
        dic_item = value_item if value_item is not None else self.__dict__
        params_dict = {}
        for name, value in args_item.items():
            params_dict[name] = dic_item.get(name, args_item[name])
        return params_dict

    def load_config(self, config):
        self.template_arguments = config
        self.set_options(config)

    def get_short_config(self, value_item=None):
        return self.get_template_arguments(self.short_process_args, value_item)


class Template:
    def __init__(self, parent=None):
        self.generator = None
        self.parent = parent
        self.is_initialized = False
        self.is_results = False
        self.template = None
        self.params_widget = ParamsWidget(self.parent)

    def set_file(self, data_rate):
        self.params_widget.sampling_rate = data_rate
        self.params_widget.update_params()

    def get_all_data(self):
        return self.remover.data_loader.init_data

    def get_rate(self):
        return self.remover.data_loader.data_rate

    def get_channels(self):
        return self.remover.data_loader.channel_names

    def update_frame(self, frame_number):
        self.params_widget.update_frame(frame_number)

    def get_contaminated_data(self, epochs=None, channel=None):
        data = self.generator.output
        if epochs is not None:
            data = data[epochs, :, :]
        if channel is not None:
            data = data[:, channel, :]
        return data

    def get_current_config(self, idx=None):
        return self.params_widget.get_args_by_idx(idx)

    def disable(self):
        self.svd_options.disable()
        self.notch_options.disable()

    def enable(self):
        self.svd_options.enable()
        self.notch_options.enable()

    def get_processed_channels(self):
        config = self.get_current_config()
        if config is None:
            return
        return [i for i, conf in enumerate(config) if conf is not None]

    def get_displayed_channels(self):
        return self.parent.display_options.channel_selecter.get_channel_idxs()
