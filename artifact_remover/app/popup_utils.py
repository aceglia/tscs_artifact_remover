from PyQt5.QtWidgets import QMessageBox, QDialog, QGridLayout, QLabel, QLineEdit, QCheckBox, QPushButton, QTableWidget, QTableWidgetItem


def popup_warning_save(text, title, fct):
    wind = QMessageBox()
    wind.setText(text)
    wind.setWindowTitle(title)
    wind.setIcon(QMessageBox.Question)
    wind.setStandardButtons(QMessageBox.Save | QMessageBox.Ignore | QMessageBox.Cancel)
    wind.setDefaultButton(QMessageBox.Save)
    wind.buttonClicked.connect(fct)
    wind.exec_()


def popup_warning_continue(text, title, fct):
    wind = QMessageBox()
    wind.setText(text)
    wind.setWindowTitle(title)
    wind.setIcon(QMessageBox.Question)
    wind.setStandardButtons(QMessageBox.Ignore | QMessageBox.Cancel)
    wind.setDefaultButton(QMessageBox.Cancel)
    wind.buttonClicked.connect(fct)
    wind.exec_()


def popup_warning_split(text, title, fct):
    wind = QMessageBox()
    wind.setText(text)
    wind.setWindowTitle(title)
    wind.setIcon(QMessageBox.Question)
    wind.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
    wind.setDefaultButton(QMessageBox.Yes)
    wind.buttonClicked.connect(fct)
    wind.exec_()


def save_popup(text, fct):

    pass


class FilterDialog(QDialog):
    def __init__(self, parent=None, fs=None):
        super().__init__(parent)
        self.setWindowTitle("Data pre-filtering")
        layout = QGridLayout()
        self.do_center_input = QCheckBox("Center signal")
        self.do_center_input.setChecked(True)
        layout.addWidget(self.do_center_input, 0, 0, 1, 3)
        self.do_filtering_input = QCheckBox("Zero-lag band pass filter")
        self.do_filtering_input.setChecked(True)
        self.do_filtering_input.stateChanged.connect(self.enable_filtering_params)
        layout.addWidget(self.do_filtering_input, 1, 0, 1, 3)
        label = QLabel("Cut-off frequencies (Low - High) (Hz):")
        self.low_cut = QLineEdit("10")
        layout.addWidget(label, 2, 0, 1, 1)
        layout.addWidget(self.low_cut, 2, 1, 1, 1)
        self.high_cut = QLineEdit("450")
        layout.addWidget(self.high_cut, 2, 2, 1, 1)
        self.order = QLineEdit("4")
        layout.addWidget(QLabel("Filter order:"), 3, 0, 1, 1)
        layout.addWidget(self.order, 3, 1, 1, 1)
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.ok_button, 4, 0, 1, 1)
        layout.addWidget(self.cancel_button, 4, 2, 1, 1)

        self.setLayout(layout)

    def enable_filtering_params(self, state):
        for widget in [self.low_cut, self.high_cut, self.order]:
            widget.setEnabled(state != 0)

    def get_filter_params(self):
        return {
            "cutoff": [float(self.low_cut.text()), float(self.high_cut.text())],
            "order": max(int(self.order.text()) // 2, 1),
            "signal_filter": self.do_filtering_input.isChecked(),
            "center": self.do_center_input.isChecked(),
        }


class ChannelsPopup(QDialog):
    def __init__(self, channels=None):
        super().__init__()
        self.setWindowTitle("Channels configuration")
        self._channels = channels
        self._create_layout()
        if channels is not None:
            self._init_channels(channels)

    def _create_layout(self):
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(1)
        self.table_widget.setHorizontalHeaderLabels(['Channel name'])
        self.add_device_button = QPushButton("Add channel")
        self.add_device_button.clicked.connect(self._on_add_channel)
        self.remove_device_button = QPushButton("Remove channel")
        self.remove_device_button.clicked.connect(self._remove_row)
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        self.layout = QGridLayout()
        self.layout.addWidget(self.table_widget, 0, 0, 1, 2)
        self.layout.addWidget(self.add_device_button, 1, 0)
        self.layout.addWidget(self.remove_device_button, 1, 1)
        self.layout.addWidget(self.ok_button, 2, 0)
        self.layout.addWidget(self.cancel_button, 2, 1)
        self.setLayout(self.layout)

    def _init_channels(self, channels):
        for channel in channels:
            self._add_channel(channel)

    def _on_add_channel(self):
        self._add_channel()

    def _add_channel(self, channel=None):
        row_index = self.table_widget.rowCount()
        self.table_widget.insertRow(row_index)
        name = f"Channel {row_index}" if (channel is None or channel == '') else channel
        self.table_widget.setItem(row_index, 0, QTableWidgetItem(name))

    def _remove_row(self):
        row_index = self.table_widget.currentRow()
        self.table_widget.removeRow(row_index)

    def get_channels(self):
        channels = []
        for i in range(self.table_widget.rowCount()):
            item = self.table_widget.item(i, 0)
            if item is not None:
                channels.append(item.text())
        return channels

    @property
    def channels(self):
        return self.get_channels()
    
