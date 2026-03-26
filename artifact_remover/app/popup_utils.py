from PyQt5.QtWidgets import QMessageBox, QDialog, QGridLayout, QLabel, QLineEdit, QCheckBox, QPushButton


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
            'center': self.do_center_input.isChecked()
        }
    

