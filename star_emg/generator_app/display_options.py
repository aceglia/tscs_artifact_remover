from PyQt5.QtWidgets import QLabel, QGridLayout, QWidget, QCheckBox
from PyQt5.QtCore import Qt


class DisplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.phase_inversion = False
        self.draw_fft = False

        self._init_layout()

    def _init_layout(self):
        layout = QGridLayout()
        self.show_fft_button = QCheckBox("Show FFT")
        self.show_fft_button.setChecked(False)
        self.show_fft_button.stateChanged.connect(self.on_draw_fft_clicked)
        self.cursor_pos = QLabel("Cursor position: x= ,y= ")
        self.inverse_phase_btn = QCheckBox("Apply phase inversion")
        self.inverse_phase_btn.stateChanged.connect(self.inverse_phase)
        self.inverse_phase_btn.setChecked(False)
        layout.addWidget(self.inverse_phase_btn, 0, 0, 1, 1)
        layout.addWidget(self.show_fft_button, 1, 0, 1, 1)
        layout.addWidget(self.cursor_pos, 2, 0, 1, 1)

        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

    def on_draw_fft_clicked(self):
        self.draw_fft = self.show_fft_button.isChecked()
        self.update_draw_params()

    def inverse_phase(self, state):
        self.phase_inversion = state != 0
        self.parent.update_template()

    def update_draw_params(self):
        self.parent.plot.update_draw_params(self.draw_fft)

    def update_mouse_pos(self, pos):
        self.cursor_pos.setText(f"Cursor position: x={pos[0]}, y={pos[1]}")

    def disable(self):
        for item in self.findChildren(QWidget):
            item.setEnabled(False)

    def enable(self):
        for item in self.findChildren(QWidget):
            item.setEnabled(True)
