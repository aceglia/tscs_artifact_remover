from PyQt5.QtWidgets import QPushButton, QLabel, QLineEdit, QGridLayout, QWidget, QCheckBox
from PyQt5.QtCore import Qt

from .gui_utils import ChannelSelecter


class DisplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self._init_layout()
        self.frame_number = 0
        self.channels = []
        self.channel_selecter = None
        self.file_list = []
        self.n_frames = 0
        self.channels_to_draw = []
        self.draw_raw = True
        self.draw_clean = True
        self.draw_fft = False

    def _init_layout(self):
        layout = QGridLayout()
        self.prev_frame = QPushButton("Previous")
        self.next_frame = QPushButton("Next")
        self.prev_frame.clicked.connect(self.on_prev_frame_clicked)
        self.next_frame.clicked.connect(self.on_next_frame_clicked)
        self.input_frame = QLineEdit()
        self.input_frame.setText("1")
        self.input_frame.textChanged.connect(self.on_frame_changed)
        self.popup_button = QPushButton("Select channels to show")
        self.popup_button.clicked.connect(self.on_popup_button_clicked)

        self.display_processed_btn = QPushButton("Show processed")
        self.display_processed_btn.clicked.connect(self.on_display_processed)
        self.display_processed_btn.setEnabled(False)

        self.display_all_btn = QPushButton("Show all")
        self.display_all_btn.clicked.connect(self.on_display_all)

        self.draw_raw_button = QCheckBox("Raw")
        self.draw_raw_button.setChecked(True)
        self.draw_raw_button.stateChanged.connect(self.on_draw_raw_clicked)
        self.draw_clean_button = QCheckBox("Processed")
        self.draw_clean_button.setChecked(True)
        self.draw_clean_button.stateChanged.connect(self.on_draw_clean_clicked)
        self.show_fft_button = QCheckBox("Show FFT")
        self.show_fft_button.stateChanged.connect(self.on_draw_fft_clicked)
        self.cursor_pos = QLabel("Cursor position: x= ,y= ")
        layout.addWidget(QLabel("<b><font size=5>Display options</font></b>"), 0, 0, 1, 4, Qt.AlignCenter)
        layout.addWidget(QLabel("Frame:"), 3, 0, 1, 1)
        layout.addWidget(self.prev_frame, 3, 1, 1, 1)
        layout.addWidget(self.next_frame, 3, 2, 1, 1)
        layout.addWidget(self.input_frame, 3, 3, 1, 1)
        layout.addWidget(self.display_processed_btn, 4, 3, 1, 1)
        layout.addWidget(self.display_all_btn, 4, 2, 1, 1)
        layout.addWidget(self.popup_button, 4, 0, 1, 2)
        layout.addWidget(self.draw_raw_button, 5, 0, 1, 1)
        layout.addWidget(self.draw_clean_button, 5, 1, 1, 1)
        layout.addWidget(self.show_fft_button, 5, 2, 1, 1)
        layout.addWidget(self.cursor_pos, 6, 0, 1, 4)

        layout.setAlignment(Qt.AlignTop) 
        self.setLayout(layout)

    def on_prev_frame_clicked(self):
        self._update_frame_number("prev")
        self.on_frame_changed()

    def on_next_frame_clicked(self):
        self._update_frame_number("next")
        self.on_frame_changed()

    def on_frame_changed(self, text=None):
        if text == '':
            return
        if text:
            self._update_frame_number(value=text)
        self.input_frame.setText(str(self.frame_number + 1))
        self.parent.update_frame(self.frame_number)
        processed = self.parent.get_processed_channels()
        if processed is not None:
            self.display_processed_btn.setEnabled(True)
        else:
            self.display_processed_btn.setEnabled(False)

    def on_popup_button_clicked(self):
        if self.channel_selecter is None:
            self.channel_selecter = ChannelSelecter(self, self.channels)
        self.channel_selecter.show()

    def on_display_processed(self):
        channels = self.parent.get_processed_channels()
        self.channel_selecter.set_channels(channels)
        self.on_draw_clicked()

    def on_display_all(self):
        channels = list(range(len(self.channels)))
        self.channel_selecter.set_channels(channels)
        self.on_draw_clicked()

    def on_draw_clicked(self):
        self.channels_to_draw = self.channel_selecter.get_selected_channels()
        self.parent.plot.update_channels(self.channels_to_draw)

    def _update_frame_number(self, direction=None, value=None):
        if direction == "prev":
            self.frame_number -= 1
        elif direction == "next":
            self.frame_number += 1
        elif value is not None:
            self.frame_number = int(value) - 1 
        self.frame_number = min(self.frame_number, self.n_frames-1)
        self.frame_number = max(self.frame_number, 0)
        return self.frame_number

    def set_file_params(self, channels, n_frames):
        self.channels = channels
        self.n_frames = n_frames
        self.channel_selecter = ChannelSelecter(self, self.channels)
        self._reset()
        self.enable()

    def _reset(self):
        self._update_frame_number('1')
        self.draw_raw_button.setChecked(True)
        self.draw_clean_button.setChecked(True)
        self.show_fft_button.setChecked(False)
        self.update_draw_params()

    def on_draw_raw_clicked(self):
        self.draw_raw = self.draw_raw_button.isChecked()
        self.update_draw_params()

    def on_draw_clean_clicked(self):
        self.draw_clean = self.draw_clean_button.isChecked()
        self.update_draw_params()
    
    def on_draw_fft_clicked(self):
        self.draw_fft = self.show_fft_button.isChecked()
        self.update_draw_params()
    
    def update_draw_params(self):
        self.parent.plot.update_draw_params(self.draw_raw, self.draw_clean, self.draw_fft)

    def update_mouse_pos(self, pos):
        self.cursor_pos.setText(f"Cursor position: x={pos[0]}, y={pos[1]}")

    def disable(self):
        for item in self.findChildren(QWidget):
            item.setEnabled(False)
        
    def enable(self):
        for item in self.findChildren(QWidget):
            item.setEnabled(True)
        self.display_processed_btn.setEnabled(False)
        


