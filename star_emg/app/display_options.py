from PyQt5.QtWidgets import QPushButton, QLabel, QLineEdit, QGridLayout, QWidget, QCheckBox
from PyQt5.QtCore import Qt

from .gui_utils import ChannelSelecter


class DisplayWidget(QWidget):
    """
    Parent widget to chose the display options for the plot.
    """

    def __init__(self, parent=None):
        """
        Intiialize the widget.
        """
        super().__init__()
        self.parent = parent
        self.frame_number = 0
        self.channels = []
        self.channel_selecter = None
        self.file_list = []
        self.channels_to_draw = []
        self.draw_raw = True
        self.draw_clean = True
        self.draw_fft = False

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

    def _reset(self):
        raise NotImplementedError


class OfflineDisplayWidget(DisplayWidget):
    """
    Widget to chose the display options for the plot in offline processing.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_layout()

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

    def on_draw_clicked(self):
        self.channels_to_draw = self.channel_selecter.get_selected_channels()
        self.parent.plot.update_channels(self.channels_to_draw)

    def on_frame_changed(self, text=None):
        if text == "":
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

    def _update_frame_number(self, direction=None, value=None):
        if direction == "prev":
            self.frame_number -= 1
        elif direction == "next":
            self.frame_number += 1
        elif value is not None:
            self.frame_number = int(value) - 1
        self.frame_number = min(self.frame_number, self.n_frames - 1)
        self.frame_number = max(self.frame_number, 0)
        return self.frame_number

    def set_file_params(self, channels, n_frames):
        self.channels = channels
        self.n_frames = n_frames
        self.channel_selecter = ChannelSelecter(self, self.channels)
        self._reset()
        self.enable()

    def _reset(self):
        self._update_frame_number("1")
        self.input_frame.setText("1")
        self.draw_raw_button.setChecked(True)
        self.draw_clean_button.setChecked(True)
        self.show_fft_button.setChecked(False)
        self.update_draw_params()


class StreamDisplayWidget(DisplayWidget):
    """
    Widget to chose the display options for the plot in online processing.
    """

    def __init__(self, parent=None, enable=True):
        super().__init__(parent)
        self.current_frame = 0
        self.frame_list = []
        self._init_layout()
        if enable is False:
            self.disable()

    def _init_layout(self):
        layout = QGridLayout()
        self.popup_button = QPushButton("Select channels to show")
        self.popup_button.clicked.connect(self.on_popup_button_clicked)

        self.display_processed_btn = QPushButton("Show processed")
        self.display_processed_btn.clicked.connect(self.on_display_processed)
        self.display_processed_btn.setEnabled(False)

        self.prev_frame = QPushButton("Previous")
        self.prev_frame.setEnabled(False)
        self.next_frame = QPushButton("Next")
        self.next_frame.setEnabled(False)
        self.sampling_frame = QPushButton("Current")
        self.sampling_frame.setEnabled(False)
        self.prev_frame.clicked.connect(self.on_prev_frame_clicked)
        self.next_frame.clicked.connect(self.on_next_frame_clicked)
        self.sampling_frame.clicked.connect(self.on_sampling_frame_clicked)
        self.input_frame = QLineEdit()
        self.input_frame.setText("1")
        self.input_frame.textEdited.connect(self.on_frame_changed)

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

        layout.addWidget(QLabel("<b><font size=5>Display options</font></b>"), 0, 0, 1, 5, Qt.AlignCenter)

        layout.addWidget(self.display_processed_btn, 1, 3, 1, 1)
        layout.addWidget(self.display_all_btn, 1, 2, 1, 1)
        layout.addWidget(self.popup_button, 1, 0, 1, 2)
        layout.addWidget(QLabel("Frame:"), 2, 0, 1, 1)
        layout.addWidget(self.prev_frame, 2, 1, 1, 1)
        layout.addWidget(self.next_frame, 2, 2, 1, 1)
        layout.addWidget(self.sampling_frame, 2, 3, 1, 1)
        layout.addWidget(self.input_frame, 2, 4, 1, 1)
        layout.addWidget(self.draw_raw_button, 3, 0, 1, 1)
        layout.addWidget(self.draw_clean_button, 3, 1, 1, 1)
        # layout.addWidget(self.show_fft_button, 2, 2, 1, 1)
        layout.addWidget(self.cursor_pos, 4, 0, 1, 5)
        self.setLayout(layout)

    def on_prev_frame_clicked(self):
        self._update_frame_number("prev")
        self.on_frame_changed()

    def on_next_frame_clicked(self):
        if self.is_sampling_frame:
            return
        self._update_frame_number("next")
        self.on_frame_changed()

    def on_sampling_frame_clicked(self):
        self._update_frame_number("sampling")

    def set_button_on(self):
        self.prev_frame.setEnabled(True)
        self.next_frame.setEnabled(True)
        self.sampling_frame.setEnabled(True)

    def on_frame_changed(self, text=None):
        if text == "":
            return
        if text:
            self._update_frame_number(value=text)
        self.input_frame.setText(str(self.current_frame + 1))
        self.parent.update_frame(self.current_frame)

    def _update_frame_number(self, direction=None, value=None):
        if direction == "prev":
            self.current_frame -= 1
        elif direction == "next":
            self.current_frame += 1
        elif direction == "sampling":
            self.current_frame = self.n_frames
        elif value is not None:
            self.current_frame = int(value) - 1
        self.current_frame = min(self.current_frame, self.n_frames)
        self.current_frame = max(self.current_frame, 0)
        return self.current_frame

    def set_file_params(self, channels):
        self.channels = channels
        self.channel_selecter = ChannelSelecter(self, self.channels)
        self._reset()
        self.enable()

    def _reset(self):
        self.current_frame = 0
        self.input_frame.setText("1")
        self.draw_raw_button.setChecked(True)
        self.draw_clean_button.setChecked(True)
        self.show_fft_button.setChecked(False)
        self.update_draw_params()

    def append_frame_number(self, frame_number):
        if self.is_sampling_frame:
            self.current_frame = frame_number
            self.input_frame.setText(str(frame_number + 1))
        self.frame_list.append(frame_number)

    def on_draw_clicked(self):
        self.channels_to_draw = self.channel_selecter.get_selected_channels()
        self.parent.plot.update_channels_visibility(self.channels_to_draw)

    @property
    def is_sampling_frame(self):
        return self.current_frame == self.n_frames

    @property
    def n_frames(self):
        if self.frame_list == []:
            return 0
        return max(self.frame_list)
