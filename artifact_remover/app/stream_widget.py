import json
import threading

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLineEdit, QDialog, QLabel
from PyQt5.QtCore import QObject, pyqtSignal

import numpy as np
from .gui_utils import ensure_list, Worker
from ..io_utils import export_csv
from ..processing_utils import Quality
from .popup_utils import ChannelsPopup
from biosiglive.streaming.async_server import AsyncTCPServer
import asyncio  

class Bridge(QObject):
    data_received = pyqtSignal(object)


class StreamWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.channels = None
        self.channels_setter = None
        self.max_height = 40
        self.setFixedHeight(self.max_height)
        self.quality = Quality()
        self._init_layout()
        self.bridge = Bridge()
        self.paused = False

    def task(self, d, t):
        self.bridge.data_received.emit(True)

    def _init_layout(self):
        self.play_button = QPushButton("Play")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self._play)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop)
        self.pause_button = QPushButton("Pause")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self._pause)
        
        self.adress_in = QLineEdit()
        self.adress_in.setText("127.0.0.1")
        self.port_in = QLineEdit()
        self.port_in.setText("12345")
        self.ac_rate_in = QLineEdit()
        self.ac_rate_in.setText("2000")
        self.set_channels_button = QPushButton("Set Channels")
        self.set_channels_button.clicked.connect(self._set_channels)
        self.display_wind_in = QLineEdit()
        self.display_wind_in.setText("5")

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.play_button)
        self.layout.addWidget(self.stop_button)
        self.layout.addWidget(self.pause_button)
        label = QLabel("Address:")
        label.setMaximumHeight(self.max_height)
        self.layout.addWidget(label)
        self.layout.addWidget(self.adress_in)
        label = QLabel("Port:")
        label.setMaximumHeight(self.max_height)
        self.layout.addWidget(label)
        self.layout.addWidget(self.port_in)
        label = QLabel("Acquisition rate:")
        label.setMaximumHeight(self.max_height)
        self.layout.addWidget(label)
        self.layout.addWidget(self.ac_rate_in)
        self.layout.addWidget(self.set_channels_button)
        self.layout.addWidget(QLabel('Display last (s):'))
        self.layout.addWidget(self.display_wind_in)
        self.setLayout(self.layout)

    def _play(self):
        self.stop_button.setEnabled(True)
        self.pause_button.setEnabled(True)
        self.play_button.setEnabled(False)
        if not self.paused:
            self.parent.parent.log_box.log(f"Connecting to the client ({self.address}:{self.port}) and starting the stream...")
            # self.server = AsyncTCPServer(self.address, self.port, buffer_length=self.display_window, expected_fs=self.acquisition_rate)
            # asyncio.run(self.server.start(task=self.task))
            thread = threading.Thread(target=self._run_asyncio, daemon=True)
            thread.start()
            self.bridge.data_received.connect(self.parent.update_data)
            self.parent.init_stream(self.display_window)

    def _run_asyncio(self):
        self.server = AsyncTCPServer(self.address, self.port, buffer_length=self.display_window, expected_fs=self.acquisition_rate)
        self.server.init_buffer(len(self.channels))
        asyncio.run(self.server.start(task=self.task))

    def _stop(self):
        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        asyncio.run(self.server.stop())

    def _pause(self):
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.paused = True

    def _set_channels(self):
        popup = ChannelsPopup(channels=self.channels)
        if popup.exec_() == QDialog.Accepted:
            self.channels = popup.channels
            self.parent.display_options.set_file_params(self.channels)
        self.play_button.setEnabled(self.channels is not None)

    def get_data(self, n_chunks=None):
        if self.server is not None and self.server.buffer is not None:
            return self.server.buffer.get(len=n_chunks)
        else:
            return None

    @property
    def address(self):
        return self.adress_in.text()
    
    @property
    def display_window(self):
        return int(self.display_wind_in.text()) * self.acquisition_rate
    
    @property
    def port(self):
        return int(self.port_in.text())

    @property
    def acquisition_rate(self):
        return int(self.ac_rate_in.text())
    
    