import pyqtgraph as pg
import numpy as np
from scipy.fft import rfft, rfftfreq
from PyQt5.QtWidgets import QGraphicsProxyWidget, QPushButton
from PyQt5.QtCore import Qt
from biosiglive.streaming.utils import CircularBuffer


class ChannelPlot:
    def __init__(self, parent, channel=None, idx=None):
        self.parent = parent
        self.time = None
        self.freqs = None
        self.plot_item = None
        self.curves = None
        self.raw_idx = 0
        self.clean_idx = 1
        self.raw = None
        self.clean = None
        self.x_range = None
        self.name = channel
        self.idx = idx
        self.visible = True
        self.visible_pen_clean = pg.mkPen(color=(255, 0, 0), width=2)
        self.visible_pen_raw = pg.mkPen(color=(0, 0, 0, 256 // 6), width=1)
        self.pen_clean = pg.mkPen(color=(255, 0, 0), width=2)
        self.pen_raw = pg.mkPen(color=(0, 0, 0, 256 // 6), width=1)
        self.invisible_pen = pg.mkPen(color=(0, 0, 0, 0))
        self.fft_plot = False
        self.view = None
        self.proxy = None

    def init_plot(self, data, time, rate):
        self.time = time
        self.freqs = rfftfreq(len(data[0]), d=1 / rate)
        self.plot_item = self.parent.addPlot(row=self.idx, col=0)
        self.plot_item.setXRange(self.time[0], self.time[-1], padding=0)
        self.plot_item.setClipToView(True)
        self.plot_item.setLabel("left", self.name, units="µV")
        self.plot_item.getAxis("left").enableAutoSIPrefix(False)
        self.plot_item.showGrid(x=True, y=True, alpha=0.2)
        self.show_config_btn = QPushButton("i")
        self.show_config_btn.setMaximumWidth(10)
        self.show_config_btn.setEnabled(False)
        self.show_config_btn.clicked.connect(self.on_info_button)
        cursor = Qt.CrossCursor
        self.plot_item.setCursor(cursor)
        self.proxy_mouse = pg.SignalProxy(self.plot_item.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.proxy_btn = QGraphicsProxyWidget()
        self.proxy_btn.setWidget(self.show_config_btn)
        self.parent.addItem(self.proxy_btn, row=self.idx, col=1)

        self.view = self.plot_item.getViewBox()
        self.view.setLimits(xMin=self.time[0], xMax=self.time[-1])
        self.raw, self.clean = data
        self.raw_fft = abs(rfft(self.raw))
        self.clean_fft = abs(rfft(self.clean))
        self.curves = self.plot_item.multiDataPlot(x=self.time, y=data)
        self._set_curve_params()

    def on_info_button(self):
        self.parent.show_config(self.idx)

    def set_visible(self, visible):
        self.proxy_btn.setVisible(visible)
        self.plot_item.setVisible(visible)
        self.visible = visible

    def _set_curve_params(self):
        for c, curve in enumerate(self.curves):
            curve.setPen([self.pen_raw, self.pen_clean][c])

    def update_plot(self, data=None, data_type="both", auto_range=True, time=None):
        if time is not None:
            self.time = time
        self.raw = data[0] if data is not None else self.raw
        self.clean = data[1] if data is not None else self.clean
        y = [self.raw, self.clean]
        x = self.time
        if self.fft_plot:
            self.raw_fft = abs(rfft(self.raw))
            self.clean_fft = abs(rfft(self.clean))
            x = self.freqs
            y = [self.raw_fft, self.clean_fft]
        if auto_range:
            self.plot_item.setXRange(x[0], x[-1], padding=0)
            self.view.setLimits(xMin=x[0], xMax=x[-1])
        self.view.enableAutoRange(x=auto_range, y=auto_range)
        if data_type == "raw" or data_type == "both":
            self.curves[0].setData(x=x, y=y[0], pen=self.pen_raw)
        if data_type == "clean" or data_type == "both":
            self.curves[1].setData(x=x, y=y[1], pen=self.pen_clean)

    def set_link_x(self, plot_item):
        self.plot_item.setXLink(plot_item)

    def change_visibility(self, raw=True, clean=True):
        self.pen_clean = self.visible_pen_clean if clean else self.invisible_pen
        self.pen_raw = self.visible_pen_raw if raw else self.invisible_pen
        self._set_curve_params()

    def change_plot_type(self, fft=False):
        if fft and not self.fft_plot:
            self.fft_plot = True
            self.view.setLimits(xMin=self.freqs[0], xMax=self.freqs[-1])
            self.plot_item.setXRange(0, self.freqs[-1], padding=0)
            self.plot_item.setLabel("left", self.name)
            self.update_plot()
        elif not fft and self.fft_plot:
            self.fft_plot = False
            self.view.setLimits(xMin=self.time[0], xMax=self.time[-1])
            self.plot_item.setXRange(self.time[0], self.time[-1], padding=0)
            self.plot_item.setLabel("left", self.name)
            self.update_plot()

    def mouseMoved(self, e):
        pos = e[0]
        # if self.plot_item.sceneBoundingRect().contains(pos):
        mousePoint = self.plot_item.vb.mapSceneToView(pos)
        x_y = [np.round(mousePoint.x(), 3), np.round(mousePoint.y(), 3)]
        # else:
        #     x_y = [' ', ' ']
        self.parent.update_mouse_pos(x_y)


class Plotter(pg.GraphicsLayoutWidget):
    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    pg.setConfigOption("leftButtonPan", False)

    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.plot_list = []
        self.channels = []
        self.visible_channels = []
        self.raw_data = None
        self.clean_notch = None
        self.clean_svd = None
        self.idx_to_plot = 0
        self.time = None
        self.current_filter = "notch"
        self.built = False
        self.timer = None

    def init_plots(self):
        for c, channel in enumerate(self.channels):
            data = [self.raw_data[self.idx_to_plot, c, :], self.clean_signal[self.idx_to_plot, c, :]]
            self.plot_list.append(ChannelPlot(self, channel, c))
            self.plot_list[-1].init_plot(data, self.time[0], self.parent.remover_options.get_rate())
            if c != 0:
                self.plot_list[c].set_link_x(self.plot_list[0].plot_item)

    def get_plot_by_name(self, name):
        names = [p.name for p in self.plot_list]
        return self.plot_list[names.index(name)]

    def get_plot_by_idx(self, idx):
        idxs = [p.idx for p in self.plot_list]
        return self.plot_list[idxs.index(idx)]

    def _reorder_channels(self, channels):
        return [c for c in self.channels if c in channels]

    def update_draw_params(self, plot_raw, plot_clean, plot_fft):
        for plot in self.plot_list:
            plot.change_visibility(plot_raw, plot_clean)
            plot.change_plot_type(plot_fft)

    def enable_config_button(self, idxs):
        for plot in self.plot_list:
            if plot.idx not in idxs:
                continue
            plot.show_config_btn.setEnabled(True)

    def update_config_button(self, process_config):
        for plot in self.plot_list:
            if process_config and len(process_config) >= plot.idx and process_config[plot.idx] is not None:
                plot.show_config_btn.setEnabled(True)
            else:
                plot.show_config_btn.setEnabled(False)

    def show_config(self, idx):
        self.parent.show_config(idx)

    def update_mouse_pos(self, pos):
        self.parent.update_mouse_pos(pos)


class OfflinePlotter(Plotter):
    def __init__(self, parent=None):
        super().__init__(parent)

    def initialize_data(self, data, channels, time, cleaned_notch=None, cleaned_svd=None):
        self.channels = channels
        self.visible_channels = self.parent.display_options.channel_selecter.get_channel_names()
        self.raw_data = data
        self.clean_notch = data.copy() if cleaned_notch is None else cleaned_notch
        self.clean_svd = data.copy() if cleaned_svd is None else cleaned_svd
        self.time = time
        self.idx_to_plot = 0
        self.current_filter = "notch"
        if self.built:
            self.clear()
        self.plot_list = []
        self.init_plots()
        self.built = True

    def update_data(self, data, channel_idxs, frame_idxs, data_type="both", auto_range=True):
        if data_type == "both":
            self.raw_data[frame_idxs, channel_idxs, :] = data[0]
            self.clean_signal[frame_idxs, channel_idxs, :] = data[1]
        elif data_type == "raw":
            self.raw_data[frame_idxs, channel_idxs, :] = data
        elif data_type == "clean":
            self.clean_signal[frame_idxs, channel_idxs, :] = data
        self.update_frame(self.idx_to_plot, force=True, data_type=data_type, auto_range=auto_range)

    def update_frame(self, idx, force=False, data_type="both", auto_range=True, update_time=False):
        visible_idx = [self.channels.index(c) for c in self.visible_channels]
        has_changed = self.idx_to_plot != idx or force
        self.idx_to_plot = idx
        time = None if not update_time else self.time[idx]
        for plot in self.plot_list:
            if plot.idx in visible_idx and plot.visible is False:
                plot.set_visible(True)
                plot.update_plot(
                    [self.raw_data[idx, plot.idx, :], self.clean_signal[idx, plot.idx, :]],
                    data_type,
                    auto_range,
                    time=time,
                )
            elif plot.idx not in visible_idx and plot.visible is True:
                plot.set_visible(False)
            elif plot.idx in visible_idx and plot.visible is True and has_changed:
                plot.update_plot(
                    [self.raw_data[idx, plot.idx, :], self.clean_signal[idx, plot.idx, :]],
                    data_type,
                    auto_range,
                    time=time,
                )

    def update_channels(self, channels):
        idx = [i[0] for i in channels]
        channels = [i[1] for i in channels]
        reordered_channels = self._reorder_channels(channels)
        if self.visible_channels == reordered_channels:
            return
        self.visible_channels = reordered_channels
        self.update_frame(self.idx_to_plot)

    def update_filter(self, filter_type):
        self.current_filter = filter_type
        self.update_frame(self.idx_to_plot, True, data_type="clean", auto_range=False)

    @property
    def clean_signal(self):
        return self.clean_notch if self.current_filter == "notch" else self.clean_svd


class StreamPlotter(Plotter):
    def __init__(self, parent=None, rate=30):
        super().__init__(parent)
        self.rate = rate
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.is_streaming = False

    def initialize_data(self, data_buffer, time, channels, display_windows):
        self.display_windows = display_windows
        self.processed_svd_buffer = CircularBuffer(len(channels), self.display_windows)
        self.processed_notch_buffer = CircularBuffer(len(channels), self.display_windows)
        self.channels = channels
        self.visible_channels = self.parent.display_options.channel_selecter.get_channel_names()
        self.raw_data = data_buffer
        self.time = time
        self.current_filter = "notch"
        if self.built:
            self.clear()
        self.plot_list = []
        self.init_plots()
        self.built = True
        self.timer.start(1000 // self.rate)        

    def update_plot(self, data_type="both", auto_range=True):
        if not self.is_streaming:
            return
        visible_idx = [self.channels.index(c) for c in self.visible_channels]
        time = None
        for plot in self.plot_list:
            if plot.idx in visible_idx and plot.visible is False:
                plot.set_visible(True)
                plot.update_plot(
                    [self.raw_signal[plot.idx, :], self.clean_signal[plot.idx, :]],
                    data_type,
                    auto_range,
                    time=time,
                )
            elif plot.idx not in visible_idx and plot.visible is True:
                plot.set_visible(False)
            elif plot.idx in visible_idx and plot.visible is True:
                plot.update_plot(
                    [self.raw_signal[plot.idx, :], self.clean_signal[plot.idx, :]],
                    data_type,
                    auto_range,
                    time=time,
                )

    def update_data(self, data_svd=None, data_notch=None, idx=None):
        if idx is not None:
            self.update_channels(data_notch, data_svd, idx)
            return
        if data_svd is not None:
            self.processed_svd_buffer.append(data_svd, fill_discontinuous=True)
        if data_notch is not None:
            self.processed_notch_buffer.append(data_notch, fill_discontinuous=True)

    def update_channels(self, data_notch=None, data_svd=None, idx=None):
        if data_svd is not None:
            self.processed_svd_buffer.append(data_svd, fill_discontinuous=True)
        if data_notch is not None:
            self.processed_notch_buffer.append(data_notch, fill_discontinuous=True)

    def update_filter(self, filter_type):
        self.current_filter = filter_type

    @property
    def clean_signal(self):
        return (
            self.processed_notch_buffer.get(self.display_windows)[0]
            if self.current_filter == "notch"
            else self.processed_svd_buffer.get(self.display_windows)[0]
        )
    
    @property
    def raw_signal(self):
        return self.raw_data.get(self.display_windows)[0]

    def init_plots(self):
        for c, channel in enumerate(self.channels):
            data = [self.raw_signal[c, :], self.clean_signal[c, :]]
            self.plot_list.append(ChannelPlot(self, channel, c))
            self.plot_list[-1].init_plot(data, self.time, self.parent.acquisition_rate)
            if c != 0:
                self.plot_list[c].set_link_x(self.plot_list[0].plot_item)
