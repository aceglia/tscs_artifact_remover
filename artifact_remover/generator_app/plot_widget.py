import pyqtgraph as pg
import numpy as np
from scipy.fftpack import rfft, rfftfreq
from PyQt5.QtWidgets import QGraphicsProxyWidget, QPushButton
from PyQt5.QtCore import Qt

class ChannelPlot:
    def __init__(self, parent, channel=None, idx=None):
        self.parent = parent
        self.time = None
        self.freqs = None
        self.plot_item = None
        self.curves = None
        self.raw_idx = 0
        self.contaminated_idx = 1
        self.raw = None
        self.contaminated = None
        self.x_range = None
        self.name = channel
        self.idx = idx
        self.visible = True
        self.visible_pen_contaminated = pg.mkPen(color=(255, 0, 0), width=2)
        self.visible_pen_raw = pg.mkPen(color=(0, 0, 0, 256//6), width=1)
        self.pen_contaminated = pg.mkPen(color=(255, 0, 0), width=2)
        self.pen_raw = pg.mkPen(color=(0, 0, 0, 256//6), width=1)
        self.invisible_pen = pg.mkPen(color=(0, 0, 0, 0))
        self.fft_plot = False
        self.view = None
        self.proxy = None

    def init_plot(self, data, time, rate):
        self.time = [time] * 2 if len(time) != 2 else time
        self.freqs = rfftfreq(len(data[0]), d=1/rate)
        self.plot_item = self.parent.addPlot(row=self.idx, col=0)
        self.plot_item.setXRange(self.time[0][0], self.time[0][-1], padding=0)
        self.plot_item.setClipToView(True)
        self.plot_item.setLabel('left', self.name, units='µV') 
        self.plot_item.getAxis('left').enableAutoSIPrefix(False)
        self.plot_item.showGrid(x=True, y=True, alpha=0.2)
        self.show_config_btn =  QPushButton("i")
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
        self.view.setLimits(
            xMin=self.time[0][0],
            xMax=self.time[0][-1]
        )
        self.raw, self.contaminated = data
        self.raw_fft = abs(rfft(self.raw))
        self.contaminated_fft = abs(rfft(self.contaminated))  
        self.curves = self.plot_item.multiDataPlot(
            x=self.time,
            y=data
        )
        self._set_curve_params()

    def on_info_button(self):
        self.parent.show_config(self.idx)

    def set_visible(self, visible):
        self.proxy_btn.setVisible(visible)
        self.plot_item.setVisible(visible)
        self.visible = visible

    def _set_curve_params(self):
        for c, curve in enumerate(self.curves):
            curve.setPen([self.pen_raw, self.pen_contaminated][c])

    def update_plot(self, data=None, data_type="both", auto_range=True, time=None):
        if time is not None:
            self.time = time
        self.raw = data[0] if data is not None else self.raw
        self.contaminated = data[1] if data is not None else self.contaminated
        self.raw_fft = abs(rfft(self.raw))
        self.contaminated_fft = abs(rfft(self.contaminated))
        y = [self.raw, self.contaminated]
        x = [self.time] * 2 if len(self.time) != 2 else self.time
        if self.fft_plot:
            x = [self.freqs, self.freqs]
            y = [self.raw_fft, self.contaminated_fft]
        if auto_range:
            self.plot_item.setXRange(x[0][0], x[0][-1], padding=0)
            self.view.setLimits(
                xMin=x[0][0],
                xMax=x[0][-1]
            )
        self.view.enableAutoRange(x=auto_range, y=auto_range)
        if data_type == "contaminated" or data_type == "both":
            self.curves[0].setData(x=x[0], y=y[0], pen=self.pen_contaminated)
        if data_type == "raw" or data_type == "both":
            self.curves[1].setData(x=x[1], y=y[1], pen=self.pen_raw)

    def set_link_x(self, plot_item):
        self.plot_item.setXLink(plot_item)

    def change_visibility(self, raw=True, contaminated=True):
        self.pen_contaminated = self.visible_pen_contaminated if contaminated else self.invisible_pen
        self.pen_raw = self.visible_pen_raw if raw else self.invisible_pen
        self._set_curve_params()

    def change_plot_type(self, fft=False):
        if fft and not self.fft_plot:
            self.fft_plot = True
            self.view.setLimits(
            xMin=self.freqs[0],
            xMax=self.freqs[-1]
            )
            self.plot_item.setXRange(0, self.freqs[-1], padding=0)
            self.plot_item.setLabel('left', self.name)
            self.update_plot()
        elif not fft and self.fft_plot:
            self.fft_plot = False
            self.view.setLimits(
            xMin=self.time[0][0],
            xMax=self.time[0][-1]
            )
            self.plot_item.setXRange(self.time[0][0], self.time[0][-1], padding=0)
            self.plot_item.setLabel('left', self.name)
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
    pg.setConfigOption('leftButtonPan', False) 

    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.plot_list = [None, None]
        self.raw_data = None
        self.contaminated_signal = None

        self.idx_to_plot = 0
        self.time = None
        self.built = False

    def _init_template_plot(self, template, sampled_template, time):
        self.plot_list[0] = ChannelPlot(self, 'Template', 0)
        self.plot_list[0].init_plot([template, sampled_template], time, 1)

    def _init_data_plot(self, data, sampling_rate):
        self.plot_list[1] = ChannelPlot(self, 'Signal', 1)
        data = [data] * 2
        time = np.linspace(0, data[0].shape[-1]/sampling_rate, data[0].shape[-1])
        self.plot_list[1].init_plot(data, time, sampling_rate)
    
    def update_template(self, template, sampled_template):
        time = [np.linspace(0, 1, len(t)) for t in [template, sampled_template]]
        if self.plot_list[0] is None:
            self._init_template_plot(template, sampled_template, time)
            return
        self.plot_list[0].update_plot([template, sampled_template], time=time)
        self.plot_list[0]._set_curve_params()
    
    def update_stim_train(self, data):
        if self.plot_list[1] is None:
            self._init_data_plot(data, self.parent.template_options.params_widget.sampling_rate)
        self.plot_list[1].update_plot([data, data], data_type="both", auto_range=False)

    def initialize_plot(self, data, time):
        self.raw_data = data
        self.contaminated_signal = data
        self.time = time
        self.idx_to_plot = 0
        if self.built:
            self.clear()
        self.plot_list = []
        self._init_data_plot(self.parent.template_options.params_widget.sampling_rate)
        self.built = True

    def update_frame(self, force=False, data_type="both", auto_range=True, update_time=False):
        visible_idx = [self.channels.index(c) for c in self.visible_channels]
        has_changed =  force
        time = None if not update_time else self.time
        for plot in self.plot_list:
            if plot.idx in visible_idx and plot.visible is False:
                plot.set_visible(True)
                plot.update_plot([self.contaminated_signal, self.raw_data], data_type, auto_range, time=time)
            elif plot.idx not in visible_idx and plot.visible is True:
                plot.set_visible(False)
            elif plot.idx in visible_idx and plot.visible is True and has_changed:
                plot.update_plot([self.contaminated_signal, self.raw_data], data_type, auto_range, time=time)
    
    def update_data(self, data, data_type="both", auto_range=True):
        if data_type == 'both':
            self.raw_data = data[0]
            self.contaminated_signal= data[1]
        elif data_type == 'raw':
            self.raw_data = data
        elif data_type == 'contaminated':
            self.contaminated_signal = data
        self.update_frame(force=True, data_type=data_type, auto_range=auto_range)            

    def update_draw_params(self, plot_fft):
        self.plot_list[1].change_plot_type(plot_fft)

    def update_mouse_pos(self, pos):
        self.parent.update_mouse_pos(pos)