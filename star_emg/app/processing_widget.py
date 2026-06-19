from functools import partial
import json
import time

from PyQt5.QtWidgets import QWidget, QSplitter, QVBoxLayout
from PyQt5.QtCore import Qt, QThreadPool, QTimer

import numpy as np
import multiprocessing as mp
from scipy.fft import rfftfreq
import scipy.io as sio

from biosiglive.streaming.utils import CircularBuffer
from .stream_utils import ClearableQueue
from .remover_widget import StreamRemover, OfflineRemover
from ..rt_automatic_remover import RtArtifactRemover
from .display_options import StreamDisplayWidget, OfflineDisplayWidget
from .plot_widget import OfflinePlotter, StreamPlotter
from .gui_utils import ensure_list, Worker
from ..io_utils import export_csv
from ..processing_utils import Quality
from ..solution import Solution
from .stream_widget import StreamWidget
from .popup_utils import FilterDialog


class ProcessingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.stream_widget = None

    def _init_layout(self):
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        right_layout.addWidget(self.remover_options.process_widgets)
        right_layout.addWidget(self.display_options)
        right_panel.setLayout(right_layout)
        right_panel.setMaximumWidth(500)

        splitter = QSplitter(Qt.Horizontal)

        splitter.addWidget(self.plot)
        splitter.addWidget(right_panel)
        self.setLayout(QVBoxLayout())
        if self.stream_widget is not None:
            self.layout().addWidget(self.stream_widget)
        self.layout().addWidget(splitter)

    def update_filter(self, name="notch"):
        self.remover_options.update_filter(name)
        self.plot.update_filter(name)
        self.plot.update_config_button(self.remover_options.get_current_config())
        processed = self.get_processed_channels()
        if processed is not None:
            self.display_options.display_processed_btn.setEnabled(True)
        else:
            self.display_options.display_processed_btn.setEnabled(False)
        self.remover_options.show_config("")

    def update_mouse_pos(self, pos):
        self.display_options.update_mouse_pos(pos)

    def get_processed_channels(self):
        return self.remover_options.get_processed_channels()


class OfflineProcessingWidget(ProcessingWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.remover_options = OfflineRemover(self)
        self.display_options = OfflineDisplayWidget(self)
        self.display_options.disable()
        self.remover_options.disable()
        self.plot = OfflinePlotter(self)
        self._init_layout()
        self.clean_notch = None
        self.clean_svd = None
        self.process_file_path = None
        self.worker = None
        self.threadpool = QThreadPool()
        self.filter_dialog = None
        self.canceled = False
        self.quality_notch = Quality()
        self.quality_svd = Quality()

    def update_frame(self, frame_number):
        self.plot.update_frame(frame_number, update_time=True)
        self.plot.update_config_button(self.remover_options.get_current_config())
        self.remover_options.update_frame(frame_number)
        self.remover_options.show_config("")

    def process(self, **kwargs):
        kwargs["batch_idxs"] = ensure_list(self.display_options.frame_number)
        # self.remover_options.disable()
        self.parent.log_box.log("Processing data...")
        if kwargs["notch_filter"]:
            results, init_shape = self._thread_safe_process(
                self.remover_options.remover.perform_window_process,
                self.remover_options.remover.data_loader.init_data,
                self.remover_options.remover.data_loader.data_rate,
                **kwargs,
            )
            self._on_processing_done((results, init_shape), kwargs)
        else:
            worker = Worker(
                self._thread_safe_process,
                self.remover_options.remover.perform_window_process,
                self.remover_options.remover.data_loader.init_data,
                self.remover_options.remover.data_loader.data_rate,
                **kwargs,
            )
            worker.signals.finished.connect(lambda result: self._on_processing_done(result, kwargs))
            worker.signals.error.connect(lambda e: self.parent.log_box.log(f"Error while processing data: {e}"))
            self.threadpool.start(worker)

    def _on_processing_done(self, result, kwargs):
        list_results, init_shape = result
        solution = Solution(self.remover_options.remover.data_loader.data_rate)
        fct = solution.from_notch_filter if kwargs["notch_filter"] else solution.from_signal_decomposition
        fct(list_results, initial_data_shape=init_shape)

        self.remover_options.remover.solution = solution
        self.update_processed_plot(kwargs["batch_idxs"], kwargs["channel_idxs"])
        self.parent.log_box.log("Data processing done!")
        qual_fct = self.quality_notch.compute_quality if kwargs["notch_filter"] else self.quality_svd.compute_quality
        qual_fct(
            self.remover_options.get_data(kwargs["batch_idxs"], kwargs["channel_idxs"]).astype(float),
            self.remover_options.get_cleaned_data().astype(float),
            ground_truth=None,
            fs=self.remover_options.get_rate(),
            idx=kwargs["batch_idxs"],
            channel=kwargs["channel_idxs"],
        )
        self.remover_options.enable(all=True)
        self.display_options.display_processed_btn.setEnabled(True)
        self.parent.set_saved_ok(False)

    @staticmethod
    def _thread_safe_process(fct, data, data_rate, batch_idxs, channel_idxs, process_window, **kwargs):
        if batch_idxs:
            if not isinstance(batch_idxs, list):
                batch_idxs = [batch_idxs]
            data = data[batch_idxs, ...]

        if channel_idxs:
            if not isinstance(channel_idxs, list):
                channel_idxs = [channel_idxs]
            data = data[:, channel_idxs, :]

        # if kwargs['data_window']:
        #     data = data[..., kwargs['data_window'][0] : min(data.shape[-1], kwargs['data_window'][1])]
        process_window = process_window if process_window is not None else data.shape[-1]
        _init_data_shape = data.shape
        data = data.reshape(-1, data.shape[-1])
        list_results = []
        fft_freqs = np.fft.rfftfreq(
            process_window - (kwargs["hankel_size"] - 1) * kwargs["hankel_delay"], 1 / data_rate
        )
        for d in range(data.shape[0]):
            list_results.append(fct(data=data[d], fs=data_rate, fft_freqs=fft_freqs, window=process_window, **kwargs))
        return list_results, _init_data_shape

    def set_file(self, file_list, process_data_file=None, filtering_params=None):
        if filtering_params is None:
            self.filter_dialog = FilterDialog(self)
            if self.filter_dialog.exec_() == 0:
                self.canceled = True
                return
            else:
                self.canceled = False
                filtering_params = self.filter_dialog.get_filter_params()
        self.file_path = file_list
        self.remover_options.set_file(file_list, **filtering_params)
        if self.remover_options.remover is None:
            self.file_path = None
            return
        clean_svd, clean_notch = None, None
        for qual in [self.quality_notch, self.quality_svd]:
            qual.init_shape(self.remover_options.get_data().shape)
        if process_data_file is not None:
            clean_svd, clean_notch = self.get_process_data(process_data_file)
            clean = [clean_notch, clean_svd]
            init_data = self.remover_options.get_data()
            for q, qual_fct in enumerate([self.quality_notch.compute_quality, self.quality_svd.compute_quality]):
                qual_fct(init_data, clean[q], ground_truth=None, fs=self.remover_options.get_rate())

        self.display_options.set_file_params(
            self.remover_options.get_channels(), self.remover_options.get_all_data().shape[0]
        )
        self.plot.initialize_data(
            self.remover_options.get_all_data(),
            self.remover_options.get_channels(),
            self.remover_options.remover.data_loader.time,
            cleaned_notch=clean_notch,
            cleaned_svd=clean_svd,
        )

        self.parent.toolbar.enable_filter_menu()
        self.parent.toolbar.radio_svd_filter_button.setEnabled(True)
        self.parent.toolbar.radio_notch_filter_button.setEnabled(False)
        self.update_frame(0)
        self.update_filter("notch")

    def update_processed_plot(self, batch_idxs, channel_idxs):
        cleaned_data = self.remover_options.get_cleaned_data()
        self.plot.update_data(
            cleaned_data, ensure_list(channel_idxs), ensure_list(batch_idxs), data_type="clean", auto_range=False
        )
        self.plot.enable_config_button(channel_idxs)

    def save_file(self, path):
        self.process_file_path = path
        dic_to_save = {
            "raw_data": self.plot.raw_data,
            "processed_data_svd": self.plot.clean_svd,
            "processed_data_notch": self.plot.clean_notch,
            "channels": self.remover_options.get_channels(),
            "rate": self.remover_options.get_rate(),
        }
        sio.savemat(path, dic_to_save)
        export_csv(path, **dic_to_save)
        self.parent.log_box.log(
            "To use the processed file in signal you can import the txt file saved at " + path.replace(".mat", ".txt")
        )
        self.parent.set_saved_ok(True)

    def load_config(self, path):
        if path == "":
            self.canceled = True
            return
        else:
            self.canceled = False
        with open(path, "r") as f:
            config_data = json.load(f)
        self.set_file(config_data["file_path"], config_data["process_file_path"], config_data["preprocessing_params"])
        self.remover_options.svd_options.load_config(config_data["filters_params_svd"])
        self.remover_options.notch_options.load_config(config_data["filters_params_notch"])

    def get_process_data(self, file_path):
        if file_path is None:
            return
        data = sio.loadmat(file_path)
        return data["processed_data_svd"], data["processed_data_notch"]

    def save_config(self, path):
        config = {
            "file_path": self.file_path,
            "process_file_path": self.process_file_path,
            "preprocessing_params": self.remover_options.remover.data_loader.filtering_params,
            "filters_params_svd": self.remover_options.svd_options.process_arguments,
            "filters_params_notch": self.remover_options.notch_options.process_arguments,
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=4)

    def _convert_quality_to_dict(self, quality):
        dict_to_return = {
            "kurtosis": [quality[0][0], quality[1][0]],
            "Line Length": [quality[0][1], quality[1][1]],
            "Median frequency": [quality[0][2], quality[1][2]],
            "FFT Amplitude": [quality[0][3], quality[1][3]],
        }
        return {k: [np.round(float(v[0]), 3), np.round(float(v[1]), 3)] for k, v in dict_to_return.items()}

    def show_config(self, idx):
        quality = self.quality_svd if self.remover_options.current_filter == "svd" else self.quality_notch
        frame = (
            self.remover_options.svd_options.current_frame
            if self.remover_options.current_filter == "svd"
            else self.remover_options.notch_options.current_frame
        )
        quality_dict = self._convert_quality_to_dict(quality.get_quality(frame, idx))
        config = self.remover_options.get_current_config(idx)
        channel = self.remover_options.get_channels()[idx]
        self.remover_options.show_config(
            f"Informations for channel {channel}:\n\n"
            + self._text_from_config(config)
            + self._text_from_quality(quality_dict)
        )

    @staticmethod
    def _text_from_config(config):
        text = ""
        if config is None:
            return text
        for name, value in config.items():
            text += name + ": " + str(value) + "\n"
        return text

    @staticmethod
    def _text_from_quality(quality):
        text = ""
        if quality is None:
            return text
        for key, value in quality.items():
            text += f"{key}: Raw: {value[0]}; Processed: {value[1]}\n"
        return text


class StreamProcessingWidget(ProcessingWidget):
    def __init__(self, parent=None, process_rate=60):
        super().__init__(parent)
        self.counter = 0
        self.last_processed_counter = 0
        self.process_rate = process_rate
        self.timer = QTimer()
        self.timer.timeout.connect(self.process)
        self.fft_freqs = None
        self.processing = False
        self._partial_fct = None
        self.last_n_arrived = 0
        self.tmp_processed = []
        self.last_processed_time = None
        self.streaming_data = False
        self.frame_counter = 0
        self.frames = {}
        self.once_updated = False
        self.last_n_processed = 0
        self.running = False
        self.last_seen = None
        self.long_process_warning = False
        self.queue_process = None
        self.queue_plot = None
        self.process_args_event = None
        self.queue_process_args = None
        self.process_buffer = None
        self.is_running_event = None
        self.channels_mapping = None
        self.n_process = mp.cpu_count() - 4 if mp.cpu_count() > 2 else 1
        self.remover_options = StreamRemover(self)
        self.display_options = StreamDisplayWidget(self)
        self.plot = StreamPlotter(self, rate=min(process_rate, 30))
        self.stream_widget = StreamWidget(self)
        self._init_layout()

    @staticmethod
    def process(
        queue_in,
        queue_out,
        process_args_event,
        queue_process_args,
        runing_event,
        buff_len,
        channels_idxs,
        acquisition_rate,
    ):
        channel_configs = None
        channel_configs_glob = {i: {} for i in channels_idxs}
        process_buffer = {i: CircularBuffer(1, buff_len) for i in channels_idxs}
        last_t = {i: -np.inf for i in channels_idxs}
        fct = partial(
            RtArtifactRemover()._remove_artifact_from_windows,
            return_dict=False,
            data_rate=acquisition_rate,
            offline=False,
        )
        runing_event.wait()
        while runing_event.is_set():
            data = queue_in.get_stacked()
            if data is None:
                time.sleep(0.001)
                continue
            idxs = list(data.keys())
            for ch in idxs:
                process_buffer[ch].append(data[ch][0], data[ch][1])

            if process_args_event.is_set():
                channel_configs = queue_process_args.get(timeout=0.02)
                process_args_event.clear()
                for ch in idxs:
                    if ch in channel_configs["channel_idxs"]:
                        channel_configs_glob[ch] = channel_configs
                # print('process: received new config for channel', ch, channel_configs_glob[ch])

            if channel_configs is not None:
                for ch in idxs:
                    if ch in channel_configs["channel_idxs"]:
                        channel_configs_glob[ch] = channel_configs

            for ch in idxs:
                buf_data, buf_t = process_buffer[ch].get()

                mask = buf_t > last_t[ch]

                if not np.any(mask):
                    continue

                t_new = buf_t[mask]
                d_new = buf_data[0][mask]  # (n_samples,)

                last_t[ch] = t_new[-1]

                config = channel_configs_glob[ch] if channel_configs_glob[ch] != {} else None

                if config is None:
                    queue_out[ch].put_nowait((d_new, t_new, ch))
                    continue

                window = config.get("process_window", None)

                if window is None or buf_data.shape[-1] < window:
                    queue_out[ch].put_nowait((d_new, t_new, ch))
                    continue

                res = StreamProcessingWidget._process_worker(
                    fct,
                    buf_data[0, -window:],  # last window
                    buf_t[-window:],
                    config,
                    ch,
                    len(t_new),
                )
                queue_out[ch].put_nowait(res)

    @staticmethod
    def _process_worker(fct, data, t, process_args, idx, n_new_data):
        res = fct(data=data, **process_args)
        return res[0][-n_new_data:], t[-n_new_data:], idx

    def init_stream(self, display_window, queue_process=None, is_running_event=None, channels_mapping=None):
        time = np.arange(0, display_window)
        self.display_window = display_window
        self.queue_process = queue_process
        self.is_running_event = is_running_event
        self.channels_mapping = channels_mapping
        self.n_process = self.n_process if len(self.channels) > 1 else 1
        self.n_process = min(self.n_process, int(np.ceil(len(self.channels) / 2)))
        self.queue_plot = {i: ClearableQueue(maxwrite=2000, name="plot") for i in range(len(self.channels))}
        self.process_args_event = [mp.Event() for _ in range(self.n_process)]
        self.queue_process_args = [ClearableQueue(maxwrite=1) for _ in range(self.n_process)]
        self.parent.log_box.log(f"Starting stream with {self.n_process} processes...")
        self.fft_freqs = rfftfreq(display_window, 1 / self.acquisition_rate)
        self.display_options.set_file_params(self.channels)
        self.remover_options.new_stream(self.channels, self.process_args_event, self.queue_process_args)
        self.plot.initialize_data(
            self.stream_widget.server.buffer,
            time,
            self.channels,
            display_window,
            self.queue_plot,
            self.is_running_event,
            self.channels_mapping,
        )
        self.parent.toolbar.enable_filter_menu()
        self.parent.toolbar.radio_svd_filter_button.setEnabled(True)
        self.parent.toolbar.radio_notch_filter_button.setEnabled(False)
        self.update_filter("notch")
        self.running = True
        self.start_processing()

    def start_processing(self):
        self.processes = []
        for i in range(len(self.queue_process)):
            queues_tmp = {ch: self.queue_plot[ch] for ch in self.channels_mapping[i]}
            p = mp.Process(
                target=StreamProcessingWidget.process,
                args=(
                    self.queue_process[i],
                    queues_tmp,
                    self.process_args_event[i],
                    self.queue_process_args[i],
                    self.is_running_event,
                    self.display_window,
                    self.channels_mapping[i],
                    self.acquisition_rate,
                ),
                daemon=True,
            )
            self.processes.append(p)
        for p in self.processes:
            p.start()

    @property
    def acquisition_rate(self):
        return self.stream_widget.acquisition_rate

    @property
    def channels(self):
        return self.stream_widget.channels

    def stop_processing(self):
        for p in self.processes:
            p.terminate()
            p.join()
        self.processing = False

    def update_filter(self, name="notch"):
        # if self.timer.isActive():
        #     self.stop_processing()
        self.remover_options.update_filter(name)
        self.plot.update_filter(name)
        # self.plot.update_config_button(self.remover_options.get_current_config())
        processed = self.get_processed_channels()
        if processed is not None:
            self.display_options.display_processed_btn.setEnabled(True)
        else:
            self.display_options.display_processed_btn.setEnabled(False)
        self.remover_options.show_config("")
        # if self.streaming_data:
        #     self.start_processing()

    def set_paused(self, paused):
        if paused:
            self.plot.pause_plot(True)
        else:
            self.plot.pause_plot(False)
        self.paused = paused
