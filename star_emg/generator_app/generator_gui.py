import os
from PyQt5.QtWidgets import (
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSplitter,
    QAction,
    QLineEdit,
    QToolBar,
    QInputDialog,
)
from PyQt5.QtCore import Qt
from ..app.gui_utils import LogBox
from ..app.file_dialog import LoadDialog, SaveDialog
from ..app.cache import Cache
from ..app.popup_utils import popup_warning_split, popup_warning_continue, popup_warning_save
from .artifact_widget import ArtifactWidget


class CustomToolBar(QToolBar):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        # self.load_files_button = QAction("Load EMG file")
        # self.load_files_button.triggered.connect(self.parent._load_file)
        # self.load_files_button.setEnabled(True)
        self.load_config_button = QAction("Load config")
        self.load_config_button.triggered.connect(self.parent._load_config)
        self.load_config_button.setEnabled(True)
        self.save_config_button = QAction("Save")
        self.save_config_button.triggered.connect(self.parent.save)
        self.save_config_button.setEnabled(True)
        self.save_as_config_button = QAction("Save As")
        self.save_as_config_button.triggered.connect(self.parent.save_as)
        self.save_as_config_button.setEnabled(True)
        self.quit_button = QAction("Quit")
        self.quit_button.triggered.connect(self.parent.quit)
        # self.addAction(self.load_files_button)
        # self.addAction(self.save_files_button)
        self.addAction(self.load_config_button)
        self.addAction(self.save_config_button)
        self.addAction(self.save_as_config_button)
        self.addSeparator()
        self.addAction(self.quit_button)
        # set fixed
        self.setMovable(False)


class GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Artifact Generator")
        self.log_box = LogBox()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        app_dir = os.getcwd()
        self._quit = False
        self._continue = False
        self.cache = Cache(cache_file=os.path.join(app_dir, "_generator_pychache.json"))
        self.toolbar = CustomToolBar(self)
        self.addToolBar(self.toolbar)
        self.generator_widget = ArtifactWidget(self)
        self._init_layout()
        self.show()
        self.saved_ok = True
        self._split = False
        self.save_as_popup = None
        self.config_save_path = None

    def _init_layout(self):
        self.clear_log_button = QPushButton("Clear Log")
        self.clear_log_button.clicked.connect(self.log_box.clear)
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.log_box)
        splitter.addWidget(self.clear_log_button)
        main_layout = QVBoxLayout()
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(self.generator_widget)
        main_splitter.addWidget(splitter)
        main_layout.addWidget(main_splitter)
        self.central_widget.setLayout(main_layout)
        self.save_dialog = None

    def _load_config(self):
        if not self.saved_ok:
            popup_warning_continue(
                "You didn't save your work, it will be erased, do you want to continue?", "Warning", self.popup_continue
            )
            if not self._continue:
                return
        LoadDialog(
            parent=self,
            caption="Load configuration file",
            filter="File type (*.json)",
            load_method=self.generator_widget.load_config,
        )

    def _save_config(self, file_path):
        self.log_box.log(f"Saving configuration file at: {file_path}")
        self.generator_widget.save_config(file_path)

    def quit(self):
        if not self.saved_ok:
            popup_warning_save("Do you want to save and exit?", "Quit", self.popup_button)
            if self._quit:
                self.close()
        else:
            self.close()

    def log(self, message):
        if self.log_box is not None:
            self.log_box.log(message)

    def popup_continue(self, button):
        if button.text() == "Ignore":
            self._continue = True
        elif button.text() == "Cancel":
            self._continue = False

    def popup_button(self, button):
        if button.text() == "Save":
            self._save_files()
            self._quit = True
        elif button.text() == "Ignore":
            self._quit = True
        elif button.text() == "Cancel":
            self._quit = False

    def save_as(self):
        self.save_dialog = SaveDialog(
            self,
            "Chose a path where to save the configuration.",
            filter="*.json",
            suffix=".json",
            save_method=self._save_config,
        )

    def show_split_windows(self, text):
        popup_warning_split(text, "File loading warning", self.popup_split)

    def popup_split(self, button):
        self._split = button.text()

    def save_close_config(self):
        if self.configuration.save_config():
            self.configuration.close()
        self.menu_bar.configuration_menu.setEnabled(True)
        self.menu_bar.run.setEnabled(True)

    def close_config(self):
        self.configuration.close_window()
        self.menu_bar.configuration_menu.setEnabled(True)
        self.menu_bar.run.setEnabled(True)
        self.menu_bar.trigger_action.setEnabled(False)
        self.menu_bar.gognio_action.setEnabled(False)

    def save(self):
        if self.save_dialog is None or self.save_dialog.filename is None:
            self.save_as()
            return
        try:
            self._save_config(self.save_dialog.filename)
        except Exception as e:
            self.log_box.log("Error occured while saving the files: ", e)

    def set_saved_ok(self, saved_ok):
        self.saved_ok = saved_ok
        if saved_ok:
            self.toolbar.save_config_button.setEnabled(False)
        else:
            self.toolbar.save_config_button.setEnabled(True)
