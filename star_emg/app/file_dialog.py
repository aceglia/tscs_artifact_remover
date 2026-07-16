from PyQt5.QtWidgets import QFileDialog

import os

kwargs = {}
if "SNAP" in os.environ:
    kwargs["options"] = QFileDialog.DontUseNativeDialog


class LoadDialog:
    """
    Helper class to load files from the file system.
    """

    def __init__(self, parent=None, caption="", filter="Any(*)", load_method=None):
        try:
            cache_dir = parent.cache.get_from_cache("last_dir")
            self.filename, _ = QFileDialog.getOpenFileName(
                parent=parent, caption=caption, filter=filter, directory=cache_dir
            )
            if self.filename is None or self.filename == "":
                return
            parent.log_box.log(f"Loading file {self.filename}...")
            parent.cache.set_to_cache("last_dir", os.path.dirname(self.filename))
            if load_method is not None:
                load_method(self.filename)
        except Exception as e:
            parent.log_box.log(f"Error occured while loading the file: {str(repr(e))}")


class LoadFolderDialog:
    """
    Helper class to load folders from the file system.
    """

    def __init__(self, parent=None, caption="", dir=None):
        if dir is not None:
            kwargs["directory"] = dir

        dialog = QFileDialog(parent=parent, caption=caption, **kwargs)
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setFileMode(QFileDialog.DirectoryOnly)

        self.filename = ""
        if not dialog.exec_():
            return

        self.filename = dialog.selectedFiles()[0]
        if self.filename is None or self.filename == "":
            self.filename = ""
            return

        kwargs["directory"] = os.path.dirname(self.filename)


class SaveDialog:
    """
    Helper class to save files to the file system.
    """

    def __init__(self, parent=None, caption="", filter="Any(*)", suffix="", dir=None, save_method=None):
        cache_dir = parent.cache.get_from_cache("last_dir")
        dialog = QFileDialog(parent=parent, caption=caption, filter=filter, directory=cache_dir)

        dialog.setDefaultSuffix(suffix)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setFileMode(QFileDialog.AnyFile)
        self.filename = None
        self.save_method = save_method
        ### Must pass via .exec_ method to apply the default suffix .json
        if not dialog.exec_():
            return

        self.filename = dialog.selectedFiles()[0]
        parent.cache.set_to_cache("last_dir", os.path.dirname(self.filename))

        kwargs["directory"] = os.path.dirname(self.filename)

        self.save_file()

    def save_file(self):
        if self.save_method is not None:
            return self.save_method(self.filename)
        else:
            return False
