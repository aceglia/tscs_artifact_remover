from artifact_remover.app.gui import GUI
from PyQt5.QtWidgets import QApplication
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = GUI()
    app.exec()
