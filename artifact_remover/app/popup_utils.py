from PyQt5.QtWidgets import QMessageBox

def popup_warning_save(text, title, fct):
    wind = QMessageBox()
    wind.setText(text)
    wind.setWindowTitle(title)
    wind.setIcon(QMessageBox.Question)
    wind.setStandardButtons(QMessageBox.Save | QMessageBox.Ignore | QMessageBox.Cancel)
    wind.setDefaultButton(QMessageBox.Save)
    wind.buttonClicked.connect(fct)
    wind.exec_()

def popup_warning_continue(text, title, fct):
    wind = QMessageBox()
    wind.setText(text)
    wind.setWindowTitle(title)
    wind.setIcon(QMessageBox.Question)
    wind.setStandardButtons(QMessageBox.Ignore | QMessageBox.Cancel)
    wind.setDefaultButton(QMessageBox.Cancel)
    wind.buttonClicked.connect(fct)
    wind.exec_()

def popup_warning_split(text, title, fct):
    wind = QMessageBox()
    wind.setText(text)
    wind.setWindowTitle(title)
    wind.setIcon(QMessageBox.Question)
    wind.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
    wind.setDefaultButton(QMessageBox.Yes)
    wind.buttonClicked.connect(fct)
    wind.exec_()