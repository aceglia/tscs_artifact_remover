from PyQt5.QtWidgets import QPlainTextEdit 
import datetime

class LogBox(QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)

    def log(self, message):
        self.appendPlainText(self._add_current_time(message))
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
    
    def _add_current_time(self, message):
        return f'[{str(datetime.datetime.now())}] {message}'

def ensure_list(x):
    return x if isinstance(x, list) else [x]

    


    


