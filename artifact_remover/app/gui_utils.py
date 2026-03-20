from PyQt5.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QWidget,
    QCheckBox
)

class ChannelSelecter(QWidget):
    def __init__(self, parent=None, channel_list=[], for_display=True):
        super().__init__()
        title = "Select channel(s) to display" if for_display else "Select channel(s) to process"
        self.setWindowTitle(title)
        self.checkbox_list = []
        self.parent = parent
        layout = QVBoxLayout()
        for channel in channel_list:
            self.checkbox_list.append(QCheckBox(channel))
            self.checkbox_list[-1].setChecked(True)
            layout.addWidget(self.checkbox_list[-1])
        self.select_all_button = QPushButton("Select all")
        self.select_all_button.clicked.connect(self.on_all_checked)
        self.unselect_all_button = QPushButton("Unselect all")
        self.unselect_all_button.clicked.connect(self.on_all_unchecked)
        layout.addWidget(self.select_all_button)
        layout.addWidget(self.unselect_all_button)
        button_name = 'Draw' if for_display else 'Apply'
        self.draw_button = QPushButton(button_name)
        self.draw_button.clicked.connect(self.parent.on_draw_clicked)
        layout.addWidget(self.draw_button)
        self.setLayout(layout)
        self.resize(300, 200)

    def on_all_unchecked(self):
        for checkbox in self.checkbox_list:
            checkbox.setChecked(False)
    
    def on_all_checked(self):
        for checkbox in self.checkbox_list:
            checkbox.setChecked(True)

    def get_selected_channels(self):
        selected_channels = []
        for c, checkbox in enumerate(self.checkbox_list):
            if checkbox.isChecked():
                selected_channels.append((c, checkbox.text()))
        return selected_channels

    def get_channel_idxs(self):
        channels = self.get_selected_channels()
        return [i[0] for i in channels]

    def get_channel_names(self):
        channels = self.get_selected_channels()
        return [i[1] for i in channels]


    def quit(self):
        self.close()