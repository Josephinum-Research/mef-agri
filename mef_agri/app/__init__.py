from PyQt5.QtWidgets import QWidget, QGridLayout, QTabWidget

from .project import ProjectTab


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('MEF-Agri')
        self.showMaximized()
        self._l = QGridLayout()
        self._tabs = QTabWidget()
        self._tabs.addTab(ProjectTab(), 'Project')
        self._l.addWidget(self._tabs)
        self.setLayout(self._l)
