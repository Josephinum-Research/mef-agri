from PyQt5.QtWidgets import QWidget, QHBoxLayout, QTabWidget, QSizePolicy

from .project import ProjectTab


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('MEF-Agri')
        self.showMaximized()
        self._l = QHBoxLayout()
        self._tabs = QTabWidget()
        self._tabs.addTab(ProjectTab(self), 'Project')
        self._l.addWidget(self._tabs)
        self.setLayout(self._l)