from PyQt5.QtWidgets import (
    QWidget, QLabel, QHBoxLayout, QGridLayout
)

from .utils.widgets import CustomTabWidget


class _TEXT:
    LBL_INIT = 'no project selected!'


class DataTab(CustomTabWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self._lm = QHBoxLayout()
        self._init_lbl = QLabel(_TEXT.LBL_INIT)
        self._lm.addWidget(self._init_lbl)

        #self._ll = QGridLayout()

        self.setLayout(self._lm)

    def init_tab(self):
        super().init_tab()
        self._init_lbl.setVisible(False)
