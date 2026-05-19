import os
from PyQt5.QtWidgets import (
    QWidget, QGridLayout, QFileDialog, QPushButton, QLabel
)

from .map import MapView
from .utils.widgets import ComboBox


class __TEXT__:
    SEL_APPROACH = 'select approach'
    SEL_APPROACH_EXISTING = 'existing project'
    SEL_APPROACH_NEW = 'create new project'
    SEL_PRJFOLDER = 'select directory'


class ProjectTab(QWidget):
    def __init__(self):
        super().__init__()
        # class internal variables
        self._pp:str = None  # path to folder where project is located

        # project creation, selection, ...
        self._cbx_prj = ComboBox(self)
        self._cbx_prj.addItem(__TEXT__.SEL_APPROACH_EXISTING)
        self._cbx_prj.addItem(__TEXT__.SEL_APPROACH_NEW)
        self._cbx_prj.setPlaceholderText(__TEXT__.SEL_APPROACH)
        self._cbx_prj.setCurrentIndex(-1)
        self._cbx_prj.currentIndexChanged.connect(self._appr_sel)

        self._btn_prj = QPushButton(__TEXT__.SEL_PRJFOLDER, self)
        self._btn_prj.clicked.connect(self.open_file_dialog)

        mmap = MapView(self)

        # layout and adding widgets
        lbl_dummy = QLabel('')
        self._l = QGridLayout()
        # addWidget(widget, row, column)
        self._l.addWidget(self._cbx_prj, 0, 2)
        self._l.addWidget(self._btn_prj, 1, 2)
        self._btn_prj.hide()
        # addWidget(widget, fromRow, fromColumn, rowSpan, columnSpan)
        self._l.addWidget(lbl_dummy, 0, 0, 5, 2)
        self._l.addWidget(lbl_dummy, 0, 4, 5, 2)
        self._l.addWidget(mmap, 0, 6, 5, 5)
        self.setLayout(self._l)

    def open_file_dialog(self):
        fd = QFileDialog(self)
        fd.setWindowTitle('select folder with projects')
        fd.setFileMode(QFileDialog.FileMode.DirectoryOnly)
        fd.show()

        if fd.exec():
            self._pp = fd.selectedFiles()[0]
            print(self._pp)

    def _appr_sel(self):
        if self._cbx_prj.currentText() == __TEXT__.SEL_APPROACH_EXISTING:
            self._btn_prj.show()
        elif self._cbx_prj.currentText() == __TEXT__.SEL_APPROACH_NEW:
            self._btn_prj.hide()
