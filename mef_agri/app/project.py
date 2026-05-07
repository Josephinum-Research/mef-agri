import os
from PyQt5.QtWidgets import (
    QWidget, QGridLayout, QFileDialog, QPushButton, QComboBox, QLabel
)

from .map import MapView


class ProjectTab(QWidget):
    def __init__(self):
        super().__init__()
        # class internal variables
        self._pp:str = None  # path to folder where projects are located
        self._pdir:str = None  # path to selected project

        # project creation, selection, ...    
        self._btn_prj = QPushButton('project folder', self)
        self._btn_prj.clicked.connect(self.open_file_dialog)
        self._cbx_prj = QComboBox()
        self._cbx_prj.addItem('no project selected')
        self._cbx_prj.currentIndexChanged.connect(self.prj_selected)
        self._cbx_prj.blockSignals(True)

        mmap = MapView(self)

        # layout and adding widgets
        lbl_dummy = QLabel('')
        self._l = QGridLayout()
        self._l.addWidget(self._btn_prj, 0, 2)
        self._l.addWidget(self._cbx_prj, 1, 2)
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
            self._cbx_prj.clear()
            self._cbx_prj.addItem('select')
            for prj in os.listdir(self._pp):
                pdir = os.path.join(self._pp, prj)
                if os.path.isdir(pdir):
                    self._cbx_prj.addItem(prj)
            self._cbx_prj.blockSignals(False)

    def prj_selected(self):
        print(self._cbx_prj.currentText())
