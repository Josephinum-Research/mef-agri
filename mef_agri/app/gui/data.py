from PyQt5.QtWidgets import (
    QLabel, QHBoxLayout, QGridLayout, QDateEdit
)
from PyQt5.QtCore import Qt

from .utils.widgets import CustomTabWidget
from .map import MapView


class _TEXT:
    LBL_INIT = 'no project selected!'
    LBL_ADD_DATA = 'add data'
    LBL_AVLBL_DATA = 'available data'
    LBL_ADD_EP1 = 'first epoch'
    LBL_ADD_EP2 = 'last epoch'


class _STYLE:
    LBL_AUX = """
        QLabel {
            border: 1px solid rgb(200, 200, 200);
            border-radius: 3px
        }
    """


class DataTab(CustomTabWidget):
    def __init__(self, parent, store):
        super().__init__(parent, store)
        # initialize grid layout and set extents
        self._l = QGridLayout()
        for ci in range(5):
            self._l.setColumnStretch(ci, 1)
        self._l.setRowStretch(0, 1)  # (add data area)-label
        self._l.setRowStretch(1, 1)  # labels for start/stop-date, fields and data sources
        self._l.setRowStretch(2, 1)  # selections for start/stop-date, fields and data sources
        self._l.setRowStretch(3, 10)  # area to show outputs from interfaces
        self._l.setRowStretch(4, 1)  # (show data area)-label
        self._l.setRowStretch(5, 20)  # area for widget containing available data
        
        # initial label which is removed when a project is selected
        self._init_lbl = QLabel(_TEXT.LBL_INIT)
        self._init_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._l.addWidget(self._init_lbl, 3, 0)

        # add widgets to grid layout
        # add-data stuff
        self._lbl_add = QLabel(_TEXT.LBL_ADD_DATA)
        self._lbl_add.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self._l.addWidget(self._lbl_add, 0, 0)
        self._lbl_aux1 = QLabel('')
        self._lbl_aux1.setStyleSheet(_STYLE.LBL_AUX)
        self._l.addWidget(self._lbl_aux1, 1, 0, 3, 5)
        self._lbl_ep1 = QLabel(_TEXT.LBL_ADD_EP1)
        self._lbl_ep1.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self._l.addWidget(self._lbl_ep1, 1, 0)
        self._lbl_ep1 = QLabel(_TEXT.LBL_ADD_EP2)
        self._lbl_ep1.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self._l.addWidget(self._lbl_ep1, 1, 1)

        # available-data stuff
        self._lbl_avlbl = QLabel(_TEXT.LBL_AVLBL_DATA)
        self._lbl_avlbl.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self._l.addWidget(self._lbl_avlbl, 4, 0)
        self._lbl_aux2 = QLabel('')
        self._lbl_aux2.setStyleSheet(_STYLE.LBL_AUX)
        self._l.addWidget(self._lbl_aux2, 5, 0, 1, 5)
        #self._d1 = QDateEdit(self._ll, calendarPopup=True)

        self.setLayout(self._l)

    def init_tab(self):
        super().init_tab()
        self._init_lbl.setVisible(False)
        if self.store.data_interfaces:
            for di in self.store.data_interfaces:
                self.store.project_data.add_data_interface(di)
