import threading
from PyQt5.QtWidgets import (
    QLabel, QGridLayout, QDateEdit, QComboBox, QPushButton, QScrollArea, 
)
from PyQt5.QtCore import Qt, QDate, QObject, QThread, pyqtSignal, pyqtSlot
from datetime import date

from ..utils.widgets import CustomTabWidget
from ....data.project import DB, ProjectData


class _TEXT:
    LBL_INIT = 'no project selected!'
    LBL_ADD_DATA = 'add data'
    LBL_AVLBL_DATA = 'available data'
    LBL_ADD_EP1 = 'first epoch'
    LBL_ADD_EP2 = 'last epoch'
    LBL_ADD_FSEL = 'field'
    LBL_ADD_DSEL = 'data-source'
    BTN_ADD_TEXT = 'add'


class _STYLE:
    LBL_AUX = """
        QLabel {
            border: 1px solid rgb(200, 200, 200);
            border-radius: 3px;
        }
    """
    LBLS_ADD = """
        QLabel {
            padding-left: 5px;
        }
    """
    DATES_ADD = """
        QDateEdit {
            margin-left: 5px;
            margin-right: 5px;
        }
    """
    DDS_ADD = """
        QComboBox {
            margin-left: 5px;
            margin-right: 5px;
            padding-left: 5px;
        }
    """
    BTN_ADD = """
        QPushButton {
            margin-left: 5px;
            margin-right: 5px;
            background-color: rgb(0, 255, 150);
        }
    """


class Worker(QObject):
    def __init__(self, parent):
        super().__init__(parent)
        self.update_text = pyqtSignal(str)

    @pyqtSlot()
    def prj_add_data(self):
        pass



class DataTab(CustomTabWidget):
    def __init__(self, parent, store):
        super().__init__(parent, store)
        # internal variables
        self._ep1_add:str = None
        self._ep2_add:str = None
        self._flds:list[str] | str = None
        self._dids:list[str] | str = None

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
        # labels
        self._lbl_add = QLabel(_TEXT.LBL_ADD_DATA)
        self._lbl_add.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self._l.addWidget(self._lbl_add, 0, 0)
        self._lbl_aux1 = QLabel('')
        self._lbl_aux1.setStyleSheet(_STYLE.LBL_AUX)
        self._l.addWidget(self._lbl_aux1, 1, 0, 3, 5)
        self._lbl_ep1 = QLabel(_TEXT.LBL_ADD_EP1)
        self._lbl_ep1.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self._lbl_ep1.setStyleSheet(_STYLE.LBLS_ADD)
        self._l.addWidget(self._lbl_ep1, 1, 0)
        self._lbl_ep2 = QLabel(_TEXT.LBL_ADD_EP2)
        self._lbl_ep2.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self._lbl_ep2.setStyleSheet(_STYLE.LBLS_ADD)
        self._l.addWidget(self._lbl_ep2, 1, 1)
        self._lbl_fsel = QLabel(_TEXT.LBL_ADD_FSEL)
        self._lbl_fsel.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self._lbl_fsel.setStyleSheet(_STYLE.LBLS_ADD)
        self._l.addWidget(self._lbl_fsel, 1, 2)
        self._lbl_dsel = QLabel(_TEXT.LBL_ADD_DSEL)
        self._lbl_dsel.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self._lbl_dsel.setStyleSheet(_STYLE.LBLS_ADD)
        self._l.addWidget(self._lbl_dsel, 1, 3)

        # date stuff
        self._d1_add = QDateEdit(calendarPopup=True)
        self._d1_add.setDisplayFormat('dd.MM.yyyy')
        self._d1_add.setStyleSheet(_STYLE.DATES_ADD)
        self._d1_add.setDate(QDate(QDate.currentDate().year(), 1, 1))
        self._d1_add.userDateChanged.connect(self._first_epoch_add)
        self._l.addWidget(self._d1_add, 2, 0)
        self._d2_add = QDateEdit(calendarPopup=True)
        self._d2_add.setDisplayFormat('dd.MM.yyyy')
        self._d2_add.setStyleSheet(_STYLE.DATES_ADD)
        self._d2_add.setDate(QDate.currentDate())
        self._d2_add.userDateChanged.connect(self._last_epoch_add)
        self._l.addWidget(self._d2_add, 2, 1)
        self._ep1_add = self._d1_add.date().toString('yyyy-MM-dd')
        self._ep2_add = self._d2_add.date().toString('yyyy-MM-dd')

        #dropdowns
        self._dd_fld = QComboBox()
        self._dd_fld.setStyleSheet(_STYLE.DDS_ADD)
        fnames = self.store.project_data.fields[DB.TBL_FIELDS.COL_FIELDNAME]
        for fname in fnames:
            if self._flds is None:
                self._flds = fname
            self._dd_fld.addItem(fname)
        self._dd_fld.addItem('all')
        self._dd_fld.currentTextChanged.connect(self._fields_add)
        self._l.addWidget(self._dd_fld, 2, 2)
        self._dd_data = QComboBox()
        self._dd_data.setStyleSheet(_STYLE.DDS_ADD)
        self._reset_data_source_dropdown()
        self._dd_data.currentTextChanged.connect(self._data_sources_add)
        self._l.addWidget(self._dd_data, 2, 3)

        # add button
        self._btn_add = QPushButton()
        self._btn_add.setText(_TEXT.BTN_ADD_TEXT)
        self._btn_add.setStyleSheet(_STYLE.BTN_ADD)
        self._btn_add.clicked.connect(self._add_data)
        self._l.addWidget(self._btn_add, 2, 4)

        # project and data-interface output
        self._view_prcok = QScrollArea()
        self._view_prcok.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._view_prcok.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._lbl_prcok = QLabel('')
        self._lbl_prcok.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._view_prcok.setWidget(self._lbl_prcok)
        self._l.addWidget(self._view_prcok, 3, 0)
        self.store.project_data.add_data_interaction(
            ProjectData.add_data_success, self._add_data_success, self
        )

        self._view_pdi = QScrollArea()
        self._view_pdi.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._lbl_pdi = QLabel('')
        self._view_pdi.setWidget(self._lbl_pdi)
        self._l.addWidget(self._view_pdi, 3, 1)
        self.store.project_data.add_data_interaction(
            ProjectData.processed_interface, self._add_data_interface, self
        )

        self._view_pfld = QScrollArea()
        self._view_pfld.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._lbl_pfld = QLabel('')
        self._view_pfld.setWidget(self._lbl_pfld)
        self._l.addWidget(self._view_pfld, 3, 2)
        self.store.project_data.add_data_interaction(
            ProjectData.processed_field, self._add_data_field, self
        )

        self._view_prgr = QScrollArea()
        self._view_prgr.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._lbl_prgr = QLabel('')
        self._view_prgr.setWidget(self._lbl_prgr)
        self._l.addWidget(self._view_prgr, 3, 3, 1, 2)
        self.store.project_data.add_data_interaction(
            ProjectData.progress, self._add_data_progress, self
        )

        # available-data stuff
        self._lbl_avlbl = QLabel(_TEXT.LBL_AVLBL_DATA)
        self._lbl_avlbl.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self._l.addWidget(self._lbl_avlbl, 4, 0)
        self._lbl_aux2 = QLabel('')
        self._lbl_aux2.setStyleSheet(_STYLE.LBL_AUX)
        self._l.addWidget(self._lbl_aux2, 5, 0, 1, 5)

        self.setLayout(self._l)

    def init_tab(self):
        super().init_tab()
        self._init_lbl.setVisible(False)
        if self.store.data_interfaces:
            for di in self.store.data_interfaces:
                self.store.project_data.add_data_interface(di)
            self._reset_data_source_dropdown()

    def _reset_data_source_dropdown(self):
        self._dd_data.clear()
        if len(self.store.project_data.interfaces) == 0:
            self._dd_data.addItem('no data-source available')
        else:
            for did in self.store.project_data.interfaces.keys():
                if self._dids is None:
                    self._dids = did
                self._dd_data.addItem(did)
            self._dd_data.addItem('all')

    ############################################################################
    # handlers for adding data
    def _add_data_success(self, flag):
        if flag:
            state = 'OK'
        else:
            state = 'ERR'
        self._update_add_data_out(self._lbl_prcok, state, self._view_prcok)

    def _add_data_field(self, fname):
        self._update_add_data_out(self._lbl_pfld, fname, self._view_pfld)

    def _add_data_interface(self, iname):
        self._update_add_data_out(self._lbl_pdi, iname, self._view_pdi)

    def _add_data_progress(self, progr):
        self._update_add_data_out(self._lbl_prgr, progr, self._view_prgr)

    @staticmethod
    def _update_add_data_out(lbl, text, view):
        lbl.setText(lbl.text() + '\n' + text)
        view.verticalScrollBar().setValue(view.verticalScrollBar().maximum())

    ############################################################################
    # signal handlers
    def _first_epoch_add(self):
        self._ep1_add = self._d1_add.date().toString('yyyy-MM-dd')

    def _last_epoch_add(self):
        self._ep2_add = self._d2_add.date().toString('yyyy-MM-dd')

    def _fields_add(self):
        if self._dd_fld.currentText() == 'all':
            self._flds = self.store.project_data.fields[
                DB.TBL_FIELDS.COL_FIELDNAME
            ].values.tolist()
        else:            
            self._flds = self._dd_fld.currentText()

    def _data_sources_add(self):
        if self._dd_data.currentText() == 'all':
            self._dids = list(self.store.project_data.interfaces.keys())
        else:
            self._dids = self._dd_data.currentText()

    def _add_data(self):
        tad = threading.Thread(
            target=self.store.project_data.add_data, 
            args=(
                date.fromisoformat(self._ep1_add),
                date.fromisoformat(self._ep2_add),
            ),
            kwargs={'dids': self._dids,'fields': self._flds},
            daemon=True
        )
        tad.start()
