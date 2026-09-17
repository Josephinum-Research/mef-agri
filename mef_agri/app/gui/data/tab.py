from PyQt5.QtWidgets import (
    QLabel, QGridLayout, QDateEdit, QComboBox, QPushButton, QTextEdit
)
from PyQt5.QtCore import (
    Qt, QDate, QObject, QThread, pyqtSignal, pyqtSlot
)
from datetime import date

from ..utils.widgets import NonProjectTab
from ....data.project import DB
from ..project import ProjectDataGUI


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
    TXT_ADD = """
        QTextEdit {
            background-color: rgb(230, 230, 230);
            border: 0px solid transparent;
        }
    """

################################################################################
# WORKER to add data to project
class PrjAddDataWorker(QObject):
    """
    Connection between :class:`mef_agri.app.gui.project.ProjectDataGUI` and 
    :class:`DataTab`
    """
    fieldChanged = pyqtSignal(str)
    interfaceChanged = pyqtSignal(str)
    progressUpdate = pyqtSignal(str)
    addDataError = pyqtSignal(str)
    addDataSuccess = pyqtSignal(bool)
    addDataFinished = pyqtSignal()

    def __init__(self, prjd:ProjectDataGUI, parent=None):
        super().__init__(parent)
        self._pd:ProjectDataGUI = prjd
        self._pd.register_add_data_interaction(
            ProjectDataGUI.processed_field, self._pfield_changed
        )
        self._pd.register_add_data_interaction(
            ProjectDataGUI.processed_interface, self._pintf_changed
        )
        self._pd.register_add_data_interaction(
            ProjectDataGUI.progress, self._progr_update
        )
        self._pd.register_add_data_interaction(
            ProjectDataGUI.add_data_error, self._add_data_error
        )
        self._pd.register_add_data_interaction(
            ProjectDataGUI.add_data_success, self._add_data_success
        )

        # internal variables
        self._ep1:date = None
        self._ep2:date = None
        self._dids:list[str] = None
        self._flds:list[str] = None

    @property
    def first_epoch(self) -> date:
        return self._ep1
    
    @first_epoch.setter
    def first_epoch(self, val):
        if isinstance(val, str):
            self._ep1 = date.fromisoformat(val)
        elif isinstance(val, date):
            self._ep1 = val

    @property
    def last_epoch(self) -> date:
        return self._ep2
    
    @last_epoch.setter
    def last_epoch(self, val):
        if isinstance(val, str):
            self._ep2 = date.fromisoformat(val)
        elif isinstance(val, date):
            self._ep2 = val

    @property
    def interfaces(self) -> list[str]:
        return self._dids
    
    @interfaces.setter
    def interfaces(self, val):
        if isinstance(val, str):
            self._dids = [val]
        elif isinstance(val, list):
            self._dids = val

    @property
    def fields(self) -> list[str]:
        return self._flds
    
    @fields.setter
    def fields(self, val):
        if isinstance(val, str):
            self._flds = [val]
        elif isinstance(val, list):
            self._flds = val

    @pyqtSlot(str)
    def _pfield_changed(self, fname):
        self.fieldChanged.emit(fname)

    @pyqtSlot(str)
    def _pintf_changed(self, iname):
        self.interfaceChanged.emit(iname)

    @pyqtSlot(str)
    def _progr_update(self, pstate):
        self.progressUpdate.emit(pstate)

    @pyqtSlot(str)
    def _add_data_error(self, err):
        self.addDataError.emit(err)

    @pyqtSlot(bool)
    def _add_data_success(self, succ):
        self.addDataSuccess.emit(succ)

    def prj_add_data(self):
        self._pd.add_data(
            self.first_epoch, self.last_epoch, dids=self._dids, 
            fields=self._flds
        )
        self.addDataFinished.emit()


################################################################################
# TAB
class DataTab(NonProjectTab):
    def __init__(self, parent, store):
        super().__init__(parent, store)
        # internal variables
        self._ep1_add:str = None
        self._ep2_add:str = None
        self._flds:list[str] | str = None
        self._dids:list[str] | str = None

        # initialize grid layout and set extents
        self.layout_main = QGridLayout()
        for ci in range(5):
            self.layout_main.setColumnStretch(ci, 1)
        self.layout_main.setRowStretch(0, 1)  # (add data area)-label
        self.layout_main.setRowStretch(1, 1)  # labels for start/stop-date, fields and data sources
        self.layout_main.setRowStretch(2, 1)  # selections for start/stop-date, fields and data sources
        self.layout_main.setRowStretch(3, 10)  # area to show outputs from interfaces
        self.layout_main.setRowStretch(4, 1)  # (show data area)-label
        self.layout_main.setRowStretch(5, 20)  # area for widget containing available data
        
        # add widgets to grid layout
        # add-data stuff
        # labels
        self._lbl_add = QLabel(_TEXT.LBL_ADD_DATA)
        self._lbl_add.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.layout_main.addWidget(self._lbl_add, 0, 0)
        self._lbl_aux1 = QLabel('')
        self._lbl_aux1.setStyleSheet(_STYLE.LBL_AUX)
        self.layout_main.addWidget(self._lbl_aux1, 1, 0, 3, 5)
        self._lbl_ep1 = QLabel(_TEXT.LBL_ADD_EP1)
        self._lbl_ep1.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self._lbl_ep1.setStyleSheet(_STYLE.LBLS_ADD)
        self.layout_main.addWidget(self._lbl_ep1, 1, 0)
        self._lbl_ep2 = QLabel(_TEXT.LBL_ADD_EP2)
        self._lbl_ep2.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self._lbl_ep2.setStyleSheet(_STYLE.LBLS_ADD)
        self.layout_main.addWidget(self._lbl_ep2, 1, 1)
        self._lbl_fsel = QLabel(_TEXT.LBL_ADD_FSEL)
        self._lbl_fsel.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self._lbl_fsel.setStyleSheet(_STYLE.LBLS_ADD)
        self.layout_main.addWidget(self._lbl_fsel, 1, 2)
        self._lbl_dsel = QLabel(_TEXT.LBL_ADD_DSEL)
        self._lbl_dsel.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self._lbl_dsel.setStyleSheet(_STYLE.LBLS_ADD)
        self.layout_main.addWidget(self._lbl_dsel, 1, 3)

        # date stuff
        self._d1_add = QDateEdit(calendarPopup=True)
        self._d1_add.setDisplayFormat('dd.MM.yyyy')
        self._d1_add.setStyleSheet(_STYLE.DATES_ADD)
        self._d1_add.setDate(QDate(QDate.currentDate().year(), 1, 1))
        self._d1_add.userDateChanged.connect(self._first_epoch_add)
        self.layout_main.addWidget(self._d1_add, 2, 0)
        self._d2_add = QDateEdit(calendarPopup=True)
        self._d2_add.setDisplayFormat('dd.MM.yyyy')
        self._d2_add.setStyleSheet(_STYLE.DATES_ADD)
        self._d2_add.setDate(QDate.currentDate())
        self._d2_add.userDateChanged.connect(self._last_epoch_add)
        self.layout_main.addWidget(self._d2_add, 2, 1)
        self._ep1_add = self._d1_add.date().toString('yyyy-MM-dd')
        self._ep2_add = self._d2_add.date().toString('yyyy-MM-dd')

        #dropdowns
        self._dd_fld = QComboBox()
        self._dd_fld.setStyleSheet(_STYLE.DDS_ADD)
        self._dd_fld.currentTextChanged.connect(self._fields_add)
        self.layout_main.addWidget(self._dd_fld, 2, 2)
        self._dd_data = QComboBox()
        self._dd_data.setStyleSheet(_STYLE.DDS_ADD)
        self._reset_data_source_dropdown()
        self._dd_data.currentTextChanged.connect(self._data_sources_add)
        self.layout_main.addWidget(self._dd_data, 2, 3)

        # project and data-interface output
        def init_txt():
            te = QTextEdit()
            te.setReadOnly(True)
            te.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            te.setStyleSheet(_STYLE.TXT_ADD)
            return te
        self._txt_prcok = init_txt()
        self.layout_main.addWidget(self._txt_prcok, 3, 0)
        self._txt_pdi = init_txt()
        self.layout_main.addWidget(self._txt_pdi, 3, 1)
        self._txt_pfld = init_txt()
        self.layout_main.addWidget(self._txt_pfld, 3, 2)
        self._txt_prgr = init_txt()
        self.layout_main.addWidget(self._txt_prgr, 3, 3, 1, 2)

        # add button
        self._btn_add = QPushButton()
        self._btn_add.setText(_TEXT.BTN_ADD_TEXT)
        self._btn_add.setStyleSheet(_STYLE.BTN_ADD)
        self._btn_add.clicked.connect(self._add_data)
        self.layout_main.addWidget(self._btn_add, 2, 4)

        # available-data stuff
        self._lbl_avlbl = QLabel(_TEXT.LBL_AVLBL_DATA)
        self._lbl_avlbl.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.layout_main.addWidget(self._lbl_avlbl, 4, 0)
        self._lbl_aux2 = QLabel('')
        self._lbl_aux2.setStyleSheet(_STYLE.LBL_AUX)
        self.layout_main.addWidget(self._lbl_aux2, 5, 0, 1, 5)

    def init_tab(self):
        super().init_tab()
        
        # set fields in comobbox
        fnames = self.store.project_data.fields[DB.TBL_FIELDS.COL_FIELDNAME]
        for fname in fnames:
            if self._flds is None:
                self._flds = fname
            self._dd_fld.addItem(fname)
        self._dd_fld.addItem('all')

        # set data sources/interfaces
        if self.store.data_interfaces:
            for di in self.store.data_interfaces:
                self.store.project_data.add_data_interface(di)
            self._reset_data_source_dropdown()

        # very good explanation of multi-threading in PyQT
        # https://realpython.com/python-pyqt-qthread/
        self._tprj = QThread(self)
        self._wprj = PrjAddDataWorker(self.store.project_data)
        # connect ui-stuff
        self._wprj.fieldChanged.connect(self._add_data_field)
        self._wprj.interfaceChanged.connect(self._add_data_interface)
        self._wprj.progressUpdate.connect(self._add_data_progress)
        self._wprj.addDataSuccess.connect(self._add_data_success)
        # move worker to thread
        self._wprj.moveToThread(self._tprj)
        # connect other stuff
        self._tprj.started.connect(self._wprj.prj_add_data)
        self._wprj.addDataFinished.connect(self._tprj.quit)

    def _reset_data_source_dropdown(self):
        if not self.store.project_data:
            return
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
        self._update_add_data_out(self._txt_prcok, state)

    def _add_data_field(self, fname):
        self._update_add_data_out(self._txt_pfld, fname)

    def _add_data_interface(self, iname):
        self._update_add_data_out(self._txt_pdi, iname)

    def _add_data_progress(self, progr):
        self._update_add_data_out(self._txt_prgr, progr)

    @staticmethod
    def _update_add_data_out(w:QTextEdit, text:str):
        if w.toPlainText():
            txt = w.toPlainText() + '\n' + text
        else:
            txt = text
        w.setText(txt)
        w.verticalScrollBar().setValue(w.verticalScrollBar().maximum())

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
        # cleart text output
        self._txt_prcok.clear()
        self._txt_pfld.clear()
        self._txt_pdi.clear()
        self._txt_prgr.clear()
        # provide info to worker
        self._wprj.first_epoch = self._ep1_add
        self._wprj.last_epoch = self._ep2_add
        self._wprj.fields = self._flds
        self._wprj.interfaces = self._dids
        # start thread
        self._tprj.start()
