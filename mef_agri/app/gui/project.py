import os
from PyQt5.QtWidgets import (
    QWidget, QGridLayout, QFileDialog, QPushButton, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QTableWidget
)

from .map import MapView

class __TEXT__:
    BTN_SELPRJ = 'select project-directory'
    BTN_NEWPRJ = 'create new project'
    LBL_DBNAME = 'name of database'
    LBL_FTBLNAME = 'name of field-table'
    LBL_FCOLNAME = 'name of field-name-column'
    LBL_CRS = 'crs/epsg'
    DEF_DBNAME = 'project'
    DEF_FTBLNAME = 'fields'
    DEF_FCOLNAME = 'fname'
    DEF_CRS = '25833'


class ProjectTab(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        # class internal variables
        self._pp:str = None  # path to folder where project is located
        self._prj_exists:bool = False  # flag to indicate if current project already exists or is newly created

        # initializing the layouts
        self._lm = QHBoxLayout()  # main layout
        self._ll = QGridLayout()  # left layout where user is interacting
        for ci in range(5):
            self._ll.setColumnStretch(ci, 1)
        self._ll.setRowStretch(0, 1)  # area to select existing project or create a new one
        self._ll.setRowStretch(1, 1)  # empty space
        self._ll.setRowStretch(2, 3)  # area with db/gpkg settings
        self._ll.setRowStretch(3, 1)  # empty space
        self._ll.setRowStretch(4, 6)  # area where fields are listed

        # button to select existing project
        self._btn_selprj = QPushButton(__TEXT__.BTN_SELPRJ, self)
        self._btn_selprj.clicked.connect(self._open_file_dialog)
        self._ll.addWidget(self._btn_selprj, 0, 1)
        self._btn_newprj = QPushButton(__TEXT__.BTN_NEWPRJ, self)
        self._btn_newprj.clicked.connect(self._create_project)
        self._ll.addWidget(self._btn_newprj, 0, 3)

        # form layout and widgets with db/gpkg settings
        self._ldb = QFormLayout()
        self._lbl_dbname = QLabel(__TEXT__.LBL_DBNAME, self)
        self._inp_dbname = QLineEdit(self)
        self._inp_dbname.setText(__TEXT__.DEF_DBNAME)
        self._ldb.addRow(self._lbl_dbname, self._inp_dbname)
        self._lbl_ftblname = QLabel(__TEXT__.LBL_FTBLNAME, self)
        self._inp_ftblname = QLineEdit(self)
        self._inp_ftblname.setText(__TEXT__.DEF_FTBLNAME)
        self._ldb.addRow(self._lbl_ftblname, self._inp_ftblname)
        self._lbl_fcolname = QLabel(__TEXT__.LBL_FCOLNAME, self)
        self._inp_fcolname = QLineEdit(self)
        self._inp_fcolname.setText(__TEXT__.DEF_FCOLNAME)
        self._ldb.addRow(self._lbl_fcolname, self._inp_fcolname)
        self._lbl_crs = QLabel(__TEXT__.LBL_CRS, self)
        self._inp_crs = QLineEdit(self)
        self._inp_crs.setText(__TEXT__.DEF_CRS)
        self._inp_crs.setEnabled(False)  # NOTE make selectable later
        self._ldb.addRow(self._lbl_crs, self._inp_crs)
        self._ldb_widgets = [
            self._lbl_dbname, self._inp_dbname, self._lbl_ftblname, 
            self._inp_ftblname, self._lbl_fcolname, self._inp_fcolname,
            self._lbl_crs, self._inp_crs
        ]
        self._ll.addLayout(self._ldb, 2, 0, 1, 2)

        # table view of fields
        self._tbl_flds = QTableWidget(self)
        self._tbl_flds.setColumnCount(3)
        self._tbl_flds.setHorizontalHeaderLabels([
            'fid', self._inp_fcolname.text(), 'geometry'
        ])
        self._fld_widgets = [
            self._tbl_flds
        ]
        self._ll.addWidget(self._tbl_flds, 4, 0, 1, 5)

        # hide most widgets initially
        for wi in self._ldb_widgets + self._fld_widgets:
            wi.hide()

        # add remaining stuff
        self._lm.addLayout(self._ll, 1)
        self._lm.addWidget(MapView(self), 1)
        self.setLayout(self._lm)

    def _show_prjcont(self):
        for wi in self._ldb_widgets + self._fld_widgets:
            wi.show()
        if self._prj_exists:
            self._inp_dbname.setEnabled(False)
            self._inp_ftblname.setEnabled(False)
            self._inp_fcolname.setEnabled(False)

    def _create_project(self):
        self._prj_exists = False
        self._show_prjcont()

    def _open_file_dialog(self):
        fd = QFileDialog(self)
        fd.setWindowTitle('select folder with projects')
        fd.setFileMode(QFileDialog.FileMode.DirectoryOnly)
        fd.show()

        if fd.exec():
            self._pp = fd.selectedFiles()[0]
            self._prj_exists = True
            self._show_prjcont()
