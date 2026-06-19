import geopandas as gpd
from PyQt5.QtWidgets import (
    QWidget, QGridLayout, QFileDialog, QPushButton, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QDialog, QMessageBox
)
from shapely.geometry import Polygon

from .map import MapView
from .conn.server import WebsocketServer
from .conn.msgs import Messages
from ...data.project import ProjectData, DB
from ...utils.misc import search_file

class _TEXT:
    BTN_SELPRJ = 'select project-directory'
    BTN_NEWPRJ = 'create new project'
    BTN_CREATEPRJ = 'create'
    LBL_DBNAME = 'name of database'
    LBL_CRS = 'crs/epsg'
    LBL_SELPRJ_NONE = 'no project selected yet'
    LBL_SELPRJ_PATH = 'selected project: "{pp}"'
    DEF_DBNAME = 'project'
    DEF_FTBLNAME = 'fields'
    DEF_FCOLNAME = 'fname'
    DEF_CRS = '25833'
    DLG_NEWPRJ_TITLE = 'create new project'

class _CustomErrorDialog(QMessageBox):
    def __init__(self, msg):
        super().__init__()
        self.setWindowTitle('project-error')
        self.setText(msg)
        self.setIcon(QMessageBox.Critical)

class _ErrorDialogs:
    @staticmethod
    def no_gpkg_in_pdir():
        msg = 'No .gpkg available in the selected directory!'
        dlg = _CustomErrorDialog(msg)
        dlg.exec()

    @staticmethod
    def gpkg_exists_in_pdir():
        msg = 'A .gpkg already exists in the selected directory!'
        dlg = _CustomErrorDialog(msg)
        dlg.exec()

    @staticmethod
    def no_project_selected():
        msg = 'No project has been created or selected yet!'
        dlg = _CustomErrorDialog(msg)
        dlg.exec()

def select_prj_dir() -> str | None:
    fd = QFileDialog()
    fd.setWindowTitle('select project directory')
    fd.setFileMode(QFileDialog.FileMode.DirectoryOnly)
    fd.show()
    if fd.exec():
        return fd.selectedFiles()[0]
    else:
        return None


class NewProject(QDialog):
    def __init__(self):
        super().__init__()
        self._pp:str = None
        self._dbn:str = None

        # UI
        self.setWindowTitle(_TEXT.DLG_NEWPRJ_TITLE)
        self._ldb = QFormLayout()
        self._lbl_dbname = QLabel(_TEXT.LBL_DBNAME, self)
        self._inp_dbname = QLineEdit(self)
        self._inp_dbname.textChanged.connect(self._get_input_dbname)
        self._ldb.addRow(self._lbl_dbname, self._inp_dbname)
        self._lbl_crs = QLabel(_TEXT.LBL_CRS, self)
        self._inp_crs = QLineEdit(self)
        self._inp_crs.setText(_TEXT.DEF_CRS)
        self._inp_crs.setEnabled(False)  # NOTE make selectable later
        self._ldb.addRow(self._lbl_crs, self._inp_crs)
        self._btn_create = QPushButton(_TEXT.BTN_CREATEPRJ, self)
        self._btn_create.clicked.connect(self.accept)
        self._ldb.addRow(self._btn_create)
        self.setLayout(self._ldb)

    @property
    def project_directory(self) -> str:
        """
        :return: selected project directory
        :rtype: str
        """
        return self._pp
    
    @property
    def project_dbname(self) -> str:
        """
        :return: name of project-database
        :rtype: str
        """
        return self._dbn
    
    def _get_input_dbname(self):
        self._dbn = self._inp_dbname.text()

    def exec(self):
        pp_temp = select_prj_dir()
        if pp_temp is None:
            return

        self._pp = pp_temp
        return super().exec()


class ProjectTab(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        # reference to project data/geopackage
        self._pd:ProjectData = None

        # websocket stuff
        self._wss:WebsocketServer = parent.websocket_server
        self._wss.register_handler(self._add_field, Messages.GotDrawnField)
        self._wss.register_handler(self._del_field, Messages.GotDeleteField)

        # initializing the layouts
        self._lm = QHBoxLayout()  # main layout
        self._ll = QGridLayout()  # left layout where user is interacting
        for ci in range(5):
            self._ll.setColumnStretch(ci, 1)
        self._ll.setRowStretch(0, 1)  # area to select existing project or create a new one
        self._ll.setRowStretch(1, 1)  # empty space
        self._ll.setRowStretch(2, 1)  # area where selected project path is shown
        self._ll.setRowStretch(3, 1)  # empty space
        self._ll.setRowStretch(4, 6)  # area where fields are listed

        # button to select existing project
        self._btn_selprj = QPushButton(_TEXT.BTN_SELPRJ, self)
        self._btn_selprj.clicked.connect(self._sel_project)
        self._ll.addWidget(self._btn_selprj, 0, 1)
        self._btn_newprj = QPushButton(_TEXT.BTN_NEWPRJ, self)
        self._btn_newprj.clicked.connect(self._new_project)
        self._ll.addWidget(self._btn_newprj, 0, 3)

        # show path of selected project
        self._lbl_selprj = QLabel(_TEXT.LBL_SELPRJ_NONE, self)
        self._ll.addWidget(self._lbl_selprj, 2, 0, 1, 5)

        # add remaining stuff
        self._lm.addLayout(self._ll, 1)
        self._lm.addWidget(MapView(self), 1)
        self.setLayout(self._lm)

    @property
    def selected_project(self) -> ProjectData:
        return self._pd

    def _show_prjcont(self):
        self._lbl_selprj.setText(
            _TEXT.LBL_SELPRJ_PATH.format(pp=self._pd.directory)
        )
        if len(self._pd.fields) > 0:
            fldInfo = Messages.SendFields()
            fldInfo.field_names = self._pd.fields[
                DB.TBL_FIELDS.COL_FIELDNAME
            ].values.tolist()
            coords = []
            f3857 = self._pd.fields.to_crs(epsg=3857, inplace=False)
            for fname in fldInfo.field_names:
                gdf = f3857[f3857[DB.TBL_FIELDS.COL_FIELDNAME] == fname]
                coords.append(gdf.geometry.get_coordinates().values.tolist())
            fldInfo.coordinates = coords
            self._wss.send_messages(fldInfo)

    def _new_project(self):
        dlg = NewProject()
        if dlg.exec():
            dbn = search_file(dlg.project_directory, '.gpkg', which='first')
            if dbn is not None:
                _ErrorDialogs.gpkg_exists_in_pdir()
                return
            
            self._pd = ProjectData(dlg.project_directory, dlg.project_dbname)
            self._pd.initialize()
            self._pd.create_tables()
            self._show_prjcont()

    def _sel_project(self):
        pdir = select_prj_dir()
        dbn = search_file(pdir, '.gpkg', which='first')
        if dbn is None:
            _ErrorDialogs.no_gpkg_in_pdir()
            return
        
        self._pd = ProjectData(pdir, dbn)
        self._show_prjcont()

    def _add_field(self, msg:Messages.GotDrawnField):
        if self._pd is None:
            _ErrorDialogs.no_project_selected()
            return
        
        gdf = gpd.GeoDataFrame(
            data={
                DB.TBL_FIELDS.COL_FIELDNAME: [msg.field_name],
                DB.TBL_FIELDS.COL_GEOM: [Polygon(shell=msg.points[0])]
            },
            crs=msg.epsg
        )
        self._pd.insert_from_gdf(DB.TBL_FIELDS.NAME, gdf)

    def _del_field(self, msg:Messages.GotDeleteField):
        if self._pd is None:
            _ErrorDialogs.no_project_selected()
            return
        
        self._pd.execute(
            DB.TBL_FIELDS.DEL_BY_FIELDNAME.format(fld=msg.field_name)
        )
