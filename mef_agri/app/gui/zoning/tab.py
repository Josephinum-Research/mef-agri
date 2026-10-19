from PyQt5.QtWidgets import (
    QGridLayout, QLabel, QDateEdit, QLineEdit, QPushButton
)
from PyQt5.QtCore import QDate
import numpy as np
from geopandas import GeoDataFrame

from ..conn.msgs import Messages
from ..utils.widgets import NonProjectTab, ComboBox
from ....data.project import DB as PrjDB
from ....evaluation.zoning import Zoning


class _TEXT:
    LBL_SELFLD_INIT = 'select field in map'
    LBL_VALFROM = 'valid from'
    LBL_VALUNTIL = 'valid until'
    LBL_OBJRES = '[m] object resolution'
    INP_ORES_DEF = '25.0'
    DD_ZMETH_INIT = 'select zoning method'
    DD_ZMETH_ONEZONE = 'one zone'
    DD_ZMETH_FROMVEC = 'from vector data'
    DD_ZMETH_REGGRID = 'regular grid'
    BTN_RENDER = 'render'
    BTN_SAVE = 'save'


class _STYLE:
    DATES_VALID = """
        QDateEdit {
            margin-left: 5px;
            margin-right: 5px;
        }
    """


class ZoningTab(NonProjectTab):
    def __init__(self, parent, store):
        super().__init__(parent, store)
        self._active_field:str = None

        # UI stuff
        self.layout_main = QGridLayout()
        for ci in range(5):
            self.layout_main.setColumnStretch(ci, 1)
        self.layout_main.setRowStretch(0, 1)
        self.layout_main.setRowStretch(1, 1)
        self.layout_main.setRowStretch(2, 1)
        self.layout_main.setRowStretch(3, 1)
        self.layout_main.setRowStretch(4, 1)
        self.layout_main.setRowStretch(5, 10)

        self._lbl_sfld = QLabel(_TEXT.LBL_SELFLD_INIT)
        self.layout_main.addWidget(self._lbl_sfld, 0, 0)
        self._lbl_vfrom = QLabel(_TEXT.LBL_VALFROM)
        self.layout_main.addWidget(self._lbl_vfrom, 1, 0)
        self._d_vfrom = QDateEdit(calendarPopup=True)
        self._d_vfrom.setDisplayFormat('dd.MM.yyyy')
        self._d_vfrom.setStyleSheet(_STYLE.DATES_VALID)
        self._d_vfrom.userDateChanged.connect(self._set_first_epoch)
        self.layout_main.addWidget(self._d_vfrom, 1, 1)
        self._lbl_vuntil = QLabel(_TEXT.LBL_VALUNTIL)
        self.layout_main.addWidget(self._lbl_vuntil, 2, 0)
        self._d_vuntil = QDateEdit(calendarPopup=True)
        self._d_vuntil.setDisplayFormat('dd.MM.yyyy')
        self._d_vuntil.setStyleSheet(_STYLE.DATES_VALID)
        self._d_vuntil.userDateChanged.connect(self._set_second_epoch)
        self.layout_main.addWidget(self._d_vuntil, 2, 1)
        self._dd_zmeth = ComboBox(_TEXT.DD_ZMETH_INIT)
        self._dd_zmeth.currentTextChanged.connect(self._zmeth_selected)
        self.layout_main.addWidget(self._dd_zmeth, 3, 0)

        # UI elements dependent on zoning method
        self._lbl_ores = QLabel(_TEXT.LBL_OBJRES)
        self._lbl_ores.setVisible(False)
        self.layout_main.addWidget(self._lbl_ores, 3, 4)
        self._inp_ores = QLineEdit()
        self._inp_ores.setText(_TEXT.INP_ORES_DEF)
        self._inp_ores.setVisible(False)
        self.layout_main.addWidget(self._inp_ores, 3, 3)
        self._btn_rndr = QPushButton(_TEXT.BTN_RENDER)
        self._btn_rndr.setVisible(False)
        self._btn_rndr.clicked.connect(self._render_zones)
        self.layout_main.addWidget(self._btn_rndr, 4, 3)
        self._btn_save = QPushButton(_TEXT.BTN_SAVE)
        self._btn_save.setVisible(False)
        self._btn_save.clicked.connect(self._save_zones)
        self.layout_main.addWidget(self._btn_save, 4, 4)
        # TODO button to select vector data file

        self._reset_date_widgets()
        self._zw1 = [
            self._lbl_vfrom, self._d_vfrom, self._lbl_vuntil, self._d_vuntil,
            self._dd_zmeth
        ]
        for widget in self._zw1:
            widget.setVisible(False)

        # internal variables
        self._zep1 = self._d_vfrom.date().toString('yyyy-MM-dd')
        self._zep2 = self._d_vuntil.date().toString('yyyy-MM-dd')
        self._ores:float = None
        self._zoning:Zoning = None

    def init_tab(self):
        super().init_tab()
        self.store.websocket_server.register_handler(
            self._field_selected, Messages.GotSelectedField
        )

    def _reset_date_widgets(self):
        self._d_vfrom.setDate(QDate(QDate.currentDate().year(), 1, 1))
        self._d_vuntil.setDate(QDate.currentDate())

    def _reset_zmeth_selection(self):
        self._dd_zmeth.clear()
        self._dd_zmeth.addItem(_TEXT.DD_ZMETH_ONEZONE)
        self._dd_zmeth.addItem(_TEXT.DD_ZMETH_FROMVEC)
        self._dd_zmeth.addItem(_TEXT.DD_ZMETH_REGGRID)
        self._lbl_ores.setVisible(False)
        self._inp_ores.setVisible(False)
        self._btn_rndr.setVisible(False)
        self._btn_save.setVisible(False)

    def _field_selected(self, msg:Messages.GotSelectedField):
        if self._active_field is None:
            self._active_field = msg.field_name
            for widget in self._zw1:
                widget.setVisible(True)
        elif msg.field_name == self._active_field:
            return

        self._active_field = msg.field_name
        self._lbl_sfld.setText(msg.field_name)
        self._reset_date_widgets()
        self._reset_zmeth_selection()

    def _set_first_epoch(self):
        self._zep1 = self._d_vfrom.date().toString('yyyy-MM-dd')

    def _set_second_epoch(self):
        self._zep2 = self._d_vuntil.date().toString('yyyy-MM-dd')

    def _zmeth_selected(self):
        self._btn_rndr.setVisible(True)
        self._btn_save.setVisible(True)

        if self._dd_zmeth.currentText() == _TEXT.DD_ZMETH_ONEZONE:
            self._inp_ores.setVisible(False)
            self._lbl_ores.setVisible(False)
        elif self._dd_zmeth.currentText() == _TEXT.DD_ZMETH_FROMVEC:
            pass
        elif self._dd_zmeth.currentText() == _TEXT.DD_ZMETH_REGGRID:
            self._inp_ores.setVisible(True)
            self._lbl_ores.setVisible(True)

    def _get_field_gdf(self) -> GeoDataFrame:
        cond = self.store.project_data.fields[
            PrjDB.TBL_FIELDS.COL_FIELDNAME
        ] == self._active_field
        return self.store.project_data.fields[cond].to_crs(
            epsg=3857, inplace=False
        )

    def _render_zones(self):
        zn_info = Messages.SendZones()

        try:
            self._ores = float(self._inp_ores.text())
        except:
            # TODO
            pass

        if self._dd_zmeth.currentText() == _TEXT.DD_ZMETH_ONEZONE:
            self._zoning = Zoning.one_zone(self._get_field_gdf())
            zgdf = self._zoning.to_gdf()
            zn_info = Messages.SendZones()
            zn_info.zone_names = self._zoning.layer_ids
            zn_info.coordinates = [
                zgdf.geometry.get_coordinates().values.tolist()
            ]

        elif self._dd_zmeth.currentText() == _TEXT.DD_ZMETH_REGGRID:
            self._zoning = Zoning.regular_grid(
                self._get_field_gdf(), objres=self._ores
            )
            zgdf = self._zoning.to_gdf()
            zgdf = zgdf[zgdf[self._zoning.layer_ids[-1]] == 1.]
            inscols = []
            for col in zgdf.columns:
                if 'zone_' in col:
                    inscols.append(col)
            zn_info.zone_names = []
            zn_info.coordinates = []
            for tpl in zgdf.itertuples():
                for col in inscols:
                    if getattr(tpl, col):
                        zn_info.zone_names.append(col)
                        break
                zn_info.coordinates.append(
                    np.array(tpl.geometry.exterior.coords)[:-1, :].tolist()
                )
        elif self._dd_zmeth.currentText() == _TEXT.DD_ZMETH_FROMVEC:
            pass

        self.store.websocket_server.send_messages([zn_info])

    def _save_zones(self):
        pass
