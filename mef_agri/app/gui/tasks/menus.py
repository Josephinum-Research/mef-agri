from PyQt5.QtWidgets import QTreeWidget, QComboBox
from pandas import DataFrame

from . import TasksTask, TasksAppl, TasksApplInfo, TaskMenu
from ..utils.widgets import ComboBox
from ....farming.tasks.sowing import SowingApplication
from ....farming.crops import DBIntegration as CDBI


class _TEXT:
    SOW_CULT_HINT = 'select cultivar'
    SOW_CROP_HINT = 'select crop'

class SowingMenu(TaskMenu):
    def __init__(self, tree):
        super().__init__(tree)
        self._cults:DataFrame = None
        self._crop_item:TasksApplInfo = None
        self._cult_item:TasksApplInfo = None
        self._pars_item:TasksApplInfo = None
        self._crop_sel:ComboBox = None
        self._cult_sel:ComboBox = None

    @property
    def available_cultivars(self) -> DataFrame:
        return self._cults
    
    @available_cultivars.setter
    def available_cultivars(self, cults):
        self._cults = cults
        if self._crop_sel is not None:
            self._crop_sel.clear()

        self._crop_sel = ComboBox(_TEXT.SOW_CROP_HINT)
        self._crop_sel.addItems(self._cults[CDBI.COL_CROP].unique())
        self._crop_sel.currentTextChanged.connect(
            lambda crop: self._crop_selected(crop)
        )
        self._cult_sel = ComboBox(_TEXT.SOW_CULT_HINT)

    def handle_descriptive_values(
            self, appl_item:TasksAppl, appl_obj:SowingApplication
        ):
        self._crop_item = TasksApplInfo(appl_obj.crop.name)
        self._crop_item.editable_cols = ()
        self._cult_item = TasksApplInfo(appl_obj.cultivar.name)
        self._cult_item.editable_cols = ()
        self._pars_item = TasksApplInfo(appl_obj.parameters.name)
        self._pars_item.editable_cols = ()
        appl_item.addChildren(
            [self._crop_item, self._cult_item, self._pars_item]
        )
        # NOTE important!!! => adding widgets in tree must be done after
        # NOTE important!!! => adding the items to its parents in the tree
        self._tree.setItemWidget(self._crop_item, 2, self._crop_sel)
        self._tree.setItemWidget(self._cult_item, 2, self._cult_sel)
        return appl_item

    def _crop_selected(self, crop):
        if not crop:
            return
        if not(self._crop_item.text(2)):
            self._crop_item.setText(2, crop)
        elif crop == self._crop_item.text(2):
            return

        self._crop_item.setText(2, crop)
        self._cult_item.setText(2, '')
        if self._cult_sel is not None:
            self._cult_sel.clear()
        self._cult_sel.addItems(
            self.available_cultivars[
                self.available_cultivars[CDBI.COL_CROP] == crop
            ][CDBI.COL_CULTIVAR].tolist()
        )
        self._cult_sel.currentTextChanged.connect(
            lambda cultivar: self._cult_selected(cultivar)
        )

    def _cult_selected(self, cultivar):
        if not cultivar:
            return
        if not(self._cult_item.text(2)):
            self._cult_item.setText(2, cultivar)
        elif cultivar == self._selcult:
            return

        self._cult_item.setText(2, cultivar)
        print(self._crop_item.text(2))
        print(self._cult_item.text(2))
        cparams = self._cults[
            (self._cults[CDBI.COL_CROP] == self._crop_item.text(2)) & 
            (self._cults[CDBI.COL_CULTIVAR] == self._cult_item.text(2))
        ][CDBI.COL_PARAMS].values[0]
        self._pars_item.setText(
            2, self._crop_item.value + ' - ' + self._cult_item.value
        )


class HarvestMenu(TaskMenu):
    def __init__(self, tree):
        super().__init__(tree)

class MinFertMenu(TaskMenu):
    def __init__(self, tree):
        super().__init__(tree)
