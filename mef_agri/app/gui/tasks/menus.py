from PyQt5.QtWidgets import QTreeWidget
from PyQt5.QtCore import QTimer
from pandas import DataFrame

from . import TasksAppl, TaskMenu, TasksApplDescrVal
from ..utils.widgets import ComboBox
from ....farming.tasks.sowing import SowingApplication
from ....farming.crops import DBIntegration as CDBI


class _TEXT:
    SOW_CULT_HINT = 'select cultivar'
    SOW_CROP_HINT = 'select crop'

class SowingMenu(TaskMenu):
    """
    Context menu containing applications which can be added to a sowing task.
    """
    def __init__(self, tree:QTreeWidget, tasks_tab):
        """
        :param tree: tasks-tree currently visible in the app
        :type tree: QTreeWidget
        :param tasks_tab: tasks-tab of the GUI
        :type tasks_tab: TasksTab
        """
        super().__init__(tree, tasks_tab)
        self._cults:DataFrame = None
        self._crop_item:TasksApplDescrVal = None
        self._cult_item:TasksApplDescrVal = None
        self._pars_item:TasksApplDescrVal = None
        self._crop_sel:ComboBox = None
        self._cult_sel:ComboBox = None

    @property
    def available_cultivars(self) -> DataFrame:
        """
        :return: cultivar information from project database
        :rtype: pandas.DataFrame
        """
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
        """
        Mapping the descriptive values of a :class:`SowingApplication` being
        the crop and cultivar names as well as crop-parameters to tasks-tree 
        nodes/items which are added to ``appl_item`` as childs

        :param appl_item: tasks-tree item representing the application
        :type appl_item: TasksAppl
        :param appl_obj: instance of :class:`SowingApplication`
        :type appl_obj: SowingApplication
        :return: ``appl_item`` with additional children being the descriptive values
        :rtype: TasksAppl
        """
        self._crop_item = TasksApplDescrVal(
            self._tree, appl_obj.crop.name, appl_obj.__class__.crop.__name__
        )
        self._crop_item.editable_cols = ()
        self._crop_item.value.default = _TEXT.SOW_CROP_HINT
        self._cult_item = TasksApplDescrVal(
            self._tree, appl_obj.cultivar.name, 
            appl_obj.__class__.cultivar.__name__
        )
        self._cult_item.editable_cols = ()
        self._cult_item.value.default = _TEXT.SOW_CULT_HINT
        self._pars_item = TasksApplDescrVal(
            self._tree, appl_obj.parameters.name, 
            appl_obj.__class__.parameters.__name__
        )
        self._pars_item.editable_cols = (2,)
        appl_item.addChildren(
            [self._crop_item, self._cult_item, self._pars_item]
        )
        self._crop_item.value.widget = self._crop_sel
        self._crop_item.value.widget_getter = self._crop_sel.currentText
        self._crop_item.value.widget_setter = self._crop_sel.setCurrentText
        self._cult_item.value.widget = self._cult_sel
        self._cult_item.value.widget_getter = self._cult_sel.currentText
        self._cult_item.value.widget_setter = self._cult_sel.setCurrentText
        return appl_item

    def _crop_selected(self, crop):
        if not crop:
            return
        if not(self._crop_item.value()):
            self._crop_item.setText(2, crop)
        elif crop == self._crop_item.text(2):
            return

        self._crop_item.value(value=crop)
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
        if not(self._cult_item.value()):
            self._cult_item.setText(2, cultivar)
        elif cultivar == self._cult_item.text(2):
            return

        self._cult_item.value(value=cultivar)
        self._cparams = self._cults[
            (self._cults[CDBI.COL_CROP] == self._crop_item.value()) & 
            (self._cults[CDBI.COL_CULTIVAR] == self._cult_item.value())
        ][CDBI.COL_PARAMS].values[0]
        QTimer.singleShot(100, self._update_params)

    def _update_params(self):
        self._pars_item.value(value=self._cparams)

class HarvestMenu(TaskMenu):
    def __init__(self, tree, tasks_tab):
        super().__init__(tree, tasks_tab)

class MinFertMenu(TaskMenu):
    def __init__(self, tree, tasks_tab):
        super().__init__(tree, tasks_tab)
