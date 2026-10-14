from PyQt5.QtWidgets import QTreeWidget, QMenu
from PyQt5.QtCore import QTimer
from pandas import DataFrame
from importlib import import_module

from .tree_items import (
    TasksYear, TasksTask, TasksInfo, TasksAppl, TasksApplNumVal, 
    TasksApplDescrVal
)
from ..utils.widgets import ComboBox
from ....farming.tasks import Task, Application
from ....farming.tasks import sowing, fertilization, harvest
from ....farming.crops import DBIntegration as CDBI


################################################################################
# NOTE #########################################################################
################################################################################
_available_tasks = (
    sowing.Sowing, fertilization.MineralFertilization, harvest.Harvest
)
################################################################################
# NOTE #########################################################################
################################################################################

class _TEXT:
    MENU_ADD_YEAR = 'add year'
    MENU_SEP_TASK = '--- tasks ---'
    MENU_ADD_TASK = 'add {}-task'
    MENU_ADD_APPL = 'add {}'
    MENU_SEL_APPLMAP = 'select application map'
    SOW_CULT_HINT = 'select cultivar'
    SOW_CROP_HINT = 'select crop'
    

class YearMenu(QMenu):
    """
    Menu which appears when right-clicking a :class:`TasksYear` item.
    It contains a point for adding a new :class:`TasksYear` item as well as 
    tasks being available in ``mef_agri.app.gui.tasks._available_tasks``.
    """
    _VAR_TNAME = '__mef__task_name'
    _VAR_TMODULE = '__mef__task_module'

    def __init__(self, tree:QTreeWidget):
        """
        :param tree: tasks-tree currently visible in the app
        :type tree: QTreeWidget
        """
        super().__init__()
        self._tree:QTreeWidget = tree
        self._iy:TasksYear = None
        self._fname:str = None

        # add actions
        addy = self.addAction(_TEXT.MENU_ADD_YEAR)
        addy.triggered.connect(self._add_year)
        sep = self.addAction(_TEXT.MENU_SEP_TASK)
        sep.setEnabled(False)

        for task in _available_tasks:
            addt = self.addAction(_TEXT.MENU_ADD_TASK.format(task.__name__))
            setattr(addt, self._VAR_TNAME, task.__name__)
            setattr(addt, self._VAR_TMODULE, task.__module__)
            addt.triggered.connect(self._add_task)

    @property
    def year_item(self) -> TasksYear:
        """
        :return: currently selected `TasksYear`-item
        :rtype: TasksYear
        """
        return self._iy
    
    @year_item.setter
    def year_item(self, item):
        self._iy = item

    @property
    def active_field(self) -> str:
        """
        :return: name of currently selected field
        :rtype: str
        """
        return self._fname
    
    @active_field.setter
    def active_field(self, fname):
        self._fname = fname

    def _add_year(self):
        """
        Method which is called when user chooses to add a new year in the 
        context menu.
        A new :class:`TasksYear` item will be added to the tree.
        """
        self._tree.addTopLevelItem(TasksYear())

    def _add_task(self):
        """
        Method which is called when the user chooses to add a new 
        :class:`TasksTask` item to the currently active field and year.
        Additionally four :class:`TasksInfo` items will be appended to the new 
        task for begin- and end-date as well as begin- and end-time.
        """
        task = TasksTask(
            self._tree,
            getattr(self.sender(), self._VAR_TNAME),
            getattr(self.sender(), self._VAR_TMODULE)
        )
        task.addChildren([
            TasksInfo(self._tree, Task.date_begin.__name__, TasksInfo.HINT_DATE),
            TasksInfo(self._tree, Task.time_begin.__name__, TasksInfo.HINT_TIME),
            TasksInfo(self._tree, Task.date_end.__name__, TasksInfo.HINT_DATE),
            TasksInfo(self._tree, Task.time_end.__name__, TasksInfo.HINT_TIME),
        ])
        self.year_item.addChild(task)


class TaskMenu(QMenu):
    """
    Context menu which appears when right-clicking on a :class:`TasksTask` item. 
    It contains the applications which can be added to a task (see 
    :func:`mef_agri.farming.tasks.Task.valid_applications`).
    """
    def __init__(self, tree:QTreeWidget, tasks_tab):
        """
        :param tree: tasks-tree currently visible in the app
        :type tree: QTreeWidget
        :param tasks_tab: tasks-tab of the GUI
        :type tasks_tab: TasksTab
        """
        super().__init__()
        from .tab import TasksTab
        self._tree:QTreeWidget = tree
        self._ttab:TasksTab = tasks_tab
        self._it:TasksTask = None

    @property
    def task_item(self) -> TasksTask:
        """
        :return: currently selected `TasksTask`-item
        :rtype: TasksTask
        """
        return self._it
    
    @task_item.setter
    def task_item(self, item):
        self._it = item
        for appl in self._it.task_obj.valid_applications:
            adda = self.addAction(_TEXT.MENU_ADD_APPL.format(appl.__name__))
            setattr(adda, '_appl_name', appl.__name__)
            adda.triggered.connect(self._add_application)

    def _add_application(self):
        appl_name = getattr(self.sender(), '_appl_name')
        appl_obj:Application = getattr(
            import_module(self.task_item.task_obj.task_module), appl_name
        )()
        appl_item = TasksAppl(appl_name)
        appl_item = self.handle_descriptive_values(appl_item, appl_obj)
        for nval in appl_obj.numeric_values:
            appl_val = TasksApplNumVal(self._tree, nval.name, nval.valid_units)
            appl_item.addChild(appl_val)
        self.task_item.addChild(appl_item)
        self._ttab._toggle_unsaved_changes(True)

    def handle_descriptive_values(
            self, appl_item:TasksAppl, appl_obj:Application
        ) -> TasksAppl:
        """
        Method to control behavior/appearance of descriptive values in the 
        tasks-tree.
        If not overridden in a child-class, every descriptive value of an 
        Application will be added to the tasks-tree as :class:`TasksApplInfo` 
        where the user has to manually enter the value.

        :param appl_item: application item of the tasks-tree
        :type appl_item: TasksAppl
        :param appl_obj: object representing the current application
        :type appl_obj: Application
        :return: updated ``appl_item``
        :rtype: TasksAppl
        """
        for dval in appl_obj.descriptive_values:
            appl_item.addChild(TasksApplDescrVal(dval.name))
        return appl_item


class NumValMenu(QMenu):
    def __init__(self):
        super().__init__()
        self._nvit:TasksApplNumVal
        selmap = self.addAction(_TEXT.MENU_SEL_APPLMAP)
        selmap.triggered.connect(self._select_applmap)

    @property
    def numval_item(self) -> TasksApplNumVal:
        return self._nvit
    
    @numval_item.setter
    def numval_item(self, item):
        self._nvit = item

    def _select_applmap(self):
        pass


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
            self, appl_item:TasksAppl, appl_obj:sowing.SowingApplication
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
        self._crop_item = TasksApplDescrVal(self._tree, appl_obj.crop.name)
        self._crop_item.editable_cols = ()
        self._crop_item.value.default = _TEXT.SOW_CROP_HINT
        self._cult_item = TasksApplDescrVal(self._tree, appl_obj.cultivar.name)
        self._cult_item.editable_cols = ()
        self._cult_item.value.default = _TEXT.SOW_CULT_HINT
        self._pars_item = TasksApplDescrVal(
            self._tree, appl_obj.parameters.name, 
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
