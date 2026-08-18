from PyQt5.QtWidgets import (
    QWidget, QMenu, QTreeWidgetItem, QTreeWidget
)
from PyQt5.QtCore import Qt, QTimer
import datetime as dt
from importlib import import_module
from copy import deepcopy

from ....farming.tasks import (
    Task, Application, DBIntegration as TDBI, sowing, fertilization, harvest
)
from ..utils.widgets import ComboBox


_available_tasks = (
    sowing.Sowing, fertilization.MineralFertilization, harvest.Harvest
)


class _TEXT:
    MENU_ADD_YEAR = 'add year'
    MENU_SEP_TASK = '--- tasks ---'
    MENU_ADD_TASK = 'add {}-task'
    MENU_ADD_APPL = 'add {}'
    MENU_SEL_APPLMAP = 'select application map'
    APPL_UNIT_HINT = 'select unit'


class _TasksItem(QTreeWidgetItem):
    """
    Base class for tasks-tree items.

    constructor arguments

    * **data** (*tuple | list*) - iterable which contains a ``str`` for each column of the ``QTreeWidgetItem``

    """
    def __init__(self, data):
        super().__init__(data)
        self.setFlags(self.flags() | Qt.ItemFlag.ItemIsEditable)
        self._ecs:tuple[int] = ()

    @property
    def editable_cols(self) -> tuple[int]:
        """
        :return: column indices representing editable columns
        :rtype: tuple[int]
        """
        return self._ecs
    
    @editable_cols.setter
    def editable_cols(self, ecols):
        self._ecs = ecols


################################################################################
# TREE NODES/ITEMS
################################################################################
class TasksYear(_TasksItem):
    """
    Tasks-tree item which represents the years. 
    A right-click opens the :class:`YearMenu`.

    constructor arguments

    * **year** (*int | str, optional*) - year which should be visualized (if ``None``, ``TasksYear.HINT_YEAR`` will be used)

    """
    HINT_YEAR = '< YYYY >'

    def __init__(self, year:int|str=None):
        if year is None:
            year = self.HINT_YEAR
        super().__init__([str(year), '', '', ''])
        self.editable_cols = (0,)

    @property
    def year(self) -> int:
        """
        :return: year for the selected field (``None`` if not provided by user yet or if it is older than 1990 or if it is more than 10 years in the future)
        :rtype: int
        """
        if self.text(0) == self.HINT_YEAR:
            return None
        else:
            try:
                year = int(self.text(0))
            except:
                return None
            if (year > 1990) and (year < dt.date.today().year + 10):
                return year
            else:
                return None


class TasksTask(_TasksItem):
    """
    Tasks-tree item which represents a whole task.
    A right-click opens a child class of :class:`TaskMenu`.
    These child classes are located in ``mef_agri.app.gui.tasks.menus``.
    """
    HINT_DATE = '< YYYY-MM-DD >'

    def __init__(
            self, tree:QTreeWidget, task_name:str, task_module:str, 
            task_date:dt.date|str=None
        ):
        """
        :param tree: currently visible tasks-tree
        :type tree: QTreeWidget
        :param task_name: name of the task (second column of item - not editable), which has to be also equal to the name of a child-class of :class:`mef_agri.farming.tasks.Task`
        :type task_name: str
        :param task_module: module containing child class of :class:`mef_agri.farming.tasks.Task`
        :type task_module: str
        :param task_date: date when task has been started (first column of item - not editable but set when providing the begin-date of the task), defaults to None
        :type task_date: datetime.date | str, optional
        """
        self._fname:str = None
        self._tree:QTreeWidget = tree
        self._tree.itemChanged.connect(self._edit_date)

        if task_date is None:
            task_date = self.HINT_DATE
        elif isinstance(task_date, dt.date):
            task_date = task_date.isoformat()
        super().__init__([task_date, task_name, '', ''])
        self.editable_cols = (0,)

        self._tmodule:str = task_module
        self._task:Task = getattr(import_module(task_module), task_name)()
        self._new_task_date:str = None
        self._db:bool = False

    @property
    def field_name(self) -> str:
        """
        :return: name of the currently selected field
        :rtype: str
        """
        return self._fname
    
    @field_name.setter
    def field_name(self, fname):
        self._fname = fname

    @property
    def task_obj(self) -> Task:
        """
        :return: task object which is loaded from the given ``task_name`` and ``task_module`` in the constructor
        :rtype: Task
        """
        return self._task
    
    @property
    def task_module(self) -> str:
        """
        :return: module which contains the task
        :rtype: str
        """
        return self._tmodule
    
    @property
    def task_date(self) -> str:
        date = self.text(2)
        if date == self.HINT_DATE:
            return None
        else:
            return date
        
    @task_date.setter
    def task_date(self, date):
        try:
            dt.date.fromisoformat(date)
        except:
            return
        self._new_task_date = date
        QTimer.singleShot(100, self._update_task_date)

    @property
    def from_db(self) -> bool:
        """
        :return: flag, if task has been loaded from project-database
        :rtype: bool
        """
        return self._db
    
    @from_db.setter
    def from_db(self, val):
        self._db = val
    
    def _edit_date(self, item:TasksTask, column:int):
        if not isinstance(item, TasksTask):
            return
        if column != 0:
            return
        try:
            dt.date.fromisoformat(item.text(0))
        except:
            return
        for i in range(self.childCount()):
            child:TasksInfo = self.child(i)
            if child.text(1) == Task.date_begin.__name__:
                child.value = item.text(0)

    def _update_task_date(self):
        self.setText(0, self._new_task_date)

    def setup_task_obj(self) -> Task:
        """
        Setting up a copy of :func:`task_obj` from the content of the tasks-tree

        :return: set up task ready for saving
        :rtype: Task
        """
        task = deepcopy(self.task_obj)
        for i1 in range(self.childCount()):
            item = self.child(i1)
            if isinstance(item, TasksInfo):
                if item.value is None:
                    continue
                setattr(task, item.name, item.value)
            elif isinstance(item, TasksAppl):
                appl:Application = getattr(
                    import_module(self.task_module), item.name
                )()
                for i2 in range(item.childCount()):
                    vitem = item.child(i2)
                    setattr(getattr(appl, vitem.name), 'value', vitem.value())
                    if isinstance(vitem, TasksApplNumVal):
                        setattr( getattr(appl, vitem.name), 'unit', vitem.unit)
                task.add_application(appl)
        task.set_up_task()
        return task


class TasksInfo(_TasksItem):
    """
    Class which represents tasks-tree items specifying temporal information for 
    a task (i.e. begin- and end-date as well as begin- and end-time).
    """
    HINT_DATE = '< YYYY-MM-DD >'
    HINT_TIME = '< hh:mm >'
    
    def __init__(self, tree:QTreeWidget, info_name:str, info_value:str):
        """
        :param tree: currently visible tasks-tree
        :type tree: QTreeWidget
        :param info_name: name of the task information (second column - not editable)
        :type info_name: str
        :param info_value: value of the task information (third column - editable), i.e iso-formatted date and time strings
        :type info_value: str
        """
        self._tree = tree
        self._tree.itemChanged.connect(self._edit_value)
        super().__init__(['', info_name, info_value, ''])
        self.editable_cols = (2,)
        self._newval = None

    @property
    def name(self) -> str:
        """
        :return: name of the task-info attribute
        :rtype: str
        """
        return self.text(1)
    
    @property
    def value(self) -> str:
        """
        :return: value of the task-info attribute (``None`` if it is still the date- or time-hint)
        :rtype: str
        """
        val = self.text(2)
        if val in (self.HINT_DATE, self.HINT_TIME):
            return None
        else:
            return val
        
    @value.setter
    def value(self, val):
        try:
            dt.date.fromisoformat(val)
        except:
            return
        self._newval = val
        QTimer.singleShot(100, self._update_value)

    def _update_value(self):
        self.setText(2, self._newval)

    def _edit_value(self, item:TasksInfo, column:int):
        if not isinstance(item, TasksInfo):
            return
        if item.name != Task.date_begin.__name__:
            return
        if column != 2:
            return
        try:
            dt.date.fromisoformat(item.value)
        except:
            return
        task:TasksTask = self.parent()
        task.task_date = item.value

class TasksAppl(_TasksItem):
    """
    Class which represents the tasks-tree items containing the application names
    """
    def __init__(self, appl_name:str):
        """
        :param appl_name: name of the application being equal to the corresponding class in ``mef_agri.farming.tasks`` (second column - not editable)
        :type appl_name: str
        """
        super().__init__(['', appl_name, '', ''])

    @property
    def name(self) -> str:
        """
        :return: name of the application
        :rtype: str
        """
        return self.text(1)


class TasksApplNumVal(_TasksItem):
    """
    Class which represents the tasks-tree items of numeric application values
    """
    class Unit(object):
        def __init__(self, tree:QTreeWidget, item:TasksApplNumVal, col:int):
            self._t = tree
            self._i = item
            self._ic = col
            self._sel:ComboBox = ComboBox(_TEXT.APPL_UNIT_HINT)
            self._wset:bool = False

        def set_valid_units(self, vu):
            self._sel.addItems(vu)
            self._t.setItemWidget(self._i, self._ic, self._sel)
            self._wset = True

        def __call__(self, unit:str=None) -> None | str:
            """
            :param unit: unit which will be set accordingly if provided, defaults to None
            :type unit: str, optional
            :return: unit if ``unit`` is not provided as argument
            :rtype: None | str
            """
            if unit is None:
                u = self._sel.currentText()
                if u == _TEXT.APPL_UNIT_HINT:
                    return None
                else:
                    return u
            else:
                self._sel.setCurrentText(unit)
                if not self._wset:
                    self._t.setItemWidget(self._i, self._ic, self._sel)

    def __init__(self, tree:QWidget, vname:str, value:str|float=None):
        data = ['', vname, '', '']
        if value is not None:
            data[2] = str(value)
        super().__init__(data)
        self._u:TasksApplNumVal.Unit = self.Unit(tree, self, 3)
        self.editable_cols = (2,)

    @property
    def name(self) -> str:
        """
        :return: name of the numeric value
        :rtype: str
        """
        return self.text(1)

    @property
    def value(self) -> TasksApplNumVal.Value:
        """
        :return: value itself (number or path to application map)
        :rtype: str | float
        """
        return self.text(2)

    @property
    def unit(self) -> TasksApplNumVal.Unit:
        """
        :return: unit of the numeric value
        :rtype: str
        """
        return self._u


class TasksApplDescrVal(_TasksItem):
    """
    Class which represents the tasks-tree items of descriptive values
    """
    class Value(object):
        def __init__(self, tree:QTreeWidget, item:TasksApplNumVal, col:int):
            self._def:str = ''
            self._t:QTreeWidget = tree
            self._i:TasksApplNumVal = item
            self._ic:int = col
            self._w:QWidget = None
            self._wg, self._ws = None, None  # methods to get and set required text from widget within tasks-tree item
        
        @property
        def default(self) -> str:
            """
            :return: default value for the descriptive value within the item or provided :func:`widget`
            :rtype: str
            """
            return self._def
        
        @default.setter
        def default(self, defval):
            self._def = defval

        @property
        def widget(self) -> QWidget:
            """
            :return: widget which is or should be contained within :class:`TasksApplDescrVal`
            :rtype: QWidget
            """
            return self._w
        
        @widget.setter
        def widget(self, w):
            self._w = w
            self._t.setItemWidget(self._i, self._ic, w)
        
        @property
        def widget_getter(self):
            """
            :return: method to derive descriptive value from :func:`widget`
            :rtype: method
            """
            return self._wg
        
        @widget_getter.setter
        def widget_getter(self, wg):
            self._wg = wg
        
        @property
        def widget_setter(self):
            """
            :return: method to set descriptive value in :func:`widget`
            :rtype: method
            """
            return self._ws
        
        @widget_setter.setter
        def widget_setter(self, ws):
            self._ws = ws
        
        def __call__(self, value:str=None) -> None | str:
            """
            :param value: descriptive value which will be set accordingly if provided, defaults to None
            :type value: str, optional
            :return: descriptive value if ``value`` is not provided
            :rtype: None | str
            """
            if value is None:
                if self.widget is None:
                    v = self._i.text(self._ic)
                else:
                    v = self.widget_getter()
                if v == self.default:
                    return None
                else:
                    return v
            else:
                if self.widget is None:
                    self._i.setText(self._ic, str(value))
                else:
                    self.widget_setter(value)
        
    def __init__(self, tree:QTreeWidget, vname:str):
        """
        :param vname: name of the descriptive value
        :type vname: str
        """
        data = ['', vname, '', '']
        super().__init__(data)
        self._v:TasksApplDescrVal.Value = self.Value(tree, self, 2)

    @property
    def name(self) -> str:
        """
        :return: name of the descriptive value
        :rtype: str
        """
        return self.text(1)
    
    @property
    def value(self) -> TasksApplDescrVal.Value:
        """
        :return: object containing value
        :rtype: str
        """
        return self._v


################################################################################
# CONTEXT MENUS FOR TREE NODES/ITEMS
################################################################################
class YearMenu(QMenu):
    """
    Menu which appears when right-clicking a :class:`TasksYear` item.
    It contains a point for adding a new :class:`TasksYear` item as well as 
    tasks being available in ``mef_agri.app.gui.tasks._available_tasks``.
    """
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
            setattr(addt, '_task_name', task.__name__)
            setattr(addt, '_task_module', task.__module__)
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
            getattr(self.sender(), '_task_name'),
            getattr(self.sender(), '_task_module')
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
    def __init__(self, tree:QTreeWidget):
        """
        :param tree: tasks-tree currently visible in the app
        :type tree: QTreeWidget
        """
        super().__init__()
        self._tree:QTreeWidget = tree
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
        # TODO integrate application map selection through context menu on the corresponding numeric value
        appl_name = getattr(self.sender(), '_appl_name')
        appl_obj:Application = getattr(
            import_module(self.task_item.task_obj.task_module), appl_name
        )()
        appl_item = TasksAppl(appl_name)
        appl_item = self.handle_descriptive_values(appl_item, appl_obj)
        for nval in appl_obj.numeric_values:
            appl_val = TasksApplNumVal(self._tree, nval.name)
            appl_item.addChild(appl_val)
            appl_val.unit.set_valid_units(nval.valid_units)
        self.task_item.addChild(appl_item)

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
