from PyQt5.QtWidgets import (
    QMenu, QTreeWidgetItem, QTreeWidget
)
from PyQt5.QtCore import Qt
import datetime as dt
from importlib import import_module

from ....farming.tasks import Task, Application, sowing, fertilization, harvest
from ..utils.widgets import ComboBox


_available_tasks = (
    sowing.Sowing, fertilization.MineralFertilization, harvest.Harvest
)


class _TEXT:
    MENU_ADD_YEAR = 'add year'
    MENU_SEP_TASK = '--- tasks ---'
    MENU_ADD_TASK = 'add {}-task'
    MENU_ADD_APPL = 'add {}'
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
            self, task_name:str, task_module:str, task_date:dt.date|str=None
        ):
        """
        :param task_name: name of the task (second column of item - not editable), which has to be also equal to the name of a child-class of :class:`mef_agri.farming.tasks.Task`
        :type task_name: str
        :param task_module: module containing child class of :class:`mef_agri.farming.tasks.Task`
        :type task_module: str
        :param task_date: date when task has been started (first column of item - not editable but set when providing the begin-date of the task), defaults to None
        :type task_date: datetime.date | str, optional
        """
        self._fname:str = None

        if task_date is None:
            task_date = self.HINT_DATE
        elif isinstance(task_date, dt.date):
            task_date = task_date.isoformat()
        super().__init__([task_date, task_name, '', ''])

        self._tmodule:str = task_module
        self._task:Task = getattr(import_module(task_module), task_name)()

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

    def setup_task(self) -> bool:
        """
        TODO
        """
        for ix1 in range(self.childCount()):
            taskcont = self.child(ix1)
            if isinstance(taskcont, TasksInfo):
                if taskcont.value is None:
                    continue
                try:
                    setattr(self.task_obj, taskcont.name, taskcont.value)
                except:
                    return False
            elif isinstance(taskcont, TasksAppl):
                appl:Application = getattr(
                    import_module(self._tmodule), taskcont.name
                )()
                for ix2 in range(taskcont.childCount()):
                    ainfo:TasksApplInfo = taskcont.child(ix2)
                    aval = getattr(appl, ainfo.name)
                    if isinstance(aval, Application.NumericValue):
                        # TODO consider path to application map as `ainfo.value`
                        try:
                            getattr(appl, ainfo.name).value = float(ainfo.value)
                        except:
                            return False
                        getattr(appl, ainfo.name).unit = ainfo.info
                    elif isinstance(aval, Application.DescriptiveValue):
                        getattr(appl, ainfo.name).value = ainfo.value
                self.task_obj.add_application(appl)

    def save_task(self):
        # TODO
        pass
    

class TasksInfo(_TasksItem):
    """
    Class which represents tasks-tree items specifying temporal information for 
    a task (i.e. begin- and end-date as well as begin- and end-time).
    """
    HINT_DATE = '< YYYY-MM-DD >'
    HINT_TIME = '< hh:mm >'
    
    def __init__(self, info_name:str, info_value:str):
        """
        :param info_name: name of the task information (second column - not editable)
        :type info_name: str
        :param info_value: value of the task information (third column - editable), i.e iso-formatted date and time strings
        :type info_value: str
        """
        super().__init__(['', info_name, info_value, ''])
        self.editable_cols = (2,)

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
    def __init__(self, vname:str, value=None, vunit:str=None):
        data = ['', vname, '', '']
        super().__init__(data)

        self._val = value
        self._vu = vunit

    @property
    def name(self) -> str:
        return self.text(1)

    @property
    def value(self):
        pass


class TasksApplDescrVal(_TasksItem):
    def __init__(self, vname:str, value:str=None):
        data = ['', vname, '', '']
        super().__init__(data)


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
            getattr(self.sender(), '_task_name'),
            getattr(self.sender(), '_task_module')
        )
        task.addChildren([
            TasksInfo(Task.date_begin.__name__, TasksInfo.HINT_DATE),
            TasksInfo(Task.time_begin.__name__, TasksInfo.HINT_TIME),
            TasksInfo(Task.date_end.__name__, TasksInfo.HINT_DATE),
            TasksInfo(Task.time_end.__name__, TasksInfo.HINT_TIME),
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
            appl_val = TasksApplInfo(nval.name)
            appl_item.addChild(appl_val)
            usel = ComboBox(_TEXT.APPL_UNIT_HINT)
            usel.addItems(nval.valid_units)
            usel.currentTextChanged.connect(
                lambda unit: self._unit_selected(unit)
            )
            self._tree.setItemWidget(appl_val, 3, usel)
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
            appl_item.addChild(TasksApplInfo(dval.name))
        return appl_item

    def _unit_selected(self, unit:str):
        print(unit)
