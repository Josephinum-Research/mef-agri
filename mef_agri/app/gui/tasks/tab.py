from PyQt5.QtWidgets import (
    QVBoxLayout, QLabel, QTreeWidget, QTreeWidgetItem, QPushButton
)
from PyQt5.QtCore import (
    Qt, QPoint, QTimer
)
from PyQt5.QtGui import (
    QCursor
)
import datetime as dt

from . import YearMenu, NumValMenu, TasksYear, TasksTask, TasksApplNumVal
from .menus import SowingMenu, HarvestMenu, MinFertMenu
from ..utils.widgets import NonProjectTab
from ..conn.msgs import Messages
from ....farming.tasks import (
    Task, ProjectTasksExtension, DBIntegration as DBTasks
)
from ....farming.tasks import sowing, harvest, fertilization
from ....farming import crops, fertilizers


class _TEXT:
    DB_TASK_TABLE = 'tasks'
    DB_CULT_TABLE = 'cultivars'
    DB_FERT_TABLE = 'fertilizers'
    LBL_SELFLD_INIT = 'select field in map'
    BTN_SAVE_DISABLED = 'nothing to save'
    BTN_SAVE_ENABLED = 'save changes'


class _STYLE:
    BTN_SAVE_ENABLED = """
        QPushButton {
            background-color: rgb(255, 127, 80)
        }
    """
    BTN_SAVE_DISABLED = """
        QPushButton {
            background-color: rgb(0, 255, 150)
        }
    """


class _ErrorDialogs:
    @staticmethod
    def task2db_error(task:str, exc:str|Exception):
        from ...gui import _CustomErrorDialog
        msg = f'Error when writing task `{task}` to project-database: {exc}'
        dlg = _CustomErrorDialog(msg)
        dlg.exec()

    @staticmethod
    def task2folder_error(task:str, exc:str|Exception):
        from ...gui import _CustomErrorDialog
        msg = f'Error when writing task `{task}` to folder/file: {exc}'
        dlg = _CustomErrorDialog(msg)
        dlg.exec()

    @staticmethod
    def task_setup_no_application(task:str):
        from ...gui import _CustomErrorDialog
        msg = f'Error when setting up task `{task}` -> there is no application '
        msg += 'added to this task!'
        dlg = _CustomErrorDialog(msg)
        dlg.exec()

    @staticmethod
    def task_setup_error(task:str, exc:str|Exception):
        from ...gui import _CustomErrorDialog
        msg = f'Error when setting up task `{task}`:\n{exc}'
        dlg = _CustomErrorDialog(msg)
        dlg.exec()


class _WarningDialogs:
    @staticmethod
    def unsaved_changes():
        from ...gui import _CustomWarningDialog
        msg = 'There are unsaved changes in the tasks-tree!'
        dlg = _CustomWarningDialog(msg)
        dlg.show()


class TasksTab(NonProjectTab):
    """
    Tab which provides access to the tasks related to the selected field.
    Tasks are organized in a tree with years being the top-level items.
    """
    _VAR_TTREE_CHANGE = '__mef__task_tree_change'

    def __init__(self, parent, store):
        super().__init__(parent, store)
        self._active_field:str = None
        self._colcont = None
        # utilities for the integration of tasks into project-db
        self._tdbi = DBTasks(_TEXT.DB_TASK_TABLE)
        self._cdbi = crops.DBIntegration(_TEXT.DB_CULT_TABLE)
        self._fdbi = fertilizers.DBIntegration(_TEXT.DB_FERT_TABLE)

        # setting up the layout
        self.layout_main = QVBoxLayout()

        # widgets
        # field-name as label
        self._lbl_sfld = QLabel(_TEXT.LBL_SELFLD_INIT)
        self.layout_main.addWidget(self._lbl_sfld)

        # tree containing tasks for selected field
        self._tree = QTreeWidget()
        self._tree.headerItem().setText(0, 'date')
        self._tree.headerItem().setText(1, 'task')
        self._tree.headerItem().setText(2, 'input')
        self._tree.headerItem().setText(3, 'unit')
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._add_item)
        self._tree.itemChanged.connect(self._item_changed)
        self._tree.itemDoubleClicked.connect(self._edit_item)
        self.layout_main.addWidget(self._tree)

        # buttons for user interaction
        self._btn_save = QPushButton(_TEXT.BTN_SAVE_DISABLED)
        self._btn_save.clicked.connect(self._save_changes)
        self.layout_main.addWidget(self._btn_save)

        self._toggle_unsaved_changes(False)

    def init_tab(self):
        """
        Initialization of the tab. The following steps are performed

        * Registering a handler (:func:`_field_selected`) to get information on selected field from the websocket server
        * Creating tables for tasks, crops and fertilizers in the project-db (if not already present)

        """
        super().init_tab()

        # websocket stuff
        self.store.websocket_server.register_handler(
            self._field_selected, Messages.GotSelectedField
        )
        # TODO properly deregister this handler

        # prepare project-db for task integration
        ttable = self.store.project_data.query(self._tdbi.sql_table_exists)
        if len(ttable) == 0:
            self.store.project_data.execute(self._tdbi.sql_create)
            msg = '(tasks/tab.py) TasksTab.init_tab => created tasks table '
            msg += 'in project-data-db.'
            print(msg)
        ctable = self.store.project_data.query(self._cdbi.sql_table_exists)
        if len(ctable) == 0:
            self.store.project_data.execute(self._cdbi.sql_create)
            self.store.project_data.execute(self._cdbi.sql_insert_defaults)
            msg = '(tasks/tab.py) TasksTab.init_tab => created cultivars table '
            msg += 'in project-data-db.'
            print(msg)
        ftable = self.store.project_data.query(self._fdbi.sql_table_exists)
        if len(ftable) == 0:
            self.store.project_data.execute(self._fdbi.sql_create)
            self.store.project_data.execute(self._fdbi.sql_insert_defaults)
            msg = '(tasks/tab.py) TasksTab.init_tab => created fertilizers '
            msg += 'table in project-data-db.'
            print(msg)

    def _field_selected(self, msg:Messages.GotSelectedField):
        """
        Handler for the websocket-server which is called when a field is 
        selected in the map.
        If a task-tree is present, it will be cleared and the new task-tree will 
        be visualized and task data will be loaded from the project-db if 
        available.

        :param msg: message provided by the wegsocket-server when the user selects a field in the map
        :type msg: Messages.GotSelectedField
        """
        if self._active_field is None:
            self._active_field = msg.field_name
        elif msg.field_name == self._active_field:
            return
        if getattr(self._tree, self._VAR_TTREE_CHANGE):
            return

        self._toggle_unsaved_changes(False)        
        self._tree.clear()
        self._active_field = msg.field_name
        self._lbl_sfld.setText(msg.field_name)

        prj_tasks = ProjectTasksExtension(
            self.store.project_data, _TEXT.DB_TASK_TABLE
        )
        tasks = prj_tasks.get_tasks(fields=msg.field_name)
        if (
            len(tasks[msg.field_name]) == 0 and 
            self._tree.topLevelItemCount() == 0
        ):
            self._tree.addTopLevelItem(TasksYear(dt.date.today().year))
        else:
            ####################################################################
            # TODO create tree from db entries and tasks in data directory
            ####################################################################
            print(tasks)

    def _add_item(self, position:QPoint):
        """
        Method which is called when user makes a right-click in the tasks-tree. 
        Depending on the clicked item, appropriate context-menus (QMenu) will be 
        created and visualized.

        New context menus representing other tasks than sowing, fertilization 
        and harvest have to be implemented here.

        :param position: widget/tree-item coordinates which has been right-clicked
        :type position: QPoint
        """
        item = self._tree.itemAt(position)
        if not isinstance(item, (TasksYear, TasksTask, TasksApplNumVal)):
            return
        
        if isinstance(item, TasksYear):
            ymenu = YearMenu(self._tree)
            ymenu.year_item = item
            ymenu.exec_(QCursor.pos())
        elif isinstance(item, TasksTask):
            tname = item.task_obj.__class__.__name__

################################################################################
# NOTE #########################################################################
################################################################################
            # add further tasks here if necessary
            if tname == sowing.Sowing.__name__:
                tmenu = SowingMenu(self._tree, self)
                tmenu.available_cultivars = self.store.project_data.query(
                    self._cdbi.sql_query_all
                )
            elif tname == harvest.Harvest.__name__:
                tmenu = HarvestMenu(self._tree, self)
            elif tname == fertilization.MineralFertilization.__name__:
                tmenu = MinFertMenu(self._tree, self)
################################################################################
# NOTE #########################################################################
################################################################################
            
            tmenu.task_item = item
            tmenu.exec_(QCursor.pos())
        elif isinstance(item, TasksApplNumVal):
            vmenu = NumValMenu()
            vmenu.numval_item = item
            vmenu.exec_(QCursor.pos())

    def _item_changed(self, item:QTreeWidgetItem, column:int):
        """
        Method which is called when text of a tree-item has been changed.
        Only has an effect if the column of the corresponding item is editable (
        see :func:`mef_agri.app.gui.tasks._TasksItem.editable_cols`)

        :param item: item which text has been changed by the user
        :type item: QTreeWidgetItem
        :param column: column in the item
        :type column: int
        """
        if not column in item.editable_cols:
            item.setText(column, self._colcont)

    def _edit_item(self, item, column):
        """
        Method which is called when an item is double-clicked, i.e. when it is 
        intended to add/change the correspoinding text.
        Herein, the present text (before editing) will be stored and again 
        inserted if the user aims to change an item-column which is not editable 
        (see :func:`mef_agri.app.gui.tasks._TasksItem.editable_cols`).

        :param item: item which has been double-clicked
        :type item: QTreeWidgetItem
        :param column: column in the item
        :type column: int
        """
        self._colcont = item.text(column)

    def _save_changes(self):
        """
        Method which is called when clicking the save-button. 
        Information from the tasks-tree is saved to the project-databse and to 
        the project's data directory.
        """
        prj_tasks = ProjectTasksExtension(
            self.store.project_data, _TEXT.DB_TASK_TABLE
        )
        for i1 in range(self._tree.topLevelItemCount()):
            yitem:TasksYear = self._tree.topLevelItem(i1)
            if yitem.year == None:
                continue
            for i2 in range(yitem.childCount()):
                titem:TasksTask = yitem.child(i2)
                titem.field_name = self._active_field
                if titem.from_db:
                    continue
                try:
                    task:Task = titem.setup_task_obj()
                    if task is None:
                        _ErrorDialogs.task_setup_no_application(titem.task_name)
                        return
                except Exception as exc:
                    _ErrorDialogs.task_setup_error(titem.task_name, exc)
                    return

                try:
                    self.store.project_data.execute(
                        self._tdbi.sql_insert(task, titem.field_name)
                    )
                except Exception as exc:
                    _ErrorDialogs.task2db_error(task.__class__.__name__, exc)
                    return

                try:
                    prj_tasks.save_task(titem.field_name, task)
                except Exception as exc:
                    self.store.project_data.execute(
                        self._tdbi.sql_delete(task, titem.field_name)
                    )
                    _ErrorDialogs.task2folder_error(
                        task.__class__.__name__, exc
                    )
                    return
                
        self._toggle_unsaved_changes(False)

    def _toggle_unsaved_changes(self, flag:bool):
        self._btn_save.setEnabled(flag)
        if flag:
            self._btn_save.setStyleSheet(_STYLE.BTN_SAVE_ENABLED)
            self._btn_save.setText(_TEXT.BTN_SAVE_ENABLED)
        else:
            self._btn_save.setStyleSheet(_STYLE.BTN_SAVE_DISABLED)
            self._btn_save.setText(_TEXT.BTN_SAVE_DISABLED)

        setattr(self._tree, self._VAR_TTREE_CHANGE, flag)
        msg = Messages.SendTasksTreeChanges()
        msg.unsaved_changes = flag
        self.store.websocket_server.send_messages(msg)