from PyQt5.QtWidgets import (
    QVBoxLayout, QLabel, QTreeWidget, QPushButton
)
from PyQt5.QtCore import (
    Qt
)
from PyQt5.QtGui import (
    QCursor
)
import datetime as dt

from . import YearMenu, TaskMenu, TasksYear, TasksTask
from .menus import SowingMenu, HarvestMenu, MinFertMenu
from ..utils.widgets import NonProjectTab
from ..conn.msgs import Messages
from ....farming.tasks import DBIntegration as DBTasks
from ....farming.tasks import sowing, harvest, fertilization
from ....farming import crops, fertilizers


class _TEXT:
    DB_TASK_TABLE = 'tasks'
    DB_CULT_TABLE = 'cultivars'
    DB_FERT_TABLE = 'fertilizers'
    LBL_SELFLD_INIT = 'select field in map'
    BTN_SAVE = 'save'


class TasksTab(NonProjectTab):
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
        self._btn_save = QPushButton(_TEXT.BTN_SAVE)
        self._btn_save.clicked.connect(self._save_changes)
        self.layout_main.addWidget(self._btn_save)

    def init_tab(self):
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
        if self._active_field is None:
            self._active_field = msg.field_name
        elif msg.field_name == self._active_field:
            return
        
        self._tree.clear()
        self._lbl_sfld.setText(msg.field_name)
        tasks = self.store.project_data.query(
            self._tdbi.sql_query(msg.field_name)
        )
        if len(tasks) == 0 and self._tree.topLevelItemCount() == 0:
            self._tree.addTopLevelItem(TasksYear(dt.date.today().year))
        else:
            pass

    def _add_item(self, position):
        item = self._tree.itemAt(position)
        if not isinstance(item, (TasksYear, TasksTask)):
            return
        
        if isinstance(item, TasksYear):
            ymenu = YearMenu(self._tree)
            ymenu.year_item = item
            ymenu.exec_(QCursor.pos())
        elif isinstance(item, TasksTask):
            tname = item.task_obj.__class__.__name__

            # NOTE #############################################################
            # add further tasks here if necessary
            if tname == sowing.Sowing.__name__:
                tmenu = SowingMenu(self._tree)
                tmenu.available_cultivars = self.store.project_data.query(
                    self._cdbi.sql_query_all
                )
            elif tname == harvest.Harvest.__name__:
                tmenu = HarvestMenu(self._tree)
            elif tname == fertilization.MineralFertilization.__name__:
                tmenu = MinFertMenu(self._tree)
            # NOTE #############################################################
            
            tmenu.task_item = item
            tmenu.exec_(QCursor.pos())

    def _item_changed(self, item, column):
        if not column in item.editable_cols:
            item.setText(column, self._colcont)

    def _edit_item(self, item, column):
        self._colcont = item.text(column)

    def _save_changes(self):
        for ix1 in range(self._tree.topLevelItemCount()):
            item:TasksYear = self._tree.topLevelItem(ix1)
            if item.year == None:
                continue
            for ix2 in range(item.childCount()):
                task:TasksTask = item.child(ix2)
                tset = task.setup_task()
                if tset:
                    print('success')
                else:
                    print('missing information')
                # TODO save task in db and in data-directory if created by user
