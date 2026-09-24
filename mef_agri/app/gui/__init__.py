from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QTabWidget, QMessageBox
)

from .project.tab import ProjectTab
from .data.tab import DataTab
from .tasks.tab import TasksTab
from .conn.server import Messages
from .map import MapView


def print_log_msgs(msg:Messages.GotLogMsg):
    print(msg.log_message)


class _CustomErrorDialog(QMessageBox):
    def __init__(self, msg):
        super().__init__()
        self.setWindowTitle('error')
        self.setText(msg)
        self.setIcon(QMessageBox.Critical)


class _CustomWarningDialog(QMessageBox):
    def __init__(self, msg):
        super().__init__()
        self.setWindowTitle('warning')
        self.setText(msg)
        self.setIcon(QMessageBox.Warning)


class _TEXT:
    TAB_PRJ = 'project'
    TAB_DATA = 'data'
    TAB_TASK = 'tasks'


class MainWindow(QWidget):
    def __init__(self, store):
        super().__init__()
        from .. import AppStore
        self._store:AppStore = store
        self._store.websocket_server.register_handler(
            print_log_msgs, Messages.GotLogMsg
        )

        # initial ui stuff
        self.setWindowTitle('MEF-Agri')
        self.showMaximized()
        self._l = QHBoxLayout()

        # creating the main-tab-widget with tabs
        self._tabs = QTabWidget()
        self._tabs.tabBarClicked.connect(self.init_tabs)
        self._tab_prj = ProjectTab(self, self._store)
        self._tab_data = DataTab(self, self._store)
        self._tab_task = TasksTab(self, self._store)
        self._tabs.addTab(self._tab_prj, _TEXT.TAB_PRJ)
        self._tabs.addTab(self._tab_data, _TEXT.TAB_DATA)
        self._tabs.addTab(self._tab_task, _TEXT.TAB_TASK)

        # final ui stuff
        self._l.addWidget(self._tabs, 1)
        self._store.map_view = MapView(self)
        self._store.map_view.load_html('index')
        self._l.addWidget(self._store.map_view, 1)
        self.setLayout(self._l)

    def init_tabs(self, index):
        msg = Messages.SendActiveTab()

        if index == self._tabs.indexOf(self._tab_prj):
            msg.tab_name = _TEXT.TAB_PRJ
        elif index == self._tabs.indexOf(self._tab_data):
            msg = self._init_non_prj_tab(msg, self._tab_data, _TEXT.TAB_DATA)
        elif index == self._tabs.indexOf(self._tab_task):
            msg = self._init_non_prj_tab(msg, self._tab_task, _TEXT.TAB_TASK)

        self._store.websocket_server.send_messages(msg)

    def _init_non_prj_tab(self, msg, tab, tabname):
        if self._store.project_data is None:
            return msg
        msg.tab_name = tabname
        if not tab.initialized:
            tab.init_tab()
        return msg
