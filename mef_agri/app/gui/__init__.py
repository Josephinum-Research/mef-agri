from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QTabWidget, QMessageBox
)

from .project import ProjectTab
from .data import DataTab
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


class _ErrorDialogs:
    @staticmethod
    def no_prj_selected():
        msg = 'No project selected yet!'
        dlg = _CustomErrorDialog(msg)
        dlg.exec()


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
        self._tabs.addTab(self._tab_prj, 'project')
        self._tabs.addTab(self._tab_data, 'data')

        # final ui stuff
        self._l.addWidget(self._tabs, 1)
        self._store.map_view = MapView(self)
        self._store.map_view.load_html('tab_prj')
        self._l.addWidget(self._store.map_view, 1)
        self.setLayout(self._l)

    def init_tabs(self, index):
        if index == self._tabs.indexOf(self._tab_data):
            if self._store.project_data is None:
                return
            if not self._tab_data.initialized:
                self._tab_data.init_tab()
