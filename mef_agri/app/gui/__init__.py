from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QTabWidget, QMessageBox
)

from .project import ProjectTab
from .data import DataTab
from .conn.server import WebsocketServer, Messages
from ...data.project import ProjectData


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
    def __init__(self, wss:WebsocketServer):
        super().__init__()
        self._wss:WebsocketServer = wss
        self._wss.register_handler(print_log_msgs, Messages.GotLogMsg)
        
        # initial ui stuff
        self.setWindowTitle('MEF-Agri')
        self.showMaximized()
        self._l = QHBoxLayout()

        # creating the main-tab-widget with tabs
        self._tabs = QTabWidget()
        self._tabs.tabBarClicked.connect(self.init_tabs)
        self._tab_prj = ProjectTab(self)
        self._tab_data = DataTab(self)
        self._tabs.addTab(self._tab_prj, 'project')
        self._tabs.addTab(self._tab_data, 'data')

        # final ui stuff
        self._l.addWidget(self._tabs)
        self.setLayout(self._l)

    @property
    def websocket_server(self) -> WebsocketServer:
        return self._wss
    
    def init_tabs(self):
        if self._tab_prj.selected_project is None:
            return
        if not self._tab_data.initialized:
            self._tab_data.init_tab()
    