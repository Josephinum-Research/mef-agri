from PyQt5.QtWidgets import QWidget, QHBoxLayout, QTabWidget, QApplication

from .project import ProjectTab
from .server import WebsocketServer


class MainWindow(QWidget):
    def __init__(self, wss:WebsocketServer):
        super().__init__()
        self._wss:WebsocketServer = wss
        
        self.setWindowTitle('MEF-Agri')
        self.showMaximized()
        self._l = QHBoxLayout()
        self._tabs = QTabWidget()
        self._tabs.addTab(ProjectTab(self), 'Project')
        self._l.addWidget(self._tabs)
        self.setLayout(self._l)

    @property
    def websocket_server(self) -> WebsocketServer:
        return self._wss
    

def run_app():
    try:
        wss = WebsocketServer()
        wss.start()

        app = QApplication([])
        mw = MainWindow(wss)
        app.exec()
    except Exception as exc:
        print(exc)
    finally:
        wss.stop()
        wss.join()
