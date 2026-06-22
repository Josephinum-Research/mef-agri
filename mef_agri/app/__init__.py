from PyQt5.QtWidgets import QApplication

from .gui.conn.server import WebsocketServer
from .gui import MainWindow
from .gui.utils.store import AppStore



_SET_STORE_ATTRS = [
    'project_path', 'data_interfaces'
]


def run_app(**kwargs):
    try:
        wss = WebsocketServer()
        wss.start()

        store = AppStore()
        store.websocket_server = wss
        for key, val in kwargs.items():
            if key in _SET_STORE_ATTRS:
                setattr(store, key, val)

        app = QApplication([])
        mw = MainWindow(store)
        app.exec()
    except Exception as exc:
        print(exc)
    finally:
        wss.stop()
        wss.join()
