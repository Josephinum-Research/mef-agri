from PyQt5.QtWidgets import QApplication

from .gui.conn.server import WebsocketServer
from .gui import MainWindow


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
