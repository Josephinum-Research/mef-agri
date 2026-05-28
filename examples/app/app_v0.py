from PyQt5.QtWidgets import QApplication

from mef_agri.app.gui import MainWindow
from mef_agri.app.gui.server import WebsocketServer

if __name__ == '__main__':
    try:
        ws = WebsocketServer()
        ws.start()

        app = QApplication([])
        mw = MainWindow()
        app.exec()
    except Exception as exc:
        print(exc)
    finally:
        ws.stop()
        ws.join()
