import os
from PyQt5.QtWidgets import QApplication

from mef_agri.app import MainWindow

if __name__ == '__main__':
    app = QApplication([])
    mw = MainWindow()
    app.exec()
