import os
from PyQt5.QtWebEngineWidgets import QWebEngineView


class MapView(QWebEngineView):
    def __init__(self, parent):
        super().__init__(parent)

        # initialize html
        cdir = os.path.split(__file__)[0]
        fio = open(os.path.join(cdir, 'map.html'), 'r')
        html = fio.read()
        fio.close()
        self.setHtml(html)

        # define websocket stuff

