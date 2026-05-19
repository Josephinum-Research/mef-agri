import os
from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings
from PyQt5.QtWebEngineCore import QWebEngineUrlRequestInterceptor


class _Interceptor(QWebEngineUrlRequestInterceptor):
    def interceptRequest(self, info):
        info.setHttpHeader(
            "Accept-Language", "en-US,en;q=0.9,es;q=0.8,de;q=0.7"
        )


class MapView(QWebEngineView):
    def __init__(self, parent):
        super().__init__(parent)

        # set intercepter, for explanation see
        # https://stackoverflow.com/questions/66925445/qt-webengine-not-loading-openstreetmap-tiles/66926399#66926399
        interceptor = _Interceptor()
        self.page().profile().setUrlRequestInterceptor(interceptor)

        self.page().settings().setAttribute(
            QWebEngineSettings.LocalContentCanAccessRemoteUrls, True
        )

        # load html
        html = os.path.join(
            os.path.split(__file__)[0], os.pardir, 'web', 'dist', 'index.html'
        )
        self.load(QUrl.fromLocalFile(html))
