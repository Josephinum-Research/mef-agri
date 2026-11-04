import os
from inspect import isclass
from PyQt5.QtWebEngineWidgets import QWebEngineView

from ...gui.conn.server import WebsocketServer
from ....data.interface import Interface
from ....evaluation.db import EvalDB
from ..project import ProjectDataGUI
from ....farming.tasks import ProjectTasksExtension


class AppStore(object):
    def __init__(self):
        self._wss:WebsocketServer = None
        self._dis:list[Interface] = None
        self._prj:ProjectDataGUI = None
        self._pp:str = None
        self._map:QWebEngineView = None
        self._edb:EvalDB = None
        self._prj_dt:ProjectTasksExtension = None

    @property
    def websocket_server(self) -> WebsocketServer:
        """
        settable

        :return: websocket instance to enable communication between qt-based gui with web-based map
        :rtype: WebsocketServer
        """
        return self._wss
    
    @websocket_server.setter
    def websocket_server(self, wss):
        self._wss = wss

    @property
    def map_view(self) -> QWebEngineView:
        """
        settable

        :return: openlayers map
        :rtype: QWebEngineView
        """
        return self._map
    
    @map_view.setter
    def map_view(self, mview):
        self._map = mview

    @property
    def project_path(self) -> str:
        """
        settable

        :return: absolute path where project-database (.gpkg) is located
        :rtype: str
        """
        return self._pp
    
    @project_path.setter
    def project_path(self, path):
        if os.path.isdir(path):
            self._pp = path
    
    @property
    def project_data(self) -> ProjectDataGUI:
        """
        settable

        :return: object for project-db access and data-handling
        :rtype: ProjectData
        """
        return self._prj
    
    @project_data.setter
    def project_data(self, prj):
        self._prj = prj

    @property
    def project_data_tasks(self) -> ProjectTasksExtension:
        """
        :return: :func:`project_data` extended by task-information
        :rtype: ProjectTasksExtension
        """
        return self._prj_dt

    @project_data_tasks.setter
    def project_data_tasks(self, prjdt):
        self._prj_dt = prjdt

    @property
    def data_interfaces(self) -> list[Interface]:
        """
        settable

        :return: data-interface instances provided by the user which should be available in the current session (if not already available in the project-db)
        :rtype: list[Interface]
        """
        return self._dis
    
    @data_interfaces.setter
    def data_interfaces(self, dis):
        if dis is None:
            return
        self._dis = []
        for di in dis:
            if isclass(di):
                self._dis.append(di())
            else:
                self._dis.append(di)

    @property
    def eval_db(self) -> EvalDB:
        """
        settable

        :return: database which contains zoning information as well as input and output data of model evaluation
        :rtype: EvalDB
        """
        return self._edb

    @eval_db.setter
    def eval_db(self, edb):
        self._edb = edb
