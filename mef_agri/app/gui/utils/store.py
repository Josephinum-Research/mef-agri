import os
from inspect import isclass

from ...gui.conn.server import WebsocketServer
from ....data.interface import Interface
from ....data.project import ProjectData


class AppStore(object):
    def __init__(self):
        self._wss:WebsocketServer = None
        self._dis:list[Interface] = None
        self._prj:ProjectData = None
        self._pp:str = None

    @property
    def websocket_server(self) -> WebsocketServer:
        """
        :return: websocket instance to enable communication between qt-based gui with web-based map
        :rtype: WebsocketServer
        """
        return self._wss
    
    @websocket_server.setter
    def websocket_server(self, wss):
        self._wss = wss

    @property
    def project_path(self) -> str:
        """
        :return: absolute path where project-database (.gpkg) is located
        :rtype: str
        """
        return self._pp
    
    @project_path.setter
    def project_path(self, path):
        if os.path.isdir(path):
            self._pp = path
    
    @property
    def project_data(self) -> ProjectData:
        """
        :return: object for project-db access and data-handling
        :rtype: ProjectData
        """
        return self._prj
    
    @project_data.setter
    def project_data(self, prj):
        self._prj = prj

    @property
    def data_interfaces(self) -> list[Interface]:
        """
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

