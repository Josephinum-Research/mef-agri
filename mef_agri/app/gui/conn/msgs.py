import json


class MsgBaseClass(object):
    MTYPE = None

    def __init__(self, msgcont:dict=None):
        """
        :param msgcont: message with appropriate content (if available), defaults to None
        :type msgcont: dict, optional
        """
        if msgcont is None:
            msgcont = {}
        for aname, aval in self.__class__.__dict__.items():
            if aname.startswith('CONT_'):
                if aval in msgcont.keys():
                    val = msgcont[aval]
                else:
                    val = None
                setattr(self, '_' + aval, val)

    @property
    def message(self) -> str:
        """
        :return: message which should be passed to websocket
        :rtype: str
        """
        msg = {Messages.KEY_MTYPE: self.MTYPE}
        cont = {}
        for aname, aval in self.__class__.__dict__.items():
            if aname.startswith('CONT_'):
                cont[aval] = getattr(self, '_' + aval)
        msg[Messages.KEY_CONT] = cont
        return json.dumps(msg)


class Messages:
    KEY_MTYPE = 'type'
    KEY_CONT = 'content'

    class GotDrawnField(MsgBaseClass):
        MTYPE = 'drawn_field'
        CONT_COORDS = 'coords'
        CONT_EPSG = 'epsg'
        CONT_FNAME = 'fname'

        @property
        def points(self) -> list:
            """
            :return: coordinates of points representing the polygon
            :rtype: list
            """
            return getattr(self, '_' + self.CONT_COORDS)
        
        @property
        def epsg(self) -> int:
            """
            :return: epsg-code of crs in which :func:`points` are defined
            :rtype: int
            """
            return getattr(self, '_' + self.CONT_EPSG)
        
        @property
        def field_name(self) -> str:
            """
            :return: name of the field
            :rtype: str
            """
            return getattr(self, '_' + self.CONT_FNAME)
        
    class GotDeleteField(MsgBaseClass):
        MTYPE = 'delete_field'
        CONT_FNAME = 'fname'

        @property
        def field_name(self) -> str:
            """
            :return: name of the field which should be deleted
            :rtype: str
            """
            return getattr(self, '_' + self.CONT_FNAME)

    class GotLogMsg(MsgBaseClass):
        MTYPE = 'logmsg'
        CONT_LOGMSG = 'logmsg'

        @property
        def log_message(self) -> str:
            return getattr(self, '_' + self.CONT_LOGMSG)

    class SendFields(MsgBaseClass):
        MTYPE = 'field_defs'
        CONT_FNAMES = 'fnames'
        CONT_COORDS = 'coords'

        @property
        def field_names(self) -> list[str]:
            """
            :return: (unique) name of the field
            :rtype: str
            """
            return getattr(self, '_' + self.CONT_FNAMES)
        
        @field_names.setter
        def field_names(self, value):
            setattr(self, '_' + self.CONT_FNAMES, value)

        @property
        def coordinates(self) -> list:
            """
            :return: list of polygon-defining point-coordinate-lists - coordinates have to be defined in web-mercator (epsg:3857)
            :rtype: list[list[list[float, float]]]
            """
            return getattr(self, '_' + self.CONT_COORDS)
        
        @coordinates.setter
        def coordinates(self, value):
            setattr(self, '_' + self.CONT_COORDS, value)

    class SendActiveTab(MsgBaseClass):
        MTYPE = 'active_tab'
        CONT_TABNAME = 'tabname'

        @property
        def tab_name(self) -> str:
            """
            :return: name of the currently active tab
            :rtype: str
            """
            return getattr(self, '_' + self.CONT_TABNAME)
        
        @tab_name.setter
        def tab_name(self, value):
            setattr(self, '_' + self.CONT_TABNAME, value)
