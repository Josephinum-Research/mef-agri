import json
from inspect import isclass
from threading import Thread
from websockets.sync.server import Server, serve
from websockets import broadcast


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
        CONT_POINTS = 'points'
        CONT_EPSG = 'epsg'
        CONT_FNAME = 'fname'

        @property
        def points(self) -> list:
            """
            :return: coordinates of points representing the polygon
            :rtype: list
            """
            return self._points
        
        @property
        def epsg(self) -> int:
            """
            :return: epsg-code of crs in which :func:`points` are defined
            :rtype: int
            """
            return self._epsg
        
        @property
        def field_name(self) -> str:
            """
            :return: name of the field
            :rtype: str
            """
            return self._fname

    class GotLogMsg(MsgBaseClass):
        MTYPE = 'logmsg'
        CONT_LOGMSG = 'logmsg'

        @property
        def log_message(self) -> str:
            return self._logmsg

    class SendFieldInfo(MsgBaseClass):
        MTYPE = 'field_info'
        CONT_FID = 'fid'
        CONT_FNAME = 'fname'

        @property
        def field_id(self) -> int:
            """
            :return: feature-id from geopackage
            :rtype: int
            """
            return self._fid
        
        @field_id.setter
        def field_id(self, value):
            self._fid = value

        @property
        def field_name(self) -> str:
            """
            :return: name of the field
            :rtype: str
            """
            return self._fname
        
        @field_name.setter
        def field_name(self, value):
            self._fname = value


class Errors:
    class UnknownMsgType(Exception):
        def __init__(self, *args):
            super().__init__(*args)


class WebsocketServer(Thread):
    def __init__(self, host='localhost', port=33611):
        super().__init__(daemon=True)
        self._srvr:Server = serve(self.incoming_messages, host, port)
        self._clients = set()
        self.host = host
        self.port = port
        self._hs = {}
        self._mts = []
        for attr in Messages.__dict__.values():
            if isclass(attr):
                if (
                    hasattr(attr, 'MTYPE') and 
                    (getattr(attr, 'MTYPE') is not None)
                ):
                    self._mts.append(getattr(attr, 'MTYPE'))


    def run(self):
        self._srvr.serve_forever()

    def stop(self):
        self._srvr.shutdown()

    def register_handler(self, handler:function, msg_class):
        """
        Register handlers for incoming messages passed through the websocket 
        which connects PyQT-GUI with web-contents (i.e. openlayers-map).
        The content of incoming messages will be used to initialize the 
        message object which will be passed to the handler.

        :param handler: function or method which will be called
        :type handler: function
        :param msg_class: message class/definition (i.e. nested classes in :class:`Messages`)
        :type msg_class: class
        """
        self._hs[msg_class.MTYPE] = {'handler': handler, 'msg_class': msg_class}

    def incoming_messages(self, ws):
        self._clients.add(ws)
        for rmsg in ws:
            msg = json.loads(rmsg)
            if msg[Messages.KEY_MTYPE] in self._mts:
                if msg[Messages.KEY_MTYPE] in self._hs:
                    hdef = self._hs[msg[Messages.KEY_MTYPE]]
                    msgobj = hdef['msg_class'](msg[Messages.KEY_CONT])
                    hdef['handler'](msgobj)
                else:
                    try:
                        print(msg)
                    except Exception as exc:
                        print(exc)
            else:
                errmsg = self.__class__.__name__ + ' >>> Message type `{}` '
                errmsg += 'not known!'
                raise Errors.UnknownMsgType(
                    errmsg.format(msg[Messages.KEY_MTYPE])
                )

    def broadcast_messages(self, msgs:list[MsgBaseClass] | MsgBaseClass):
        if not isinstance(msgs, list):
            msgs = [msgs,]
        for msg in msgs:
            broadcast(self._clients, json.dumps(msg.message))
