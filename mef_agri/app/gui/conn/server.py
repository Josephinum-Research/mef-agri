import json
from inspect import isclass
from threading import Thread
from websockets.sync.server import Server, serve

from .msgs import Messages, MsgBaseClass

class Errors:
    class UnknownMsgType(Exception):
        def __init__(self, *args):
            super().__init__(*args)


class WebsocketServer(Thread):
    def __init__(self, host='localhost', port=33611):
        super().__init__(daemon=True)
        serve
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
        """
        Handler function for incoming messages

        :param ws: websockets connection object, which holds information on client and messages
        :type ws: ServerConnection
        """
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
            
    def send_messages(self, msgs:list[MsgBaseClass] | MsgBaseClass):
        """
        Send message via websocket to the web-components

        :param msgs: child class ``mef_agri.app.gui.conn.msgs.MsgBaseClass``
        :type msgs: list[mef_agri.app.gui.conn.msgs.MsgBaseClass] | mef_agri.app.gui.conn.msgs.MsgBaseClass
        """
        if not isinstance(msgs, list):
            msgs = [msgs,]
        for ws in self._clients:
            for msg in msgs:
                ws.send(msg.message)
