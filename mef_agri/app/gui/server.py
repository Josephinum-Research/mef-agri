from threading import Thread
from websockets.sync.server import serve, broadcast


class WebSocketServer(Thread):
    def __init__(self, host='localhost', port='32501'):
        super().__init__(daemon=True)
        self._server = serve(self.handle_msg, host, port)

    def run(self):
        self._server.serve_forever()

    def stop(self):
        self._server.shutdown()

    @staticmethod
    def handle_msg(websocket):
        cont = websocket.recv()
        print(cont)
        websocket.send('got information!')


class WebSocketClient(object):
    pass
