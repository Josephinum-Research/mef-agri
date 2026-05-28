import json
from threading import Thread
from websockets.sync.server import Server, serve


class WebsocketServer(Thread):
    def __init__(self, host='localhost', port=33611):
        super().__init__(daemon=True)
        self._srvr:Server = serve(self.incoming_messages, host, port)
        self.host = host
        self.port = port

    def run(self):
        self._srvr.serve_forever()

    def stop(self):
        self._srvr.shutdown()

    def incoming_messages(self, ws):
        for msg in ws:
            print(msg)

    def broadcast_messages(self, msgs):
        # broadcast messages to registered connections/clients
        pass


"""
if __name__ == '__main__':
    server = WebsocketServer()
    server.start()

    ws:ClientConnection = connect('ws://{}:{}'.format(server.host, server.port))
    while True:
        print('message for server')
        msg = input()
        if msg == 'exit':
            server.stop()
            server.join()
            print('stopped server')
            break
        ws.send(msg)
        print(ws.recv())
        print('---------------------------------------------------------------')
"""
