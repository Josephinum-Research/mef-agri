////////////////////////////////////////////////////////////////////////////////
///   WEBSOCKET TO CONNECT WEB WITH GUI
////////////////////////////////////////////////////////////////////////////////
class SendDrawnField {
    constructor(coords, fname) {
        this.message = {
            type: 'drawn_field',
            content: {
                points: coords,
                epsg: 4326,
                fname: fname
            }
        };
    }
}


class SendLogMsg{
    constructor(msg) {
        this.message = {
            type: 'logmsg',
            content: {
                logmsg: msg
            }
        }
    }
}


class GotFieldInfo{}


export class AppConnection {
    constructor () {
        this.ws = new WebSocket('ws://127.0.0.1:33611/');
        this.ws.addEventListener('message', this.#incomingMessage.bind(this));
        this.ws.addEventListener('open', this.#connEstablished.bind(this));
        this.connOpen = false;
    }

    #connEstablished() {
        var msg = '(conn.js) AppConnection.connEstablished => ';
        msg += 'Connection successfully established';
        this.sendLogMessage(msg);
        this.connOpen = true;
    }

    #incomingMessage(event) {}

    sendPolygon(feat, fname) {
        var msg = new SendDrawnField(
            feat.getGeometry().getCoordinates(), fname
        );
        this.#send(msg);
    }

    sendLogMessage(logmsg) {
        var msg = new SendLogMsg(logmsg);
        this.#send(msg);
    }

    #sendConnOpen(msg_str) {
        if (this.connOpen) {
            this.ws.send(msg_str);
        } else {
            setTimeout(this.#sendConnOpen.bind(this), 10, msg_str);
        }
    }

    #send(msg) {
        var msg_str = JSON.stringify(msg.message);
        this.#sendConnOpen(msg_str);
    }
}