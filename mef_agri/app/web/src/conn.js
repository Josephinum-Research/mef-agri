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
        this.ws.addEventListener('open', this.handleOpenConn.bind(this));
        this.ws.addEventListener('message', this.incomingMessage.bind(this));
        
        this.connOpen = false;
        this.log_recv = [];
    }

    handleOpenConn() {
        this.connOpen = true;
    }

    incomingMessage(event) {
        this.log_recv.push(JSON.parse(event.data));
    }

    sendPolygon(feat, fname) {
        var mdef = new SendDrawnField(
            feat.getGeometry().getCoordinates(), fname
        );
        this.ws.send(JSON.stringify(mdef.message));
    }

    sendLogMessage(msg) {
        mdef = new SendLogMsg(msg)
        this.ws.send(JSON.stringify(mdef.message));
    }
}