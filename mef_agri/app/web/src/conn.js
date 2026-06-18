import { Messages } from "./msgs";

export class AppConnection {
    #handlers = {};

    constructor () {
        this.ws = new WebSocket('ws://127.0.0.1:33611/');
        this.ws.addEventListener('message', this.#incomingMessage.bind(this));
        this.ws.addEventListener('open', this.#connEstablished.bind(this));
        this.connOpen = false;
    }

    #connEstablished() {
        this.connOpen = true;
        // send log message to frontend
        var logmsg = '(conn.js) AppConnection.connEstablished => ';
        logmsg += 'Connection successfully established';
        var msg = new Messages.SendLogMsg();
        msg.logMsg = logmsg;
        this.send(msg);
    }

    /**
     * Register a handler-function which will be called when a message with 
     * corresponding type arrives.
     * 
     * @param {any} msgClass - class definition of message which should be handled
     * @param {func} func - function which should be used to handle the corresponding message
     * @param {object} obj - if `func` is a method, the object/instance should be provided to use `func(...).bind(obj)` when message will be handled - defaults to `null`
     */
    registerHandler(msgClass, func, obj=null) {
        this.#handlers[msgClass.msgType] = {
            msgClass: msgClass,
            func: func,
            obj: obj
        };
    }

    #incomingMessage(event) {
        var logmsg = Messages.SendLogMsg();
        logmsg.logMsg = '(conn.js) - message from frontend arrived';
        this.send(logmsg);
        var msg = JSON.parse(event.data);
        var hdefs = this.#handlers[msg['type']];
        hdefs.func(new hdefs.msgClass(msg)).bind(hdefs.obj);
    }

    #sendConnOpen(msg_str) {
        if (this.connOpen) {
            this.ws.send(msg_str);
        } else {
            setTimeout(this.#sendConnOpen.bind(this), 10, msg_str);
        }
    }

    /**
     * Send message to the frontend.
     * 
     * @param {object} msg - object/instance of child-class of `Messages.SendMessage`
     */
    send(msg) {
        this.#sendConnOpen(JSON.stringify(msg.message));
    }
}
