import { Messages } from "./msgs";

export class AppConnection {
    static logMsgPrefix = '(conn.js) AppConnection';
    #handlers = {};

    constructor () {
        this.ws = new WebSocket('ws://127.0.0.1:33611/');
        this.ws.addEventListener('message', this.#incomingMessage.bind(this));
        this.ws.addEventListener('open', this.#connEstablished.bind(this));
        this.connOpen = false;
        this.dummy = 'hello';
    }

    #connEstablished() {
        this.connOpen = true;
        // send log message to frontend
        var logmsg = AppConnection.logMsgPrefix + '.connEstablished => ';
        logmsg += 'Connection successfully established';
        var msg = new Messages.SendLogMsg();
        msg.logMsg = logmsg;
        this.send(msg);
    }

    /**
     * Register a handler-function which will be called when a message with 
     * corresponding type arrives.
     * 
     * `func` has to accept exactly two arguments:
     * 
     * - the first argument is an arbitrary object which should be accessible within `func` (or null if not required)
     * - the second argument corresponds to the message-object which should be handled by `func` (i.e. instance of `msgClass`)
     * 
     * @param {any} msgClass - class definition of message which should be handled
     * @param {func} func - function which should be used to handle the corresponding message
     * @param {object} obj - object/instance which should be accessible within `func`
     */
    registerHandler(msgClass, func, obj=null) {
        this.#handlers[msgClass.msgType] = {
            msgClass: msgClass,
            func: func,
            obj: obj
        };
    }

    #incomingMessage(event) {
        var msg = JSON.parse(event.data);
        if (Object.keys(this.#handlers).length > 0) {
            if (msg[Messages.keyMsgType] in this.#handlers) {
                var hdefs = this.#handlers[msg['type']];
                hdefs.func(hdefs.obj, new hdefs.msgClass(msg));
                return;
            }
        }
        var sendlog = new Messages.SendLogMsg();
        var logmsg = AppConnection.logMsgPrefix + '.#incomingMessages ';
        logmsg += '=> no handler registered for message-type: ';
        logmsg += msg[Messages.keyMsgType];
        sendlog.logMsg = logmsg;
        this.send(sendlog);
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

    sendLog(log) {
        var logmsg = new Messages.SendLogMsg();
        logmsg.logMsg = log;
        this.send(logmsg);
    }
}
