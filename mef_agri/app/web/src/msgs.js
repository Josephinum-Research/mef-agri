/**
 * Class acting as namespace for messages which enable communication between 
 * frontend (PyQT) and map in WebEngine.
 * @class
 */
export class Messages {
    static keyMsgType = 'type';
    static keyMsgCont = 'content';

    ////////////////////////////////////////////////////////////////////////////
    // SEND-MESSAGES
    ////////////////////////////////////////////////////////////////////////////
    /**
     * Base class for messages which should be sent to the frontend
     */
    static SendMessage = class {
        #msg = {};

        constructor () {
            this.#msg[Messages.keyMsgType] = null;
            this.#msg[Messages.keyMsgCont] = {};
        }

        /**
         * There has to be a counterpart for message-content key in 
         * `mef_agri.app.gui.conn.msgs.Messages`, i.e. a class variable starting 
         * with `CONT_` in a nested class.
         * 
         * @param {string} key - key/identifier of a content field 
         * @param {any} val - value of a content field (should be JSON-serializable)
         */
        setCont(key, val) {
            this.#msg[Messages.keyMsgCont][key] = val;
        }

        /**
         * Setter for message-type.
         * 
         * There has to be a counterpart for the message-type in 
         * `mef_agri.app.gui.conn.msgs.Messages`, i.e. a class variable 
         * `MTYPE` of a nested class.
         * 
         * @param {string} val - type/identifier of message
         */
        set msgType(val) {
            this.#msg[Messages.keyMsgType] = val;
        }

        /**
         * @returns {object} - message (containing all information) which will be parsed with JSON and sent to the frontend
         */
        get message() {
            return this.#msg;
        }
    }

    /**
     * Class to send drawn field/polygons (coordinates and labels) to the 
     * frontend.
     * 
     * The crs/epsg-code is fixed to WGS84/4326.
     */
    static SendDrawnField = class extends Messages.SendMessage {
        constructor() {
            super();
            this.msgType = 'drawn_field';
            this.setCont('epsg', 4326);
        }

        /**
         * Setter for the field name.
         * 
         * @param {string} fname - name of the drawn field/polygon
         */
        set fieldName(fname) {
            this.setCont('fname', fname);
        }

        /**
         * Setter for the corner-coordinates of the drawn field/polygon
         * 
         * @param {Array} coords - array with corners/points (i.e. again arrays with two coordinates)
         */
        set cornerCoords(coords) {
            this.setCont('coords', coords);
        }
    };

    /**
     * Class to send log-messages to frontend
     */
    static SendLogMsg = class extends Messages.SendMessage {
        constructor() {
            super();
            this.msgType = 'logmsg';
        }
        
        /**
         * setter
         * @param {string} msg - log-message which should be sent to the frontend
         */
        set logMsg(msg) {
            this.setCont('logmsg', msg);
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // INCOMING-MESSAGES
    ////////////////////////////////////////////////////////////////////////////
    /**
     * Base class for messages which come from the frontend.
     */
    static GotMessage = class {
        #msgCont;
        constructor(msg) {
            this.#msgCont = msg[Messages.keyMsgCont];
        }

        /**
         * @param {string} key - key specifying the content field for which a value should be returned
         * @returns {any} - value which belongs to the content-field
         */
        getCont(key) {
            return this.#msgCont[key];
        }
    }

    /**
     * Class which handles incoming messages containing information about 
     * available fields.
     */
    static GeoFieldInfo = class extends Messages.GotMessage {
        static msgType = 'field_defs';

        /**
         * @returns {Array} - array with field names being strings
         */
        get fieldNames() {
            return this.getCont('fnames');
        }

        /**
         * @returns {Array} - array containing polygon-defining corner-coordinate-lists - defined in WGS84
         */
        get cornerCoords() {
            return this.getCont('coords');
        }
    }
}
