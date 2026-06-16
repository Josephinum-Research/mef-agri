import TileLayer from "ol/layer/Tile";
import View from "ol/View";
import Map from 'ol/Map.js';
import WMTS, {optionsFromCapabilities} from 'ol/source/WMTS.js';
import WMTSCapabilities from 'ol/format/WMTSCapabilities.js';
import { useGeographic } from 'ol/proj';
import { defaults as defaultControls } from 'ol/control/defaults.js';
import VectorSource from 'ol/source/Vector';
import VectorLayer from 'ol/layer/Vector';
import Style from "ol/style/Style";
import Text from "ol/style/Text";
import Fill from "ol/style/Fill";
import Stroke from "ol/style/Stroke";

export class FieldMap {
    static logMsgPrefix = '(fmap.js) FieldMap';

    constructor(appConn) {
        this.appConn = appConn;
        ////////////////////////////////////////////////////////////////////////
        // WMTS ortho-image
        this.wmtsUrl = 'https://mapsneu.wien.gv.at/basemapneu/1.0.0/WMTSCapabilities.xml';
        this.wmtsLayerName = 'bmaporthofoto30cm';
        this.wmtsEpsg = 'EPSG:4326';
        this.wmtsLayer = null;

        ////////////////////////////////////////////////////////////////////////
        // FIELDS vector-layer
        this.labelStyle = new Style({
            text: new Text({
                font: '13px Calibri,sans-serif',
                fill: new Fill({color: '#000'}),
                stroke: new Stroke({color: '#fff', width: 4})
            })
        })
        this.fldSource = new VectorSource({wrapX: false});
        this.fldLayer = new VectorLayer({
            source: this.fldSource,
            style: this.fieldStyle.bind(this)
        });

        ////////////////////////////////////////////////////////////////////////
        // MAP STUFF
        this.initPos = [15.1445, 48.1328];
        this.initZoom = 14;
        this.map = null;
        this.controls = [];
        this.interactions = [];
    }

    initializeWMTS() {
        const wmtsSettings = {
            layer: this.wmtsLayerName,
            matrixSet: this.wmtsEpsg
        };
        fetch(this.wmtsUrl)
            .then(function (response) {
                return response.text();
            })
            .then(function (text) {
                const wmtsParser = new WMTSCapabilities();
                const result = wmtsParser.read(text);
                const options = optionsFromCapabilities(result, wmtsSettings);
                const wmtsSource = new WMTS(options);
                this.wmtsLayer = new TileLayer({source: wmtsSource});
                // logging
                var msg = FieldMap.logMsgPrefix + '.initializeWMTS => ';
                msg += 'Basemap ortho-image successfully initialized';
                this.appConn.sendLogMessage(msg);
            }.bind(this));
    }

    addCustomControl(control) {
        this.controls.push(control);
    }

    addCustomInteraction(interaction) {
        this.interactions.push(interaction);
    }

    fieldStyle(feature) {
        const textDef = [`${feature.get('fname')}`, ''];
        this.labelStyle.getText().setText(textDef);
        return [this.labelStyle];
    }

    #createMap() {
        useGeographic();
        const initView = new View({
            center: this.initPos,
            zoom: this.initZoom
        })
        this.map = new Map({
            target: 'map',
            layers: [this.wmtsLayer, this.fldLayer],
            view: initView,
            controls: defaultControls().extend(this.controls)
        });
        for (const interaction of this.interactions) {
            this.map.addInteraction(interaction);
        }
    }
 
    run() {
        if (this.wmtsLayer != null) {
            this.#createMap();
            var msg = FieldMap.logMsgPrefix + '.run => ';
            msg += 'map successfully created';
            this.appConn.sendLogMessage(msg);
        } else {
            setTimeout(this.run.bind(this), 10);
        }
    }
}