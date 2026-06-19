import TileLayer from "ol/layer/Tile";
import View from "ol/View";
import Map from 'ol/Map.js';
import WMTS, {optionsFromCapabilities} from 'ol/source/WMTS.js';
import WMTSCapabilities from 'ol/format/WMTSCapabilities.js';
import { defaults as defaultControls } from 'ol/control/defaults.js';
import VectorSource from 'ol/source/Vector';
import VectorLayer from 'ol/layer/Vector';
import Style from "ol/style/Style";
import Text from "ol/style/Text";
import Fill from "ol/style/Fill";
import Stroke from "ol/style/Stroke";
import { Polygon } from "ol/geom";
import Feature from "ol/Feature";
import { Messages } from "./msgs";
import { fromLonLat } from "ol/proj";
import { OSM } from "ol/source";


export class FieldLayerStyle {
    constructor() {
        this.strokeColor = 'rgb(0, 255, 150)';
        this.strokeWidth = 4;
        this.fillColor = 'rgba(0, 255, 150, 0.25)';
        this.textAlign = 'center';
        this.textOverflow = true;
        this.textFont = '16px Calibri,sans-serif';
        this.textFillColor = 'rgb(0, 0, 0)';
        this.textStrokeColor = 'rgb(255, 255, 255)';
        this.textStrokeWidth = 2;
    }
}


export class FieldMap {
    static logMsgPrefix = '(fmap.js) FieldMap';

    constructor(appConn) {
        this.appConn = appConn;
        ////////////////////////////////////////////////////////////////////////
        this.osmLayer = new TileLayer({source: new OSM()});

        ////////////////////////////////////////////////////////////////////////
        // WMTS ortho-image
        this.wmtsUrl = 'https://mapsneu.wien.gv.at/basemapneu/1.0.0/WMTSCapabilities.xml';
        this.wmtsLayerName = 'bmaporthofoto30cm';
        this.wmtsLayer = null;

        ////////////////////////////////////////////////////////////////////////
        // FIELDS vector-layer
        this.fldStyleOptions = new FieldLayerStyle();
        this.fldSource = new VectorSource({wrapX: false});
        this.fldLayer = new VectorLayer({
            source: this.fldSource,
            style: this.#getFieldStyle.bind(this)
        });

        ////////////////////////////////////////////////////////////////////////
        // MAP STUFF
        this.initPos = fromLonLat([15.1445, 48.1328]);
        this.initZoom = 14;
        this.map = null;
        this.mapView = null;
        this.controls = [];
        this.interactions = [];
    }

    initializeWMTS() {
        const wmtsSettings = {
            layer: this.wmtsLayerName,
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
                var logmsg = FieldMap.logMsgPrefix + '.initializeWMTS => ';
                logmsg += 'Basemap ortho-image successfully initialized';
                var msg = new Messages.SendLogMsg();
                msg.logMsg = logmsg;
                this.appConn.send(msg);
            }.bind(this));
    }

    addCustomControl(control) {
        this.controls.push(control);
    }

    addCustomInteraction(interaction) {
        this.interactions.push(interaction);
    }

    #getFieldStyle(feature) {
        return new Style({
            stroke: new Stroke({
                color: this.fldStyleOptions.strokeColor,
                width: this.fldStyleOptions.strokeWidth
            }),
            fill: new Fill({
                color: this.fldStyleOptions.fillColor
            }),
            text: new Text({
                textAlign: this.fldStyleOptions.textAlign,
                overflow: this.fldStyleOptions.textOverflow,
                font: this.fldStyleOptions.textFont,
                stroke: new Stroke({
                    color: this.fldStyleOptions.textStrokeColor,
                    width: this.fldStyleOptions.textStrokeWidth
                }),
                fill: new Fill({
                    color: this.fldStyleOptions.textFillColor
                }),
                text: feature.get('fname')
            })
        });
    }

    #createMap() {
        this.mapView = new View({
            center: this.initPos,
            zoom: this.initZoom
        })
        this.map = new Map({
            target: 'map',
            //projection: new Projection('EPSG:' + this.mapEpsg.toString()),
            layers: [this.osmLayer, this.wmtsLayer, this.fldLayer],
            view: this.mapView,
            controls: defaultControls().extend(this.controls)
        });
        for (const interaction of this.interactions) {
            this.map.addInteraction(interaction);
        }
    }
 
    run() {
        if (this.wmtsLayer != null) {
            this.#createMap();
            var logmsg = FieldMap.logMsgPrefix + '.run => ';
            logmsg += 'map successfully created';
            var msg = new Messages.SendLogMsg();
            msg.logMsg = logmsg;
            this.appConn.send(msg);
        } else {
            setTimeout(this.run.bind(this), 10);
        }
    }

    static addFields(fieldMap, msg) {
        fieldMap.fldSource.clear(true);
        var features = [];
        for (let i = 0; i < msg.fieldNames.length; i++) {
            var corners = msg.cornerCoords[i];
            corners.push(corners[0]);
            var feat = new Feature(new Polygon([corners]));
            feat.set('fname', msg.fieldNames[i]);
            features.push(feat);
        }
        fieldMap.fldSource.addFeatures(features);
    }
}