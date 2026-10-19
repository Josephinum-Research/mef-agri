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
import { extend as extendExtent } from "ol/extent";


class LayerStyle {
    constructor() {
        this.strokeColor = 'rgb(0, 0, 0)';
        this.strokeWidth = 1;
        this.fillColor = 'rgba(0, 0, 0, 0.25)';
        this.textAlign = 'center';
        this.textOverflow = true;
        this.textFont = '12px Calibri,sans-serif';
        this.textFillColor = 'rgb(0, 0, 0)';
        this.textStrokeColor = 'rgb(255, 255, 255)';
        this.textStrokeWidth = 1;
        this.textVisible = true;
    }

    getStyle(feature, attr) {
        var textStyle = null;
        if (this.textVisible) {
            textStyle = {
                textAlign: this.textAlign,
                overflow: this.textOverflow,
                font: this.textFont,
                stroke: new Stroke({
                    color: this.textStrokeColor,
                    width: this.textStrokeWidth
                }),
                fill: new Fill({
                    color: this.textFillColor
                }),
                text: feature.get(attr)
            };
        } else {
            textStyle = {};
        }
        return new Style({
            stroke: new Stroke({
                color: this.strokeColor,
                width: this.strokeWidth
            }),
            fill: new Fill({
                color: this.fillColor
            }),
            text: new Text(textStyle)
        });
    }
}


export class FieldLayerStyle extends LayerStyle {
    constructor() {
        super();
        this.strokeColor = 'rgb(0, 255, 150)';
        this.strokeWidth = 4;
        this.fillColor = 'rgba(0, 255, 150, 0.25)';
        this.textFont = '16px Calibri,sans-serif';
        this.textStrokeWidth = 2;
    }
}


export class ZoneLayerStyle extends LayerStyle {
    constructor() {
        super();
        this.strokeColor = 'rgb(100, 5, 240)';
        this.strokeWidth = 2;
        this.fillColor = 'rgba(100, 5, 240, 0.25)';
        this.textFont = '16px Calibri,sans-serif';
        this.textStrokeWidth = 2;
    }
}


export class MiscPolygonLayerStyle extends LayerStyle {
    constructor() {
        super();
        this.strokeColor = 'rgb(245, 155, 40)';
        this.strokeWidth = 2;
        this.fillColor = 'rgba(245, 155, 40, 0.25)';
        this.textVisible = false;
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
            style: this.#getFldStyle.bind(this),
        });

        ////////////////////////////////////////////////////////////////////////
        // MISC polygon-layer
        this.miscStyleOptions = new MiscPolygonLayerStyle();
        this.miscSource = new VectorSource({wrapX: false});
        this.miscLayer = new VectorLayer({
            source: this.miscSource,
            style: this.#getMiscStyle.bind(this)
        })

        ////////////////////////////////////////////////////////////////////////
        // ZONES vector-layer
        this.znStyleOptions = new ZoneLayerStyle();
        this.znSource = new VectorSource({wrapX: false});
        this.znLayer = new VectorLayer({
            source: this.znSource,
            style: this.#getZnStyle.bind(this)
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

    #getFldStyle(feature) {
        return this.fldStyleOptions.getStyle(feature, 'fname');
    }

    #getZnStyle(feature) {
        return this.znStyleOptions.getStyle(feature, 'zname');
    }

    #getMiscStyle(feature) {
        return this.miscStyleOptions.getStyle(feature, null);
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

    #createMap() {
        this.mapView = new View({
            center: this.initPos,
            zoom: this.initZoom
        })
        this.map = new Map({
            target: 'map',
            //projection: new Projection('EPSG:' + this.mapEpsg.toString()),
            layers: [
                this.osmLayer, this.wmtsLayer, this.fldLayer, this.miscLayer,
                this.znLayer
            ],
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
        if (fieldMap.map != null) {
            fieldMap.fldSource.clear(true);
            var features = [];
            var extent = null;
            for (let i = 0; i < msg.fieldNames.length; i++) {
                var corners = msg.cornerCoords[i];
                corners.push(corners[0]);
                var feat = new Feature(new Polygon([corners]));
                feat.set('fname', msg.fieldNames[i]);
                features.push(feat);
                if (extent == null) {
                    extent = feat.getGeometry().getExtent();
                } else {
                    extent = extendExtent(
                        extent, feat.getGeometry().getExtent()
                    );
                }
            }
            fieldMap.fldSource.addFeatures(features);
            fieldMap.map.getView().fit(extent);
        } else {
            setTimeout(FieldMap.addFields(fieldMap, msg), 10);
        }
    }

    static addZones(fieldMap, msg) {
        fieldMap.znSource.clear(true);
        var features = [];
        for (let i = 0; i < msg.zoneNames.length; i++) {
            var corners = msg.cornerCoords[i];
            corners.push(corners[0]);
            var feat = new Feature(new Polygon([corners]));
            feat.set('zname', msg.zoneNames[i])
            features.push(feat);
        }
        fieldMap.znSource.addFeatures(features);
    }

    static addMiscPolygons(fieldMap, msg) {
        fieldMap.miscSource.clear(true);
        var features = [];
        for (let i = 0; i < msg.plgnIds.length; i++) {
            var corners = msg.cornerCoords[i];
            corners.push(corners[0]);
            var feat = new Feature(new Polygon([corners]));
            features.push(feat);
        }
        fieldMap.miscSource.addFeatures(features);
    }

    static clearSource(fieldMap, msg) {
        if (msg.mapSource == 'fields') {
            fieldMap.fldSource.clear(true);
        } else if (msg.mapSource == 'zones') {
            fieldMap.znSource.clear(true);
        } else if (msg.mapSource == 'misc') {
            fieldMap.miscSource.clear(true);
        }
    }
}