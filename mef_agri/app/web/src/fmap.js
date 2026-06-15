import TileLayer from "ol/layer/Tile";
import View from "ol/View";

export class FieldMap {
    constructor() {
        useGeographic();
        this.wmtsParser = new WMTSCapabilities();
        this.wmtsUrl = 'https://mapsneu.wien.gv.at/basemapneu/1.0.0/WMTSCapabilities.xml'
        this.wmtsLayer = 'bmaporthofoto30cm';
        this.wmtsEpsg = 'EPSG:4326';
        this.wmtsSource = null;
        this.wmtsLayer = null;
        this.fldSource = new VectorSource({wrapX: false});
        this.fldLayer = new VectorLayer({source: this.fldSource});
        this.initPos = [15.1445, 48.1328];
        this.initZoom = 14;
        this.map = null;
        this.controls = [];
        this.interactions = [];
    }

    initializeWMTS() {
        const wmtsSettings = {
            layer: this.wmtsLayer,
            matrixSet: this.wmtsEpsg
        };
        fetch(this.imgUrl)
            .then(function (response) {
                return response.text();
            })
            .then(function (text) {
                const result = this.wmtsParser.read(text);
                const options = optionsFromCapabilities(result, wmtsSettings);
                this.wmtsSource = new WMTS(options);
                this.wmtsLayer = new TileLayer({source: this.wmtsSource});
            });
    }

    addControl(control) {
        this.controls.push(control);
    }

    addInteraction(interaction) {
        this.interactions.push(interaction);
    }
 
    run() {
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
        for (interaction in this.interactions) {
            this.map.addInteraction(interaction);
        }
    }
}