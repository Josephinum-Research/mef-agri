import './style.css'
import Map from 'ol/Map.js';
import WMTS, {optionsFromCapabilities} from 'ol/source/WMTS.js';
import WMTSCapabilities from 'ol/format/WMTSCapabilities.js';
import TileLayer from 'ol/layer/Tile.js';
import View from 'ol/View.js'
import { useGeographic } from 'ol/proj';
import Control from 'ol/control/Control.js';
import { defaults as defaultControls } from 'ol/control/defaults.js';
import Draw from 'ol/interaction/Draw.js';
import VectorSource from 'ol/source/Vector';
import VectorLayer from 'ol/layer/Vector';


class AddField extends Control {
    constructor() {
        const addBtn = document.createElement('button');
        addBtn.id = 'add-field-button';
        addBtn.innerHTML = 'add field';

        const finishBtn = document.createElement('button');
        finishBtn.id = 'finish-field-button';
        finishBtn.innerHTML = 'finish (Enter)';
        finishBtn.style.visibility = 'hidden';

        const undoBtn = document.createElement('button');
        undoBtn.id = 'redo-point-button';
        undoBtn.innerHTML = 'undo (Backspace)';
        undoBtn.style.visibility = 'hidden';

        const cancelBtn = document.createElement('button');
        cancelBtn.id = 'cancel-field-button';
        cancelBtn.innerHTML = 'cancel (Escape)';
        cancelBtn.style.visibility = 'hidden';

        const fnameLbl = document.createElement('label');
        fnameLbl.id = 'field-name-label';
        fnameLbl.innerHTML = 'field-name:';
        fnameLbl.style.visibility = 'hidden';

        const fnameInp = document.createElement('input');
        fnameInp.type = 'text';
        fnameInp.id = 'field-name-input';
        fnameInp.style.visibility = 'hidden';

        const addCtrl = document.createElement('div');
        addCtrl.id = 'add-field-control';
        addCtrl.appendChild(addBtn);
        addCtrl.appendChild(finishBtn);
        addCtrl.appendChild(undoBtn);
        addCtrl.appendChild(cancelBtn);
        addCtrl.appendChild(fnameLbl);
        addCtrl.appendChild(fnameInp);

        super({element: addCtrl});

        addBtn.addEventListener('click', this.handleAddField.bind(this));
        finishBtn.addEventListener('click', this.handleFinishField.bind(this));
        cancelBtn.addEventListener('click', this.handleCancelField.bind(this));
        undoBtn.addEventListener('click', this.handleUndoPoint.bind(this));
        document.addEventListener('keyup', this.handleKeyUp.bind(this));

        this.uiElements = {
            addBtn: addBtn,
            finishBtn: finishBtn,
            undoBtn: undoBtn,
            cancelBtn: cancelBtn,
            fnameLbl: fnameLbl,
            fnameInp: fnameInp
        };

        this.addFieldActive = false;  // flag to indicate if a field is currently created
        this.newFeature = null;  // last added feature
    }

    handleKeyUp(event) {
        if (!this.addFieldActive) {
            return;
        }
        if (event.code == 'Enter') {
            this.#stopAddField();
        } else if (event.code == 'Escape') {
            this.#stopAddField();
        } else if (event.code == 'Backspace') {
            this.#undoDrawnPoint();
        }
    }

    handleAddField(event) {
        this.uiElements['fnameInp'].value = '';
        for (const [key, val] of Object.entries(this.uiElements)) {
            if (key == 'addBtn') {
                val.disabled = true;
            } else {
                val.style.visibility = 'visible';
            }
        }
        this.addFieldActive = true;
        document.body.style.cursor = 'crosshair';
        this.nFeaturesBefore = fldSource.getFeatures().length;
        
        // enable drawing of polygon
        draw = new Draw({
            source: fldSource,
            type: 'Polygon'
        });
        draw.addEventListener('drawend', this.handleDrawEnd.bind(this));
        this.getMap().addInteraction(draw);
    }

    handleDrawEnd(event) {
        appConn.sendPolygon(event.feature);
        //if (this.newFeature != null) {
        //    fldSource.removeFeature(event.feature);
        //    return;
        //}
        //this.newFeature = event.feature;
    }

    handleUndoPoint(event) {
        this.#undoDrawnPoint();
    }

    handleFinishField(event) {
        this.#stopAddField();
        // TODO send field data to gpkg
        //this.newFeature = null;
    }

    handleCancelField(event) {
        this.#stopAddField();
        //if (this.newFeature != null) {
        //    fldSource.removeFeature(this.newFeature);
        //}
        //this.newFeature = null;
    }

    #undoDrawnPoint() {
        draw.removeLastPoint();
    }

    #stopAddField() {
        for (const [key, val] of Object.entries(this.uiElements)) {
            if (key == 'addBtn') {
                val.disabled = false;
            } else {
                val.style.visibility = 'hidden';
            }
        }
        this.addFieldActive = false;
        document.body.style.cursor = 'auto';
        this.getMap().removeInteraction(draw);
    }
}

class AppConnection {
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

    sendPolygon(feat) {
        this.ws.send(JSON.stringify({method: 'GET'}));
    }
}

////////////////////////////////////////////////////////////////////////////////
///   SCRIPT STARTS HERE
////////////////////////////////////////////////////////////////////////////////

useGeographic();

const parser = new WMTSCapabilities();
const fldSource = new VectorSource({wrapX: false});
const appConn = new AppConnection();
let map;
let draw;

fetch('https://mapsneu.wien.gv.at/basemapneu/1.0.0/WMTSCapabilities.xml')
    .then(function (response) {
        return response.text();
    })
    .then(function (text) {
        const result = parser.read(text);
        const wmtsSettings = {
            layer: 'bmaporthofoto30cm',
            matrixSet: 'EPSG:4326',
        };
        const options = optionsFromCapabilities(result, wmtsSettings);
        const layer1 = new TileLayer({source: new WMTS(options)});
        const layer2 = new VectorLayer({source: fldSource});
        const mapSettings = {
            target: 'map',
            layers: [layer1, layer2],
            view: new View({center: [15.1445, 48.1328], zoom:14}),
            controls: defaultControls().extend([new AddField()])
        }
        map = new Map(mapSettings);
    });
