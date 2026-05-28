import './style.css'
import Map from 'ol/Map.js';
import WMTS, {optionsFromCapabilities} from 'ol/source/WMTS.js';
import WMTSCapabilities from 'ol/format/WMTSCapabilities.js';
import TileLayer from 'ol/layer/Tile.js';
import View from 'ol/View.js'
import { useGeographic } from 'ol/proj';
import Control from 'ol/control/Control.js';
import { defaults as defaultControls } from 'ol/control/defaults.js';


class AddField extends Control {
    constructor() {
        const addBtn = document.createElement('button');
        addBtn.id = 'add-field-button';
        addBtn.innerHTML = 'add field';

        const finishBtn = document.createElement('button');
        finishBtn.id = 'finish-field-button';
        finishBtn.innerHTML = 'finish (Enter)';

        const redoBtn = document.createElement('button');
        redoBtn.id = 'redo-point-button';
        redoBtn.innerHTML = 'redo (right mouse)';

        const cancelBtn = document.createElement('button');
        cancelBtn.id = 'cancel-field-button';
        cancelBtn.innerHTML = 'cancel (Escape)';

        const fnameLbl = document.createElement('label');
        fnameLbl.id = 'field-name-label';
        fnameLbl.innerHTML = 'field-name:';

        const fnameInp = document.createElement('input');
        fnameInp.type = 'text';
        fnameInp.id = 'field-name-input';
        fnameInp.ed

        const addCtrl = document.createElement('div');
        addCtrl.id = 'add-field-control';
        addCtrl.appendChild(addBtn);
        addCtrl.appendChild(finishBtn);
        addCtrl.appendChild(redoBtn);
        addCtrl.appendChild(cancelBtn);
        addCtrl.appendChild(fnameLbl);
        addCtrl.appendChild(fnameInp);

        super({element: addCtrl});
        addBtn.addEventListener('click', this.handleAddField.bind(this), false);
    }

    handleAddField() {}
}


useGeographic();

const parser = new WMTSCapabilities();
let map;

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
        const mapSettings = {
            target: 'map',
            layers: [layer1],
            view: new View({center: [15.1445, 48.1328], zoom:14}),
            controls: defaultControls().extend([new AddField()])
        }
        map = new Map(mapSettings);
    });
