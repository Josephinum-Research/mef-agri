import './style.css'
import Map from 'ol/Map.js';
import WMTS, {optionsFromCapabilities} from 'ol/source/WMTS.js';
import WMTSCapabilities from 'ol/format/WMTSCapabilities.js';
import TileLayer from 'ol/layer/Tile.js';
import View from 'ol/View.js'
import { useGeographic } from 'ol/proj';

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
            view: new View({center: [15.1445, 48.1328], zoom:14})
        }
        map = new Map(mapSettings);
    });
