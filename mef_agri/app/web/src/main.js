import './style.css'
import Map from 'ol/Map.js';
import OSM from 'ol/source/OSM.js';
import TileLayer from 'ol/layer/Tile.js'
import View from 'ol/View.js'
import { useGeographic } from 'ol/proj';

useGeographic();

const map = new Map(
    {
        target: 'map',
        layers: [new TileLayer({source: new OSM()})],
        view: new View({center: [15.1445, 48.1328], zoom:16})
    }
);