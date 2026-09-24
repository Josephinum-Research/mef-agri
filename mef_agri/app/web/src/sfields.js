import { Messages } from './msgs';
import Style from 'ol/style/Style.js';
import Fill from 'ol/style/Fill.js';
import Stroke from 'ol/style/Stroke.js';
import Select from 'ol/interaction/Select.js';


export class SelectField {
    constructor(fMap, appConn) {
        this.selectedField = null;
        this.selectedStyle = new Style({
            fill: new Fill({
                color: 'rgba(255, 150, 150, 0.5)'
            }),
            stroke: new Stroke({
                color: 'rgba(255, 0, 0, 1.0)'  // rgb+opacity
            })
        });
        this.selectDef = new Select({style: this.selectMethod.bind(this)});
        this.appConn = appConn;
        this.fMap = fMap;
        this.suppress_selection = false;
    }

    selectMethod(feature) {
        // style stuff
        const color = feature.get('COLOR') || 'rgba(255, 150, 150, 0.5)';
        this.selectedStyle.getFill().setColor(color);

        var c1 = feature != this.selectedField;
        var c2 = !this.suppress_selection;
        if (c1 && c2) {
            // zoom to feature
            this.selectedField = feature;
            var extent = feature.getGeometry().getExtent();
            this.fMap.map.getView().fit(extent);
            // send message with feature/field name
            var msg = new Messages.SendSelectedField();
            msg.fieldName = feature.get('fname');
            this.appConn.send(msg);
        }
        return this.selectedStyle;
    }

    toggle(flag) {
        if (flag) {
            this.fMap.map.addInteraction(this.selectDef);
        } else {
            this.fMap.map.removeInteraction(this.selectDef);
            this.selectedField = null;
        }
    }

    static handle_tasks_tree_changes(selectField, flag) {
        selectField.suppress_selection = flag;
    }
}