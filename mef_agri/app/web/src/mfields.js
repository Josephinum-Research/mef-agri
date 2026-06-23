import { Messages } from './msgs';
import Control from 'ol/control/Control.js';
import Draw from 'ol/interaction/Draw.js';
import Select from 'ol/interaction/Select.js';
import Style from 'ol/style/Style.js';
import Fill from 'ol/style/Fill.js';
import Stroke from 'ol/style/Stroke.js';


export class ManipulateFields extends Control {
    constructor(fldSource, appConn) {
        ////////////////////////////////////////////////////////////////////////
        // UI-ELEMENTS ON MAP
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
        fnameInp.addEventListener('change', this.handleFnameInput.bind(this));

        this.uiElements = {
            addBtn: addBtn,
            finishBtn: finishBtn,
            undoBtn: undoBtn,
            cancelBtn: cancelBtn,
            fnameLbl: fnameLbl,
            fnameInp: fnameInp,
            ctrlArea: addCtrl
        };

        ////////////////////////////////////////////////////////////////////////
        // SELECT FIELDS
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

        ////////////////////////////////////////////////////////////////////////
        // DRAW FIELDS
        this.fldSource = fldSource;
        this.drawDef = new Draw({
            source: this.fldSource,
            type: 'Polygon'
        });
        this.drawDef.addEventListener(
            'drawstart', this.handleDrawStart.bind(this)
        );
        this.drawDef.addEventListener(
            'drawend', this.handleDrawEnd.bind(this)
        );

        ////////////////////////////////////////////////////////////////////////
        // OTHER STUFF
        document.addEventListener('keyup', this.handleKeysAddField.bind(this));
        document.addEventListener('keyup', this.handleDeleteField.bind(this));
        this.addFieldActive = false;  // flag to indicate if a field is currently created
        this.interactionRemoved = false // flag to indicate if map interaction has already been canceled (i.e. when user tries to draw a second field-polygon)
        this.newFeature = null;  // last added feature
        this.fName = null;  // value of input for field name
        this.appConn = appConn;
    }

    ////////////////////////////////////////////////////////////////////////////
    // HANDLERS FOR BUTTONS/KEYPRESSES
    handleAddField(event) {
        this.uiElements['fnameInp'].value = '';
        for (const [key, val] of Object.entries(this.uiElements)) {
            if (key == 'addBtn') {
                val.disabled = true;
            } else if (key == 'ctrlArea') {
                continue;
            } else {
                val.style.visibility = 'visible';
            }
        }
        this.addFieldActive = true;
        document.body.style.cursor = 'crosshair';
        this.getMap().removeInteraction(this.selectDef);
        this.interactionRemoved = false;
        this.selectedField = null;
        this.getMap().addInteraction(this.drawDef);
    }

    handleUndoPoint(event) {
        this.drawDef.removeLastPoint();
    }

    handleFinishField(event) {
        if (!this.fName) {
            this.uiElements.fnameInp.placeholder = 'enter fieldname!';
            return;
        }
        this.stopAddField();
        // set property of feature such that label is visible in map
        this.newFeature.set('fname', this.fName);
        // send field-data to frontend
        var msg = new Messages.SendDrawnField();
        msg.fieldName = this.fName;
        msg.cornerCoords = this.newFeature.getGeometry().getCoordinates();
        this.appConn.send(msg);
        // remainin stuff
        this.fName = null;
        this.uiElements.fnameInp.placeholder = '';
        this.newFeature = null;
    }

    handleCancelField(event) {
        this.stopAddField();
        if (this.newFeature != null) {
            this.fldSource.removeFeature(this.newFeature);
        }
        this.newFeature = null;
    }

    ////////////////////////////////////////////////////////////////////////////
    // HANDLERS FOR DRAW START/END
    handleDrawStart(event) {
        if (this.newFeature != null) {
            document.body.style.cursor = 'auto';
            this.getMap().removeInteraction(this.drawDef);
            this.interactionRemoved = true;
        }
    }

    handleDrawEnd(event) {
        this.newFeature = event.feature;
    }

    ////////////////////////////////////////////////////////////////////////////
    // COMMON METHOD WHEN DRAWING FIELDS HAS BEEN STOPPED
    stopAddField() {
        for (const [key, val] of Object.entries(this.uiElements)) {
            if (key == 'addBtn') {
                val.disabled = false;
            } else if (key == 'ctrlArea') {
                continue;
            } else {
                val.style.visibility = 'hidden';
            }
        }
        this.addFieldActive = false;
        if (!this.interactionRemoved) {
            document.body.style.cursor = 'auto';
            this.getMap().removeInteraction(this.drawDef);
        }
        this.getMap().addInteraction(this.selectDef);
    }

    ////////////////////////////////////////////////////////////////////////////
    // KEY-PRESS HANDLERS
    handleKeysAddField(event) {
        if (!this.addFieldActive) {
            return;
        }
        if (event.code == 'Enter') {
            this.handleFinishField();
        } else if (event.code == 'Escape') {
            this.handleCancelField();
        } else if (event.code == 'Backspace') {
            this.handleUndoPoint();
        }
    }

    handleDeleteField(event) {
        if (this.selectedField != null) {
            if (event.code == 'Delete') {
                var msg = new Messages.SendDeleteField();
                msg.fieldName = this.selectedField.get('fname');
                this.appConn.send(msg);
                this.fldSource.removeFeature(this.selectedField);
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // REMAINING METHODS
    selectMethod(feature) {
        const color = feature.get('COLOR') || 'rgba(255, 150, 150, 0.5)';
        this.selectedStyle.getFill().setColor(color);
        this.selectedField = feature;
        return this.selectedStyle
    }

    handleFnameInput(event) {
        this.fName = event.currentTarget.value;
    }

    toggle(flag) {
        if (flag) {
            this.uiElements.ctrlArea.style.visibility = 'visible';
            this.getMap().addInteraction(this.selectDef);
        } else {
            this.uiElements.ctrlArea.style.visibility = 'hidden';
            this.getMap().removeInteraction(this.selectDef);
            this.selectedField = null;
        }
    }
}