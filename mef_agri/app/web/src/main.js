import './style.css';
import { AppConnection } from './conn.js';
import { Messages } from './msgs.js';
import { FieldMap } from './fmap.js';
import { ManipulateFields } from './mfields.js';
import { SelectField } from './sfields.js';


function handle_tab_change(_, msg) {
    if (msg.activeTab == 'project') {
        mFields.toggle(true);
        sFields.toggle(false);
    } else if (msg.activeTab == 'data') {
        mFields.toggle(false);
        sFields.toggle(false);
    } else if (msg.activeTab == 'tasks') {
        mFields.toggle(false);
        sFields.toggle(true);
    }
}


const appConn = new AppConnection();
const fMap = new FieldMap(appConn);
appConn.registerHandler(Messages.GotFieldInfo, FieldMap.addFields, fMap);
appConn.registerHandler(Messages.GotActiveTab, handle_tab_change);
fMap.initializeWMTS();
const mFields = new ManipulateFields(fMap.fldSource, appConn);
fMap.addCustomControl(mFields);
fMap.addCustomInteraction(mFields.selectDef);
const sFields = new SelectField(fMap, appConn);
fMap.addCustomInteraction(sFields.selectDef);
fMap.run();
