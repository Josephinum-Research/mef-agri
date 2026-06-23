import './style.css';
import { AppConnection } from './conn.js';
import { Messages } from './msgs.js';
import { FieldMap } from './fmap.js';
import { ManipulateFields } from './mfields.js';


function handle_tab_change(_, msg) {
    if (msg.activeTab == 'project') {
        mFields.toggle(true);
    } else if (msg.activeTab == 'data') {
        mFields.toggle(false);
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
fMap.run();
