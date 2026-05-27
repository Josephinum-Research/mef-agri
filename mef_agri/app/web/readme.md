# Guide to build openlayers webapp in manjaro

- install nvm

`pamac install nvm`

- locate the the directory where init-nvm.sh is located (probably `/usr/share/nvm/`) and source it

`source /usr/share/nvm/init-nvm.sh`

- this sets everything up, such that it is possible to use nvm in the terminal. close the current terminal, open a new one and type

`source .nvm/nvm.sh`

- install node

`nvm install node`

- next install webpack and dependencies with

`npm install webpack webpack-cli --save-dev`

`npm install style-loader css-loader html-webpack-plugin --save-dev`

- navigate to `mef-agri/mef_agri/app/web` and do (should be already available)

`npm install ol`

`npm install proj4`

- if changes are present in `mef-agri/mef_agri/app/web/src/`, it is necessary to bundle this. Therefore perform the following comand in the `mef-agri/mef_agri/app/web/` directory

`npm run build`

- bundled webapp is available `mef-agri/mef_agri/app/web/dist`