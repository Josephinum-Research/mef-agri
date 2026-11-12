import json

#from mef_agri.evaluation.interfaces.mdef import ModelDefinition
#from mef_agri.models.zone.model_jr import ZoneJR_V1
#from mef_agri.models.crop.model_epic import Crop_Extended
from mef_agri.data.geosphere_austria.inca.inca import INCALABELS


if __name__ == '__main__':
    #mdef = ModelDefinition(ZoneJR_V1)
    #print(json.dumps(mdef.model_tree, indent=4))
    for key, val in INCALABELS.__dict__.items():
        print(key, val)
