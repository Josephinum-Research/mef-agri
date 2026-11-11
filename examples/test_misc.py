import json

from mef_agri.evaluation.interfaces.mdef import ModelDefinition
from mef_agri.models.zone.model_jr import ZoneJR_V1
from mef_agri.models.crop.model_epic import Crop_Extended


if __name__ == '__main__':
    mdef = ModelDefinition(ZoneJR_V1)
    print(json.dumps(mdef.model_tree, indent=4))
