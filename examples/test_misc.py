from importlib import import_module


if __name__ == '__main__':
    module = import_module('mef_agri.models.crop.model_epic')
    for key, val in module.__dict__.items():
        print(key, val)
