from inspect import isclass

from ...models.base import Model
from . import Interface


class ModelDefinition(object):
    def __init__(self, zone_model:Model):
        if isclass(zone_model):
            zone_model = zone_model()
        self._defs:dict = {}
        for mid, minfo in zone_model.model_tree.tree_dict.items():
            # TODO
            ...
