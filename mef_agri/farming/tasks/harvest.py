from numpy import ndarray

from . import Task, Application
from ...models.utils import Units


class CYield(Application):
    def __init__(self):
        super().__init__()
        self._cyld = Application.NumericValue()
        self._cyld.name = 'yield'
        self._cyld.description = """
crop yield (mass per area)
        """
        self._cyld.valid_units = [Units.t_ha, Units.kg_ha]
        self._resr = Application.NumericValue()
        self._resr.name = 'residues-removed'
        self._resr.description = """
fraction of above-ground biomass which is removed from the field after harvest
        """
        self._resr.valid_units = [Units.frac,]

    @property
    def name(self):
        return 'crop-yield'
    
    @property
    def crop_yield(self) -> Application.NumericValue:
        """
        :return: definition of crop yield
        :rtype: Application.NumericValue
        """
        return self._cyld
    
    @property
    def residues_removed(self) -> Application.NumericValue:
        """
        :return: definition of removed residues after harvest
        :rtype: Application.NumericValue
        """
        return self._resr
    

class Harvest(Task):
    @property
    def valid_applications(self):
        return [CYield,]
