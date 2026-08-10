import json
from numpy import ndarray

from . import Task, Application
from ...models.utils import Units


class SowingApplication(Application):
    def __init__(self):
        super().__init__()
        self._crop = Application.DescriptiveValue()
        self._crop.name = 'crop'
        self._crop.description = """
name of the sown crop
        """
        self._cult = Application.DescriptiveValue()
        self._cult.name = 'cultivar'
        self._cult.description = """
name of the sown cultivar
        """
        self._cparams = Application.DescriptiveValue()
        self._cparams.name = 'parameters'
        self._cparams.description = """
parameters which further describe the sown cultivar (i.e. a dict or json-string 
containing appropriate information being used in the model evaluation e.g. as 
initial states or parameters)
        """

    @property
    def crop(self) -> Application.DescriptiveValue:
        """
        :return: sown crop
        :rtype: Application.DescriptiveValue
        """
        return self._crop
    
    @property
    def cultivar(self) -> Application.DescriptiveValue:
        """
        :return: sown cultivar
        :rtype: Application.DescriptiveValue
        """
        return self._cult

    @property
    def parameters(self) -> Application.DescriptiveValue:
        """
        :return: cultivar parameters which will be used to adjust initial values of parameters in evaluation
        :rtype: Application.DescriptiveValue
        """
        return self._cparams


class SowingAmount(SowingApplication):
    def __init__(self):
        super().__init__()
        self._amnt = Application.NumericValue()
        self._amnt.name = 'amount'
        self._amnt.description = """
sowing amount in (bio)mass per area
        """
        self._amnt.valid_units = (
            Units.kg_ha, Units.t_ha, Units.kg_m2, Units.g_m2
        )

    @property
    def name(self):
        return 'amount-of -> ' + self.cultivar.value + '-' + self.crop.value
    
    @property
    def amount(self) -> Application.NumericValue:
        """
        :return: definition of sowing amount
        :rtype: Application.NumericValue
        """
        return self._amnt


class SowingDensity(SowingApplication):
    def __init__(self):
        super().__init__()
        self._dens = Application.NumericValue()
        self._dens.name = 'density'
        self._dens.description = """
sowing density in plants per area
        """
        self._dens.valid_units = (Units.n_m2, Units.n_ha)
        self._tgw = Application.NumericValue()
        self._tgw.name = 'thousand-grain-weight'
        self._tgw.valid_units = (Units.g, Units.kg)

    @property
    def name(self):
        return 'density-of -> ' + self.cultivar.value + '-' + self.crop.value
    
    @property
    def density(self) -> Application.NumericValue:
        """
        :return: definition of sowing density
        :rtype: Application.NumericValue
        """
        return self._dens
    
    @property
    def tgw(self) -> Application.NumericValue:
        """
        :return: definition of thousand grain weight
        :rtype: Application.NumericValue
        """
        return self._tgw


class Sowing(Task):
    @property
    def valid_applications(self):
        return [SowingAmount, SowingDensity]
