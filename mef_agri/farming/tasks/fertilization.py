from . import Task, Application
from ..fertilizers import Fertilizer
from ...models.utils import Units


class FertilizerAmount(Application):
    def __init__(self):
        super().__init__()
        self._amnt = Application.NumericValue()
        self._amnt.name = 'amount'
        self._amnt.description = """
amount of applied mineral fertilizer in mass per area
        """
        self._amnt.valid_units = (
            Units.t_ha, Units.kg_ha, Units.kg_m2, Units.g_m2
        )
        self._fname = Application.DescriptiveValue()
        self._fname.name = 'fertilizer'
        self._fname.description = """
name of applied mineral fertilizer
        """
        self._fparams = Application.DescriptiveValue()
        self._fparams.name = 'parameters'
        self._fparams.description = """
parameters of applied mineral fertilizer (i.e. nutrient fractions) provided as 
dict or json-string with the structure from 
`mef_agri.farming.fertilizers.Fertilizer.get_dict_repr()`
        """

    @property
    def name(self):
        return 'fertilizer-amount'
    
    @property
    def amount(self) -> Application.NumericValue:
        """
        :return: definition of fertilizer amount
        :rtype: Application.NumericValue
        """
        return self._amnt
    
    @property
    def fertilizer_name(self) -> Application.DescriptiveValue:
        """
        :return: name of applied mineral fertilizer
        :rtype: Application.DescriptiveValue
        """
        return self._fname
    
    @property
    def fertilizer_params(self) -> Application.DescriptiveValue:
        """
        :return: fertilizer parameters (i.e. nutrient fractions)
        :rtype: Application.DescriptiveValue
        """
        return self._fparams
    
    @classmethod
    def from_fertilizer_obj(cls, fert:Fertilizer):
        """
        Create :class:`FertilizerAmount` application from provided fertilizer 
        ``fert`` by setting descriptive values :func:`fertilizer_name` and 
        :func:`fertilizer_params`.

        :param fert: fertilizer object inheriting from :class:`mef_agri.farming.fertilizers.Fertilizer`
        :type fert: Fertilizer
        """
        obj = cls()
        obj.fertilizer_name.value = fert.name
        obj.fertilizer_params.value = fert.get_dict_repr()


class MineralFertilization(Task):
    @property
    def valid_applications(self):
        return [FertilizerAmount,]
