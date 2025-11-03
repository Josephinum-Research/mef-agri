from inspect import isclass

from .task import Task
from .. import fertilizers
from ...models.utils import Units


class MineralFertilization(Task):
    """
    Inherits from :class:`mef_agri.farming.tasks.task.Task`.

    In the :func:`application_map`, there needs to be at least one layer which 
    is named :func:`avname_fertilizer_amount` (being ``'fertilizer_amount'``).
    The :func:`fertilizer` property is settable and one can provide a string 
    which equals one of the fertilizers in ``mef_agri.farming.fertilizers`` or 
    directly provide one of these classes (instances are also supported).
    """
    def __init__(self):
        super().__init__()
        self._meta['fertilizer'] = None
        self._meta['fertilizer_module'] = None
        self._av_fa:str = 'fertilizer_amount'

    @property
    def avname_fertilizer_amount(self) -> str:
        """
        :return: required name of application value (i.e. layer id of the ``GeoRaster``) if it equals the applied amount of fertilizer
        :rtype: str
        """
        return self._av_fa

    @property
    def fertilizer(self) -> str:
        """
        :return: fertilizer name, i.e. the name of the corresponding class in ``mef_agri.farming.fertilizers`` (settable)
        :rtype: str
        """
        return self._meta['fertilizer']
    
    @fertilizer.setter
    def fertilizer(self, val):
        if isclass(val):
            fname = val.__name__
            fmod = val.__module__
        elif isinstance(val, str):
            fname = val
            fmod = None
        else:
            fname = val.__class__.__name__
            fmod = val.__class__.__name__
        if not hasattr(fertilizers, fname) or fname == 'Fertilizer':
            msg = 'Provided fertilizer (i.e. its name) is not available in '
            msg += '`mef_agri.farming.fertilizers`!'
            raise ValueError(msg)
        
        if fmod is None:
            fmod = getattr(fertilizers, fname).__class__.__module__
        self._meta['fertilizer'] = fname
        self._meta['fertilizer_module'] = fmod

    def specify_application(self, appl_val, appl_name, appl_unit, appl_ix=None):
        super().specify_application(appl_val, appl_name, appl_unit, appl_ix)
        
        if not self.avname_fertilizer_amount in self.layer_ids:
            msg = 'At least one application needs to be `fertilizer_amount`!'
            raise ValueError(msg)
        validu = (Units.kg_ha, Units.t_ha, Units.kg_m2, Units.g_m2)
        au = self.layer_infos['units'][self.avname_fertilizer_amount]
        if not au in validu:
            msg = 'Non valid unit for `fertilizer_amount` - has to be mass per '
            msg += 'unit area (see `mef_agri.models.utils.Units`)!'
            raise ValueError(msg)
