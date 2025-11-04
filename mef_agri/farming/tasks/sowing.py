from inspect import isclass

from .task import Task
from .. import crops
from ...models.utils import Units


class Sowing(Task):
    """
    Inherits from :class:`mef_agri.farming.tasks.task.Task`.

    The sowing task indicates the beginning of vegetation periods in the model 
    evaluation part of **mef_agri**. Thus, it exhibits specific properties and 
    constraints on the provided application information.
    In the :func:`application_map`, there needs to be at least one layer which 
    is named :func:`avname_sowing_density` (being ``'sowing_density'``) or 
    :func:`avname_sowing_amount` (being ``'sowing_amount'``). 
    If ``'sowing_density'`` is provided it is also necessary to provide the 
    thousand grain weight (:func:`tgw`) to derive the initial amount of biomass.
    Additionally, it is also necessary to provide the sown crop and cultivar (
    settable properties :func:`crop` and :func:`cultivar`).
    It is recommended to use classes from the ``mef_agri.farming.crops`` module 
    which hold the cultivars for this.
    This class only supports one sown crop/cultivar (e.g. catch crop mixtures 
    are not supported yet).
    If there is cultivar information appended (i.e. non-empty dictionaries which 
    are returned from the methods of crop-classes decorated with ``@cultivar`` 
    in ``mef_agri.farming.crop``), it will be added to the :func:`metadata` with 
    key ``'cultivar_info'``.
    """
    def __init__(self):
        super().__init__()
        self._meta['crop'] = None
        self._meta['cultivar'] = None
        self._meta['cultivar_info'] = None
        self._meta['crop_module'] = None
        self._meta['tgw'] = None
        self._meta['tgw_unit'] = None

        self._av_sd:str = 'sowing_density'
        self._av_sa:str = 'sowing_amount'

    @property
    def avname_sowing_density(self) -> str:
        """
        :return: required name of application value (i.e. layer id of the ``GeoRaster``) if it equals the sowing density (amount of seeds per area unit). In this case it is necessary to provide the thousand grain weight (:func:`tgw`)
        :rtype: str
        """
        return self._av_sd
    
    @property
    def avname_sowing_amount(self) -> str:
        """
        :return: required name of application value (i.e. layer id of the ``GeoRaster``) if it equals the sowing amount (mass of seeds per area unit)
        :rtype: str
        """
        return self._av_sa
    
    @property
    def crop_module(self) -> str:
        """
        :return: module containing the class representing :func:`crop`
        :rtype: str
        """
        return self._meta['crop_module']

    @property
    def crop(self) -> str:
        """
        :return: name of sown crop (i.e. names from classes in module ``mef_agri.farming.crops``)
        :rtype: str
        """
        return self._meta['crop']
    
    @crop.setter
    def crop(self, val):
        if isinstance(val, str):
            if not hasattr(crops, val):
                msg = 'Provided string does not match any crop-class-name from '
                msg += '`mef_agri.farming.crops`!'
                raise ValueError(msg)
            val = getattr(crops, val)

        if isclass(val):
            cn = val.__name__
            val = val()
        else:
            cn = val.__class__.__name__
        if not isinstance(val, crops.Crop):
            msg = 'Use crop definitions from `mef_agri.farming.crops` for '
            msg += 'setting information in `Sowing`-Task!'
            raise ValueError(msg)
        self._meta['crop_module'] = val.__module__
        self._meta['crop'] = cn

    @property
    def cultivar(self) -> str:
        """
        :return: name of sown cultivar (i.e. names of methods from crop classes decorated with ``@cultivar`` in module ``mef_agri.farming.crops``)
        :rtype: str
        """
        return self._meta['cultivar']
    
    @cultivar.setter
    def cultivar(self, val):
        if callable(val):
            cn, val = val.__qualname__.split('.')
            if not hasattr(crops, cn):
                msg = 'Provided value is not a member of a crop-class from '
                msg += '`mef_agri.farming.crops`!'
                raise ValueError(msg)
            self.crop = cn

        if isinstance(val, str):
            if self.crop is None:
                msg = 'If provided cultivar variable is a string, '
                msg += '`Sowing.crop` has to be set first!'
                raise ValueError(msg)
            crop = getattr(crops, self.crop)()
            vinfo = getattr(crop, val)()
            if not val in crop.cultivars:
                msg = 'Provided cultivar name is not available in '
                msg += 'mef_agri.farming.crops.{}!'.format(self.crop)
                raise ValueError(msg)
            self._meta['cultivar'] = val
            if vinfo:
                self._meta['cultivar_info'] = vinfo
            return

        msg = 'Provided value for `cultivar` has to be a string or a method '
        msg += 'from a `mef_agri.farming.crops.Crop` child-class being '
        msg += 'decorated with `mef_agri.farming.crops.cultivar`!'
        raise ValueError(msg)
    
    @property
    def tgw(self) -> float:
        """
        :return: thousand grain weight of seeds
        :rtype: float
        """
        return self._meta['tgw']
    
    @tgw.setter
    def tgw(self, val):
        if not isinstance(val, float) or isinstance(val, int):
            msg = 'thousand grain weight `tgw` has to be a numeric value!'
            raise ValueError(msg)
        self._meta['tgw'] = val
    
    @property
    def tgw_unit(self) -> str:
        """
        :return: units of :func:`tgw` (see :class:`mef_agri.models.utils.Units`)
        :rtype: str
        """
        return self._meta['tgw_unit']
    
    @tgw_unit.setter
    def tgw_unit(self, val):
        if not val in (Units.g, Units.kg, Units.t):
            msg = 'Provided unit has to be gram, kilogram or ton (see '
            msg += '`mef_agri.models.utils.Units`)!'
            raise ValueError(msg)
        self._meta['tgw_unit'] = val

    
    def specify_application(
            self, appl_val, appl_name, appl_unit, appl_ix=None
        ):
        super().specify_application(
            appl_val, appl_name, appl_unit, appl_ix
        )
        check1 = self.avname_sowing_amount in self.layer_ids
        check2 = self.avname_sowing_density in self.layer_ids
        if not (check1 or check2):
            msg = 'At least one application needs to be `sowing_density` ('
            msg += 'seeds per unit area) or `sowing_amount` (mass per unit '
            msg += 'area)!'
            raise ValueError(msg)
        if check1:
            validu = (
                Units.g_ha, Units.kg_ha, Units.t_ha, Units.g_m2, Units.kg_m2
            )
            au = self.layer_infos['units'][self.avname_sowing_amount]
            if not (au in validu):
                msg = 'Non valid unit for `sowing_amount` - has to be mass per'
                msg += 'unit area (see `mef_agri.models.utils.Units`)!'
                raise ValueError(msg)
        if check2:
            validu = (Units.n_m2, Units.n_ha)
            au = self.layer_infos[self.avname_sowing_density]['unit']
            if not (au in validu):
                msg = 'Non valid unit for `sowing_density` - has to be number '
                msg += 'per unit area (see `mef_agri.models.utils.Units`)!'
                raise ValueError(msg)
            if None in (self.tgw, self.tgw_unit):
                msg = 'When providing `sowing_density`, the properties `tgw` '
                msg += 'and `tgw_units` have to be provided!'