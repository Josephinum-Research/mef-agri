from .task import Task


class Harvest(Task):
    """
    Inherits from :class:`mef_agri.farming.tasks.task.Task`.

    The harvest task indicates the end of the vegetation period in the model 
    evaluation part of **mef_agri**. Yield biomass is removed from the field 
    and :func:`residues_removed` indicates how much of the remaining biomass 
    (i.e. aboveground biomass minus yield biomass) is removed from the field 
    and not brought into the soil.

    There are no restrictions on the provided values in the applicaton map 
    layers (i.e. ``layer_ids`` property of 
    :class:`mef_agri.utils.raster.GeoRaster`). Anything can be added such as 
    achieved yield or moisture.
    """
    def __init__(self):
        super().__init__()
        self._meta['residues_removed'] = 0.0

    @property
    def residues_removed(self) -> float:
        """
        :return: fraction of crop residues which are removed from the field (e.g. through using straw) - if not set, the default value is zero (i.e. nothing is removed)
        :rtype: float
        """
        return self._meta['residues_removed']
    
    @residues_removed.setter
    def residues_removed(self, val):
        if isinstance(val, float) or isinstance(val, int):
            self._meta['residues_removed'] = float(val)
        else:
            msg = '`residues_removed` has to be a numeric/scalar value!'
            raise ValueError(msg)
