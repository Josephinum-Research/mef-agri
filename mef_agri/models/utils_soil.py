import numpy as np

from .base import Model

def _check_soil_model(smdl:Model) -> None:
    if not hasattr(smdl, 'layers'):
        msg = 'Provided soil model cannot be used becaus it has not `layers` '
        msg += 'attribute (i.e. list of layer models)!'
        raise ValueError(msg)


def init_layer_dimensions(smdl, rdm) -> Model:
    """
    Method which determines layer dimensions (i.e. depth and thickness) such 
    that the layers are equally spaced from zero (i.e. surface) to the max. 
    depth (i.e. the max. rootable depth ``rdm``). 

    :param smdl: soil model
    :type smdl: :class:`mef_agri.models.base.Model`
    :param rdm: max. rootable depth of the soil
    :type rdm: float or numpy.ndarray
    :raises ValueError: if ``smdl`` does not have a ``layers``-attribute containing the layer model instances
    :return: soil model
    :rtype: :class:`mef_agri.models.base.Model`
    """
    _check_soil_model(smdl)
    i, lt = 1, rdm / len(smdl.layers)
    for layer in smdl.layers:
        layer.thickness = lt
        layer.depth = i * lt
        layer.center_depth = layer.depth - 0.5 * lt
        i += 1

    return smdl


def distribute_nutrient_amount(nl:int, mode='linear') -> np.ndarray:
    """
    Possible values for ``mode``

    * **linear**

    linearly decreasing partitioning such that a virtual amount of 0 is assigned 
    to the (n + 1)th layer.

    :param nl: number of soil layers
    :type nl: int
    :param mode: distribution mode, defaults to ``'linear'``
    :type mode: str, optional
    :return: fractions partitioning and assigning nutrient amount to the corresponding layers
    :rtype: numpy.ndarray
    """
    if mode == 'linear':
        fs = np.interp(np.arange(1, nl + 1), [1, nl + 1], [1.0, 0.0])
        return fs / np.sum(fs)
    else:
        raise NotImplementedError()

