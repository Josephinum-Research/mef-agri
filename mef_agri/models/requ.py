import numpy as np

from .utils import PFunction, Units


class Requirement(object):
    """
    This class represents a quantity from a model in the model tree 
    required by another model in the model tree.
    Via the ``value`` property, it is possible to get the quantity and also a 
    setter is implemented, if one wants to change the value of the quantity 
    in the foreign model of the model tree.
    Unit conversion is considered when the ``value`` property is called through 
    the usage of the :func:`mef_agri.models.tree.ModelTree.get_quantity` method.

    An instance of ``Requirement`` is callable if the underlying quantity is of 
    type :class:`mef_agri.models.utils.HPFunction`.
    In this case the provided value is passed to the ``__call__()`` method of 
    the HPFunction and the computed value is returned (``numpy.ndarray``).

    :param qname: name of the required quantity in the corresponding model of the model tree
    :type qname: str
    :param qmodel_id: id of the model which contains the required quantity
    :type qmodel_id: str
    :param requesting_model: object reference of the requesting model
    :type requesting_model: :class:`mef_agri.models.base.Model`
    :param required_unit: unit of quantity which is required by the requesting model, defaults to None (see :class:`mef_agri.models.utils.__UNITS__`)
    :type required_unit: str, optional
    """
    def __init__(
            self, qname:str, qmodel_id:str, requesting_model, 
            required_unit:str=None
        ) -> None:
        from .base import Model

        self._qname:str = qname
        self._qmdl_id:str = qmodel_id
        self._rmdl:Model = requesting_model  # object which requires the desired quantity
        self._qmdl:Model = self._rmdl.model_tree.get_model(self._qmdl_id)
        if required_unit is None:
            required_unit = Units.undef
        self._ur = required_unit  # unit required by the requesting model

    @property
    def value(self):
        """
        :return: get the requested quantity from the model tree (also acts as setter)
        :rtype: numpy.ndarray
        """
        if self._qmdl is None:
            self._qmdl = self._rmdl.model_tree.get_model(self._qmdl_id)
        return self._qmdl.model_tree.get_quantity(
            self._qname, self._qmdl_id, unit=self._ur
        )
    
    @value.setter
    def value(self, value):
        if self._qmdl is None:
            self._qmdl = self._rmdl.model_tree.get_model(self._qmdl_id)
        self._qmdl.model_tree.set_quantity(
            self._qname, self._qmdl_id, value, unit=self._ur
        )

    def __call__(self, value) -> np.ndarray:
        """
        A required quantity is callable if the underlying quantity is of type 
        `PFunction`. In this case, `value` is directly passed to the 
        `__call__` of the `PFunction`.

        :param value: value which is passed to `__call__` of `PFunction`
        :type value: numpy.ndarray
        :return: computed value from underlying `PFunction`
        :rtype: numpy.ndarray
        """
        if not isinstance(self.value, PFunction):
            msg = 'Only required quantities of type `HPFunction` are callable!'
            raise ValueError(msg)
        self.value(value)