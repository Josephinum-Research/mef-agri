import functools
from copy import deepcopy

from ..zoning import Zoning
from ...utils.misc import get_decorated_methods
from ...data.interface import Interface as DataInterface


class Interface(object):
    class MQDefinition(object):
        def __init__(self):
            self._vals:list = None
            self._eps:list[str] = None
            self._qt:str = None
            self._mdl:str = None
            self._distr:dict = None

        @property
        def values(self) -> list[any]:
            """
            settable - can be also provided as scalar value; when accessing this 
            property, it will be returned as list with one entry

            :return: values for model-quantity in the specified evaluation interval
            :rtype: list[any]
            """
            if isinstance(self._vals, (int, float, str, bool)):
                return [self._vals,]
            return self._vals

        @values.setter
        def values(self, vals):
            self._vals = vals

        @property
        def epochs(self) -> list[str]:
            """
            settable - can be also provided as scalar value; when accessing this 
            property, it will be returned as list with one entry

            :return: epochs/iso-formatted dates for :func:`values`
            :rtype: list[str]
            """
            if isinstance(self._eps, str):
                return [self._eps,]
            return self._eps
        
        @epochs.setter
        def epochs(self, epochs):
            self._eps = epochs

        @property
        def qtype(self) -> str:
            """
            settable

            :return: identifier for quantity type (see :obj:`mef_agri.models.base.Quantities` and :class:`mef_agri.models.base.__QS__`)
            :rtype: str
            """
            return self._qt
        
        @qtype.setter
        def qtype(self, qt):
            self._qt = qt

        @property
        def model(self) -> str:
            """
            settable

            if the quantity belongs to TODO

            :return: model-id to which the quantity belongs to (see :func:`mef_agri.models.base.Model.model_id`)
            :rtype: str
            """
            return self._mdl

        @model.setter
        def model(self, model):
            self._mdl = model

        @property
        def distributions(self) -> list[dict]:
            """
            settable

            The dictionary to define a distribution has to contain

            * the mandatory key ``distr_id`` with an appropriate value (see :class:`mef_agri.evaluation.stats_utils.DISTRIBUTIONS`)
            * remaining keys according to specified ``distr_id`` (see :func:`mef_agri.evaluation.stats_utils.RVSampler.get_sampled_values`)

            If the distribution varies with time, it can be also a list of 
            dictionaries with appropriate distribution information (i.e. same 
            length as :func:`values` and :func:`epochs`).
            
            It can provided as one dict (i.e. stationary distribution).
            In this case, it will be returned as a list with same length as 
            :func:`values`, when accessing this property.

            :return: definition of the quantity's distribution 
            :rtype: list[dict]
            """
            if isinstance(self._distr, dict):
                return [self._distr for i in range(len(self.values))]
            return self._distr
        
        @distributions.setter
        def distributions(self, distr):
            self._distr = distr

    def __init__(self):
        self._mqs:list[str] = []
        self._did:str = None
        
        for mq in get_decorated_methods(
            self, ['model_quantity'], iterate_super_class=Interface
        ):
            getattr(self, mq)()
        self.initialize_model_quantities()

    @property
    def model_quantities(self) -> list[str]:
        """
        :return: names of the functions decorated with :func:`model_quantity`
        :rtype: list[str]
        """
        return deepcopy(self._mqs)
    
    @property
    def data_source_id(self) -> str:
        """
        settable - fix it in the constructor of the child class

        :return: has to match :func:`mef_agri.data.interface.Interface.data_source_id` of the desired data-interface
        :rtype: str
        """
        return self._did
    
    @data_source_id.setter
    def data_source_id(self, did):
        self._did = did

    def process_data(self, data:dict):
        pass
        
    def initialize_model_quantities(self):
        for mq in self._mqs:
            setattr(self, mq, Interface.MQDefinition())

    @staticmethod
    def model_quantity(func):
        @functools.wraps(func)
        def wrapper(obj):
            obj._mqs.append(func.__name__)
        return wrapper
    