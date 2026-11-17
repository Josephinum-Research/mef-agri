import functools
from copy import deepcopy

from ...utils.misc import get_decorated_methods
from ...models.base import Quantities as Q


class Interface(object):
    ZONE_MODEL_KEY = 'zone'
    CROP_MODELS_KEY = 'crop_models'
    CROP_KEY = 'crop'
    CULT_KEY = 'cultivar'
    SOWDATE_KEY = 'sowing_date'
    CMODEL_QS_KEY = 'quantities'
    QDEF_KEY = 'qdef'
    UNIT_KEY = 'unit'

    class MQDefinition(object):
        # TODO
        def __init__(self, unit:str):
            self._u = unit
            self._vals:list = None
            self._eps:list[str] = None
            self._dis:list = None

        @property
        def values(self) -> list:
            return self._vals

        @values.setter
        def values(self, values):
            self._vals = values

        @property
        def epochs(self) -> list[str]:
            return self._eps

        @epochs.setter
        def epochs(self, epochs):
            self._eps = epochs

        @property
        def distributions(self) -> list:
            return self._dis

        @distributions.setter
        def distributions(self, distributions):
            self._dis = distributions

        @property
        def unit(self) -> str:
            return self._u

        def reset(self):
            pass

        def to_dict(self) -> dict:
            return {
                'values': self.values,
                'epochs': self.epochs,
                'distributions': self.distributions,
                'unit': self.unit
            }

        @classmethod
        def from_dict(cls, d:dict):
            obj = cls(d['unit'])
            obj.values = d['values']
            obj.epochs = d['epochs']
            obj.distributions = d['distributions']
            return obj


    def __init__(self):
        self._mqs:dict = {
            Q.STATE: {}, Q.PARAM: {}, Q.PFUNC: {}, Q.OBS: {}
        }
        self._cmdefs:dict = {
            self.CROP_KEY: None,
            self.CULT_KEY: None,
            self.SOWDATE_KEY: None,
            self.CMODEL_QS_KEY: {
                Q.STATE: {}, Q.PARAM: {}, Q.PFUNC: {}, Q.OBS: {}
            }
        }
        self._qs = {
            self.ZONE_MODEL_KEY: deepcopy(self._mqs), 
            self.CROP_MODELS_KEY: {}
        }
        self._nthcrop:int = 1

    @property
    def quantities(self) -> dict:
        return self._qs

    def crop_model_id(self, crop:str, cultivar:str, sowing_date:str) -> str:
        return self._search_crop_model(crop, cultivar, sowing_date)[0]


    def add_quantity(
            self, qname:str, qtype:str, unit:str, crop:str|None, 
            cultivar:str|None, sowing_date:str|None
        ):
        if None in (crop, cultivar, sowing_date):
            # assumption, that quantity is a zone-model quantity
            self._qs[self.ZONE_MODEL_KEY][qtype][qname] = \
                self.MQDefinition(unit)
        else:
            # check if a crop model is available for provided crop, cultivar 
            # and sowing-date
            cmid, cmdefs = self._search_crop_model(crop, cultivar, sowing_date)

            # create a new dictionary with quantity definitions if none is 
            # available
            if cmid is None:
                cmid = f'crop_{self._nthcrop}'
                self._nthcrop += 1
                cmdefs = deepcopy(self._cmdefs)
                cmdefs[self.CROP_KEY] = crop
                cmdefs[self.CULT_KEY] = cultivar
                cmdefs[self.SOWDATE_KEY] = sowing_date

            # add quantity definition to the dictionary
            cmdefs[self.CMODEL_QS_KEY][qtype][qname] = self.MQDefinition(unit)
            self._qs[self.CROP_MODELS_KEY][cmid] = cmdefs

    def get_qdef(
            self, qname:str, qtype:str, cmid:str|None
        ) -> Interface.MQDefinition:
        if cmid is None:
            return self.MQDefinition.from_dict(
                self._qs[self.ZONE_MODEL_KEY][qtype][qname]
            )
        else:
            return self.MQDefinition.from_dict(
                self._qs[
                    self.CROP_MODELS_KEY
                ][cmid][self.CMODEL_QS_KEY][qtype][qname]
            )

    def set_qdef(
            self, qname:str, qtype:str, qdef:Interface.MQDefinition, 
            cmid:str|None
    ):
        if cmid is None:
            self._qs[self.ZONE_MODEL_KEY][qtype][qname] = qdef.to_dict()
        else:
            self._qs[
                self.CROP_MODELS_KEY
            ][cmid][self.CMODEL_QS_KEY][qtype][qname] = qdef.to_dict()

    def reset_quantity_defs(self):
        for mid in self._qs.keys():
            for qtype in (Q.STATE, Q.PARAM, Q.PFUNC, Q.OBS):
                if mid == self.ZONE_MODEL_KEY:
                    for qname in self._qs[mid][qtype].keys():
                        self._qs[mid][qtype][qname].reset()
                else:
                    for qname in (
                        self._qs[mid][self.CMODEL_QS_KEY][qtype].keys()
                    ):
                        self._qs[mid][self.CMODEL_QS_KEY][qtype][qname].reset()

    def _search_crop_model(self, crop:str, cultivar:str, sowing_date:str):
        cmid, cmdefs = None, None
        for cmodel_id, cmodel_def in self._qs[self.CROP_MODELS_KEY].items():
            c1 = cmodel_def[self.CROP_KEY] = crop
            c2 = cmodel_def[self.CULT_KEY] = cultivar
            c3 = cmodel_def[self.SOWDATE_KEY] = sowing_date
            if c1 and c2 and c3:
                cmid = cmodel_id
                cmdefs = cmodel_def
        return cmid, cmdefs


class InterfaceOLD(object):
    """
    An evaluation-interface has to provide the definition of model-quantities.
    This can be done in a child class by decorating a method with 
    :func:`Interface.model_quantity`.
    In the initialization, the name of the decorated methods is added to 
    :func:`Interface.model_quantities` and their values are set to an instance 
    :class:`Interface.MQDefinition`.
    Properties of this class have to be set when implementing an 
    evaluation-interface through a child class.
    The definition of a model quanitity consists of

    * the quantity name => name of the decorated method
    * the model it belongs to (i.e. model-id in the model-tree, see :class:`mef_agri.models.tree.ModelTree`) => :func:`Interface.MQDefinition.model`
    * the values which should be used to generate the distribution (i.e. for sampling) => :func:`Interface.MQDefinition.values`
    * the epochs to which values belong to => :func:`Interface.MQDefinition.epochs`
    * the defnition of the distribution (see :class:`mef_agri.evalution.stats_utils.RVSampler`) => :func:`Interface.MQDefinition.distributions`
    * the unit of the quantity (see :class:`mef_agri.models.utils.__UNITS__`) => :func:`Interface.MQDefinition.unit`
    * the type of quantity (see :class:`mef_agri.model.base.__QS__`) => :func:`Interface.MQDefinition.qtype`

    """
    class MQDefinition(object):
        def __init__(self):
            self._vals:list = None
            self._eps:list[str] = None
            self._qt:str = None
            self._mdl:str = None
            self._distr:dict = None
            self._unit:str = None
            self._cmassign:dict = None

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

            If no epochs are provided for model-states or -parameters, the 
            initial evluation epoch will be used in these cases.
            For observations it is necessary to provide the epochs.

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
        def unit(self) -> str:
            """
            settable

            :return: unit of the quantity (see :class:`mef_agri.models.utils.__QS__`)
            :rtype: str
            """
            return self._unit

        @unit.setter
        def unit(self, unit):
            self._unit = unit

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

            :return: model-id to which the quantity belongs to (see :func:`mef_agri.models.base.Model.model_id`)
            :rtype: str
            """
            return self._mdl

        @model.setter
        def model(self, model):
            self._mdl = model

        @property
        def assign_crop_model(self) -> dict | None:
            """
            settable

            If it has not been set (i.e. ``None``) it will be assumed, that the 
            corresponding quantity belongs to the zone-model.

            dictionary with following content

            * ``'crop'``: name of the crop (see :class:`mef_agri.farming.crops.Crops`)
            * ``'cultivar'``: name of the cultivar (see :class:`mef_agri.farming.crops.Crops`)
            * ``'sowing_date'``: iso-formatted date-string representing sowing-date

            from these informations, the correct ``mid`` can be determined from 
            the evaluation database (see 
            :func:`mef_agri.evaluation.db.DBSchema.table_croprot`).

            :return: dictionary containing information to assign a quantity definition to a specific crop model from the evaluation-database (see :class:`mef_agri.evaluation.db.DBSchema`)
            :rtype: dict | None
            """
            return self._cmassign

        @assign_crop_model.setter
        def assign_crop_model(self, cmassign):
            self._cmassing = cmassign

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

        Has to match :func:`mef_agri.data.interface.Interface.data_source_id` 
        of the desired data-interface.
        If it is used for tasks, it has to match the name of a class in 
        :func:`mef_agri.farming.tasks`.

        :return: id to find desired data-source
        :rtype: str
        """
        return self._did
    
    @data_source_id.setter
    def data_source_id(self, did):
        self._did = did

    def process_data(self, data:dict):
        """
        The ``data`` dictionary has to have the following structure

        * the first level of keys has to contain the epochs (i.e. iso-formatted date-strings)
        * depending on :func:`mef_agri.data.interface.Interface.data_types`, there are only two level of keys or a third one

            * if it is  ``None``, the second level of keys represents the layer-ids of the georaster
            * otherwise, the second level of keys represents the data-types and the third level contains the layer-ids

        :param data: dictionary containing zoned values (i.e. representative value for all pixels in one zone)
        :type data: dict
        """
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
    