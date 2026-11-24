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
        def __init__(
                self, unit:str, model_id:str|None=None, 
                model_hints:str|list[str]|None=None
            ):
            self._u:str = unit
            self._mid:str = model_id
            self._mhints:str = model_hints
            self._vals:list = None
            self._eps:list[str] = None
            self._dis:list = None
            self.reset()

        @property
        def values(self) -> list:
            """
            settable - can be also provided as scalar value; when accessing this 
            property, it will be returned as list with one entry

            :return: values for model-quantity in the specified evaluation interval
            :rtype: list[any]
            """
            if not isinstance(self._vals, list):
                return [self._vals,]
            return self._vals

        @values.setter
        def values(self, values):
            self._vals = values

        @property
        def epochs(self) -> list[str]:
            """
            settable - can be also provided as scalar value; when accessing this 
            property, it will be returned as list with one entry

            If no epochs are provided for model-states or -parameters, the 
            initial evluation epoch should be used in these cases.
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
        def distributions(self) -> list:
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
            if isinstance(self._dis, dict):
                return [self._dis for i in range(len(self.values))]
            return self._dis

        @distributions.setter
        def distributions(self, distributions):
            self._dis = distributions

        @property
        def unit(self) -> str:
            """
            :return: unit of the quantity (see :class:`mef_agri.models.utils.__QS__`)
            :rtype: str
            """
            return self._u

        @property
        def model_id(self) -> str|None:
            """
            :return: model-id to which quantity belongs to
            :rtype: str | None
            """
            return self._mid

        @property
        def model_hints(self) -> list[str]|None:
            """
            :return: hints to find the model in a model-tree to which the quantity belongs to
            :rtype: list[str]|None
            """
            if isinstance(self._mhints, str):
                return [self._mhints,]
            return self._mhints

        def reset(self):
            self._vals:list = []
            self._eps:list[str] = []
            self._dis:list = []

        def to_dict(self) -> dict:
            return {
                'values': self.values,
                'epochs': self.epochs,
                'distributions': self.distributions,
                'unit': self.unit,
                'model_id': self.model_id,
                'model_hints': self.model_hints
            }

        @classmethod
        def from_dict(cls, d:dict):
            obj = cls(d['unit'])
            obj.values = d['values']
            obj.epochs = d['epochs']
            obj.distributions = d['distributions']
            obj.model_id = d['model_id']
            obj.model_hints = d['model_hints']
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

    def crop_model_id(
            self, crop:str, cultivar:str, sowing_date:str, add:bool=False
        ) -> str:
        """
        Get crop-model-id which belongs to the provided ``crop``, ``cultivar``
        and ``sowing_date``.
        If there is no one available, ``None`` will be returned if ``add=False``
        or a new crop-model will be added and the corresponding crop-model-id 
        will be returned if ``add=True``.

        :param crop: name of sown crop
        :type crop: str
        :param cultivar: name of sown cultivar
        :type cultivar: str
        :param sowing_date: iso-formatted date-string
        :type sowing_date: str
        :param add: specify if a new crop-model should be created in :func:`quantities` if it is not available, defaults to False
        :type add: bool, optional
        :return: crop-model-id
        :rtype: str
        """
        cmid = self._search_crop_model(crop, cultivar, sowing_date)[0]
        if cmid is None:
            if add:
                return self._add_crop_model(crop, cultivar, sowing_date)
            else:
                return None
        else:
            return cmid

    def add_quantity(
            self, qname:str, qtype:str, unit:str, crop:str|None=None, 
            cultivar:str|None=None, sowing_date:str|None=None, 
            cmid:str|None=None, model_hints:str|list[str]|None=None, 
            model_id:str|None=None
        ):
        """
        Add a model-quantity definition to :func:`quantities`.
        If one of the arguments ``crop``, ``cultivar`` or ``sowing_date`` is not 
        provided, it will be assumed, that the quantity belongs to the 
        zone-model-tree.
        Otherwise, it will be added to the corresponding crop-model being 
        identified by the arguments ``crop``, ``cultivar`` or ``sowing_date``.
        If there is no one available, a new entry will be created in 
        :func:`quantities` (i.e. a new dict ``self._cmdefs`` in the top level 
        of :func:`quantities` with a unique key).
        If the quantity is already available (independently of zone- or 
        crop-model), nothing will be done.

        :param qname: name of the model-quantity
        :type qname: str
        :param qtype: type of model-quantity (see :class:`mef_agri.models.base.__QS__`)
        :type qtype: str
        :param unit: unit of the model-quantity (see :class:`mef_agri.models.utils.__UNITS__`)
        :type unit: str
        :param crop: name of the sown crop (has to match a name in :class:`mef_agri.farming.crops.Crops`), defaults to None
        :type crop: str | None, optional
        :param cultivar: name of the sown cultivar (has to match a cultivar-name in :class:`mef_agri.farming.crops.Crops`), defaults to None
        :type cultivar: str | None, optional
        :param sowing_date: iso-formatted date-string, defaults to None
        :type sowing_date: str | None, optional
        :param cmid: crop-model-id (instead of providing ``crop``, ``cultivar`` and ``sowing_date``), defaults to None
        :type cmid: str | None, optional
        :param model_hints: hints to assign the quantity to the correct model (e.g. if model-id is not known but the name of top-level model and model containing the quantity), defaults to None
        :type model_hints: str | list[str] | None, optional
        :param model_id: model-id in the model-tree if it is known, defaults to None
        :type model_id: str | None, optional
        :raises KeyError: if provided ``cmid`` is not available in :func:`quantities`
        """
        if (None in (crop, cultivar, sowing_date)) and (cmid is None):
            # do nothing if quantity is already available
            if qname in self._qs[self.ZONE_MODEL_KEY][qtype].keys():
                return
            # assumption, that quantity is a zone-model quantity
            self._qs[self.ZONE_MODEL_KEY][qtype][qname] = \
                self.MQDefinition(
                    unit, model_id=model_id, model_hints=model_hints
                )
        else:
            if cmid is None:
                # check if a crop model is available for provided crop, cultivar 
                # and sowing-date
                cmid, cmdefs = self._search_crop_model(
                    crop, cultivar, sowing_date
                )
                # create a new dictionary with quantity definitions if none is 
                # available
                if cmid is None:
                    cmid = self.crop_model_id(
                        crop, cultivar, sowing_date, add=True
                    )
                    cmdefs = self._qs[self.CROP_MODELS_KEY][cmid]
            else:
                cmdefs = self._qs[self.CROP_MODELS_KEY][cmid]

            # do nothing if quantity is already available
            if qname in cmdefs[self.CMODEL_QS_KEY][qtype].keys():
                return

            # add quantity definition to the dictionary
            cmdefs[self.CMODEL_QS_KEY][qtype][qname] = self.MQDefinition(
                unit, model_id=model_id, model_hints=model_hints
            )
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

    def _add_crop_model(self, crop:str, cultivar:str, sowing_date:str) -> str:
        cmid = f'crop_{self._nthcrop}'
        self._nthcrop += 1
        cmdefs = deepcopy(self._cmdefs)
        cmdefs[self.CROP_KEY] = crop
        cmdefs[self.CULT_KEY] = cultivar
        cmdefs[self.SOWDATE_KEY] = sowing_date
        self._qs[self.CMODEL_QS_KEY][cmid] = cmdefs
        return cmid

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
