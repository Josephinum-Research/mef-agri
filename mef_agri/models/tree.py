from copy import deepcopy
from datetime import date

from .base import Quantities
from .utils import Units
from ..evaluation.stats_utils import DISTRIBUTION_TYPE


class ModelTree(object):
    """
    Crop and soil models are connected and arranged in a tree-representation.
    Thus, there is one root/top-level model (``model`` argument in the 
    initialization) and further models are added by decorating attributes of a 
    parent-model (instance of a class inheriting from 
    :class:`mef_agri.models.base.Model`)  with the decorator 
    :func:`mef_agri.models.base.Model.is_child_model`.

    An unique model-id is assigned to each model in the model-tree. This id 
    corresponds to the ``model_id`` attribute of 
    :class:`mef_agri.models.base.Model`. The model-id is designed like an 
    absolute path, where each level corresponds to the name of the model (
    ``model_name`` attribute of :class:`mef_agri.models.base.Model`). The 
    separation character is a point or dot.
    The :func:`extend_model_tree` method is used to create the appropriate 
    model-id and it extends the dictionary (``_tree`` attribute or property
    :func:`tree_dict`) which 
    represents the model tree and holds references to all models.
    An example of ``_tree`` is shown in the following where the root 
    model name is *m1* and its two child models *m2* and *m3*

    .. code-block::

        self._tree = {
            'm1': {
                'ref': mdl1,  # object reference
                'parent_id': None,  # no parent for the root model
                'child_ids': ['m1.m2', 'm1.m3']
            },
            'm1.m2': {
                'ref': mdl2,  # object reference
                'parent_id': 'm1',
                'child_ids: []
            },
            'm1.m3': {
                'ref': mdl3,  # object reference
                'parent_id': 'm1',
                'child_ids: []
            }
        }

    :param model: root/top-level model in the tree
    :type model: :class:`mef_agri.models.base.Model`
    """
    def __init__(self, model) -> None:
        # class private variables
        self._toplvl:str = model.model_name
        self._tree:dict = {model.model_name: {
            'ref': model, 'parent_id': None, 'child_ids': []
        }}
        self._nps:int = None
        self._conn:dict = {}

    @property
    def n_particles(self) -> int:
        """
        :return: number of particles (i.e. realization or values) representing the distributions of the model quantities
        :rtype: int
        """
        return self._nps
    
    @n_particles.setter
    def n_particles(self, value:int):
        self._nps = value

    @property
    def tree_dict(self) -> dict:
        """
        :return: model-tree representation as dictionary
        :rtype: dict
        """
        return self._tree

    @property
    def model_ids(self) -> list[str]:
        """
        :return: ids of the models in the tree (also connected trees are considered)
        :rtype: list[str]
        """
        ret = list(self._tree.keys())
        for tree in self._conn.values():
            ret += list(tree._tree.keys())
        return ret
    
    @property
    def models(self) -> list:
        """
        :return: references to the objects representing the models in the tree (also connected trees are considered)
        :rtype: list
        """
        ret = []
        for entry in self._tree.values():
            ret.append(entry['ref'])
        for tree in self._conn.values():
            for entry in tree._tree.values():
                ret.append(entry['ref'])
        return ret
    
    @property
    def text_representation(self) -> str:
        """
        :return: text representation of the model tree
        :rtype: str
        """
        def loop(model, outstr, indent, praefix):
            ws1 = ''.join(' ' for i in range(len(indent)))
            us = ''.join('_' for i in range(3 + len(model.model_name)))
            ws2 = ''.join(' ' for i in range(len(us)))

            outstr += ws1 + ' ' + us + '\n' + ws1 + '|' + ws2 + '|\n'
            outstr += indent + '| ' + model.model_name + '  |\n'
            outstr += ws1 + '|' + us + '|\n'

            qs = [
                'state', 'observation', 'hyper_parameter', 'output', 
                'hp_function'
            ]
            for dtype in qs:
                if len(getattr(model, dtype + '_names')) > 0:
                    outstr += ws1 + '  |\n'
                    outstr += ws1 + '  |__ ' + dtype + 's\n'
                    outstr += ws1 + '      ' + ''.join('-' for i in range(len(dtype) + 1)) + '\n'
                    for qname in getattr(model, dtype + '_names'):
                        if ((dtype == 'output') and 
                            (qname in model._qts[Quantities.DOUT].keys())):
                            outstr += ws1 + '      <d> ' + qname + '\n'
                        else:
                            outstr += ws1 + '      < > ' + qname + '\n'
            
            indent += ' => '
            for child_id in self._tree[praefix]['child_ids']:
                outstr = loop(
                    self._tree[child_id]['ref'], outstr, indent, child_id
                )

            return outstr

        outstr = '\n\n\n'
        outstr = loop(self._tree[self._toplvl]['ref'], outstr, '', self._toplvl)
        outstr += '\n\n\n'
        return outstr
    
    def extend_model_tree(
            self, parent_model_id:str, child_model_name:str
        ) -> str:
        """
        Get the desired structure of the model-id in the model-tree. The 
        returned model-id can be seen as an absolute path, where the 
        individual levels correspond to the model names which are separated by 
        points.
        This method is used by :func:`mef_agri.models.base.Models.is_child_model`

        :param parent_model_id: model-id of the parent-model (attribute ``model_id`` of :class:`mef_agri.models.base.Model`)
        :type parent_model_id: str
        :param child_model_name: name of the child-model (attribute ``model_name`` of :class:`mef_agri.models.base.Model`)
        :type child_model_name: str
        :return: unique model-id in the tree
        :rtype: str
        """
        mid = parent_model_id + '.' + child_model_name
        self._tree[mid] = {
            'ref': None,
            'parent_id': parent_model_id,
            'child_ids': []
        }
        self._tree[parent_model_id]['child_ids'].append(mid)
        return mid

    def add_model_reference(self, model) -> None:
        """
        Add a model-reference to the model tree. 
        This method is used by :func:`mef_agri.models.base.Models.is_child_model`.

        :param model: object representing a model
        :type model: :class:`mef_agri.models.base.Model`
        """
        # add reference to child model to the tree
        self._tree[model.model_id]['ref'] = model

    def get_model_id_from_relative_path(self, model_id:str, relpath:str) -> str:
        """
        Get the model-id from a given relative path representing the way of 
        "walking" through the model tree. A relative path starts with a dot 
        followed either by 
        
        * the model names (attribute ``model_name`` of :class:`mef_agri.models.base.Model`) which means to go "down" in the model-tree 
        * or the string ``__parent__`` indicating to go "up" in the model tree.

        :param model_id: model-id of the model which is the "starting point" of the provided relative path
        :type model_id: str
        :param relpath: the relative path representing the way to the desired model in the model tree
        :type relpath: str
        :raises ValueError: if ``relpath`` does not start with a dot
        :return: model-id of the desired model
        :rtype: str
        """
        if not relpath.startswith('.'):
            msg = 'A relative path has to start with a dot `.` followed either '
            msg += 'by the model names (i.e. going down the model tree) '
            msg += 'or `__parent__`-terms (i.e. going up the model tree).'
            raise ValueError(msg)

        mid = deepcopy(model_id)
        for model_name in relpath.split('.')[1:]:
            if model_name == '__parent__':
                mid = self._tree[mid]['parent_id']
            else:
                mid += '.' + model_name
        return mid

    def _get_model_check_qname(self, qname:str, model_id:str):
        qmdl = self.get_model(model_id)
        if qmdl is None:
            msg = 'Provided `model_id` is not available neither in this model '
            msg += 'tree nor in the connected model trees.'
            raise ValueError(msg)
        if not qname in qmdl.quantity_names:
            msg = 'Provided `qname` not available in the specified `model_id`.'
            raise ValueError(msg)
        
        return qmdl
    
    def get_model(self, model_id:str):
        """
        Get model from model-tree or from other connnected model-trees. If 
        return value is ``None``, no model with provided ``model_id`` is found.

        :param model_id: model-id of the required model
        :type model_id: str
        :return: object representing the requested model
        :rtype: :class:`mef_agri.models.base.Model`
        """
        if model_id in self._tree.keys():
            return self._tree[model_id]['ref']
        else:
            for tree in self._conn.values():
                if model_id in tree._tree.keys():
                    return tree._tree[model_id]['ref']
                
    def has_model(self, model_id:str) -> bool:
        """
        Check if the model-tree contains a model.

        :param model_id: model-id of requested model
        :type model_id: str
        :return: boolean indicator
        :rtype: bool
        """
        if self.get_model(model_id) is None:
            return False
        else:
            return True
                
    def get_qinfos(self, qname:str, model_id:str) -> dict:
        """
        Get information about a model-quantity (see arguments of 
        :func:`ssc_csm.models.base.Model.is_quantity`)

        :param qname: name of the model-quantity
        :type qname: str
        :param model_id: model-id of the model containing the required quantity
        :type model_id: str
        :return: dictionary with quantity information (keys: ``qtype``, ``unit``, ``distr_type``, ``epoch`` in the case of observations)
        :rtype: dict
        """
        return self.get_model(model_id)._qs[qname]
    
    def is_q_discrete(self, qname:str, model_id:str) -> bool:
        """
        Check if quantity exhibits a discrete distribution

        :param qname: name of the model-quantity
        :type qname: str
        :param model_id: model-id of the model containing the required quantity
        :type model_id: str
        :return: boolean indicator
        :rtype: bool
        """
        dt = self.get_qinfos(qname, model_id)['distr_type']
        return True if dt == DISTRIBUTION_TYPE.DISCRETE else False

    def get_quantity(self, qname:str, model_id:str, unit=None):
        """
        Get a model-quantity from a model in the model-tree. 
        If it is not found, also the connected trees are screened. 
        If ``unit`` is provided, the appropriate conversion is applied.

        :param qname: name of the quantity in the model
        :type qname: str
        :param model_id: id of the model which contains the quantity
        :type model_id: str
        :param unit: desired unit of the requested quantity (see :class:`mef_agri.models.utils.__UNITS__`), defaults to None
        :type unit: str, optional
        :return: requested quantity
        :rtype: any
        """
        if unit is None:
            return getattr(self._get_model_check_qname(qname, model_id), qname)
        value =  getattr(self._get_model_check_qname(qname, model_id), qname)
        usrc = self.get_model(model_id)._qs[qname]['unit']
        return Units.convert(value, usrc, unit)
    
    def get_obs_epoch(self, obs_name:str, model_id:str) -> date:
        """
        Returns the epoch of the current observations.

        :param obs_name: name of the observation (i.e. quantity) in the model
        :type obs_name: str
        :param model_id: id of the model wich contains the quantity
        :type model_id: str
        :return: epoch of the observation
        :rtype: datetime.date
        """
        qmdl = self._get_model_check_qname(obs_name, model_id)
        if not (obs_name in qmdl.observation_names):
            return
        return qmdl._qs[obs_name]['epoch']
    
    def set_quantity(
            self, qname:str, model_id:str, value, unit=None, epoch:date=None
        ) -> None:
        """
        Set the value of a quantity in a model in the tree. 
        If ``unit`` is provided and ``value`` is not a dict (i.e. set a 
        :class:`mef_agri.models.utils.HPFunction`), 
        the appropriate conversion is applied.

        :param qname: name of the quantity in the model
        :type qname: str
        :param model_id: id of the model which contains the quantity
        :type model_id: str
        :param value: value which should be set for the quantity
        :type value: float or 1D numpy.ndarray, dict for `HPFunction`
        :param unit: unit of the provided value (see :class:`mef_agri.models.utils.__UNITS__`), defaults to None
        :type unit: str, optional
        :param epoch: epoch of an observation, which is necessary to indicate new incoming observations
        :type epoch: datetime.date
        """
        qmdl = self._get_model_check_qname(qname, model_id)
        if isinstance(value, dict):
            from .utils import HPFunction
            hpf = HPFunction()
            hpf.define(value)
            setattr(qmdl, qname, hpf)
            return
        if qname in qmdl.observation_names:
            if epoch is None:
                msg = 'Setting an observation requires to pass an epoch!'
                raise ValueError(msg)
            qmdl._qs[qname]['epoch'] = epoch
        if unit is None:
            setattr(qmdl, qname, value)
            return
        utrg = qmdl._qs[qname]['unit']
        setattr(qmdl, qname, Units.convert(value, unit, utrg))

    def check_conditions(self) -> None:
        """
        Iterates all models in the tree (also models in connected trees) and 
        calls all methods which represent a condition (see 
        :func:`mef_agri.models.base.Model.is_condition`)
        """
        for mdl in self.models:
            for cname in mdl._cs:
                getattr(mdl, cname)()

    def connnect(self, tree) -> None:
        """
        Connect two model trees. This method does the connection bi-directional, 
        i.e. it sets the provided model-tree object in the ``_conn`` 
        dictionary of this model-tree object and vice versa. Thus, it is not 
        necessary to call this method a second time from the provided model-tree 
        object.
        The ``_conn`` dictionary holds the reference to the model-tree object 
        with its key being equal to the model-id of the root/top-level model.

        :param tree: model-tree object which should be connected to this model-tree object
        :type tree: :class:`mef_agri.models.tree.ModelTree`
        """
        self._conn[tree._toplvl] = tree
        tree._conn[self._toplvl] = self

    def disconnect(self, tree) -> None:
        """
        Disconnect model-trees (i.e. remove the corresponding entries in the 
        ``_conn`` dictionaries of this model-tree and the provided model-tree).

        :param tree: model-tree object which should be disconnected from this model-tree object
        :type tree: :class:`mef_agri.models.tree.ModelTree`
        """
        if tree._toplvl in self._conn.keys():
            del self._conn[tree._toplvl]
        if self._toplvl in tree._conn.keys():
            del tree._conn[self._toplvl]
