import functools
import inspect
from datetime import date
import numpy as np

from ..evaluation.stats_utils import DISTRIBUTION_TYPE


class __QS__(object):
    """
    Helper Class which holds the string values of the model quantities. It is 
    recommended to use/import ``Quantities`` from this module - being an 
    instance of ``__QS__`` - which holds the initialized properties. 
    """

    @property
    def STATE(self) -> str:
        """
        States are the target quantities which should be determined through 
        model evaluation. They exhibit a dynamic behavior and temporal 
        variability.
        """
        return 'state'
    
    @property
    def PARAM(self) -> str:
        """
        Parameters are quantities which define the model behavior. They 
        exhibit a considerably lower dynamic behavior than the states or can be 
        also treated as stationary random processes.
        """
        return 'parameter'

    @property
    def PFUNC(self) -> str:
        """
        Parametric functions are an extension of the parameters. 
        They combine a number of parameters which should not introduced 
        directly to the models to keep the complexity lower.
        """
        return 'pfunction'

    @property
    def OBS(self) -> str:
        """
        Observations are the driving variables in the model propagation step as 
        well as the quantities which are used in the state update step.
        """
        return 'observation'

    @property
    def ROUT(self) -> str:
        """
        Random outputs are random quantities which are computed by the models 
        and are available for model analysis and to get deeper insights into 
        crop growth.
        """
        return 'random_output'

    @property
    def DOUT(self) -> str:
        """
        Deterministic outputs are similar to the random outputs except that they 
        do not exhibit a probability distribution.
        """
        return 'deterministic_output'

Quantities = __QS__()


MODEL_DECORATORS = [
    '@Model.is_child_model', '@Model.is_required', '@Model.is_quantity',
    '@Model.is_condition'
]


class Model(object):
    """
    Base class for all models which should be evaluated for computing crop 
    growth. 
    Each model which should be evaluated and connected with other models has to 
    inherit from this class. 

    Only the root/top-level model has to be initialized "manually".
    In this case, only the ``model_name`` of the model has to be provided as 
    keyword argument.
    All child models in the model-tree (added with :func:`is_child_model`) are 
    initialized and added to the model-tree automatically in the initialization 
    of the root/top-level model.
    The ``model_name`` of child models equals the name of the decorated method.

    The conversion of decorated methods (quantity, child model or requirement) 
    into attributes happens at the time of model initialization.
    
    Quantities
    ----------
    
    There are five main types of quantities (summarized in 
    :class:`__QS__`).
    In order to add such quantities as attributes to a model, one can use 
    the decorator :func:`is_quantity`, provided as static method in this class.
    Methods, with this decorator are automatically accessable from outside 
    of the model as attributes (similar to the python built-in ``property``
    decorator). 

    Child Models
    ------------

    Child models are added as methods decorated with :func:`is_child_model`. 
    In the decorator, the model-tree will be extended (see 
    :class:`mef_agri.models.tree.ModelTree`) and the child model is implemented 
    as an attribute.

    Requirements
    ------------

    Requirements are quantities contained in other models of the model-tree 
    which are necessary in the computations of the current model.
    The advantage of using :class:`mef_agri.models.requ.Requirement` is, that 
    unit conversions are automatically applied.
    Required quantities from child models could be also directly accessed via 
    the corresponding attributes.
    But in these cases unit conversions have to be done manually.
    Another option to access required quantities from other models is, to use 
    the :func:`model_tree` property of this class.

    kwargs
    ------

    * **model_name** (*str*) - has to be provided in the initialization of the root/top-level model, otherwise it is automatically passed in the :func:`is_child_model` decorator
    * **model_id** (*str*) - is automatically passed in the :func:`is_child_model` decorator, when a child model is initialized
    * **model_tree** (:class:`mef_agri.models.tree.ModelTree`) - is automatically passed in the :func:`is_child_model` decorator, when a child model is initialized


    """
    def __init__(self, **kwargs) -> None:
        from .tree import ModelTree

        self._qs:dict = {}  # dict containing quantities with further information
        self._qnames:dict = {
            Quantities.STATE: [],
            Quantities.PARAM: [],
            Quantities.PFUNC: [],
            Quantities.OBS: [],
            Quantities.ROUT: [],
            Quantities.DOUT: [],
        }
        self._requ:dict = {}  # dict containing the required quantities from other models in the model tree
        self._cs:list = []  # list containing method names which represent conditions
        self._cinit:bool = False  # flag which is checked in the is_condition-decorator

        kwas = list(kwargs.keys())
        # case 1: only `model_name` is specified by the user/developer in the
        #         super-call of the inheriting class.
        #         This means, that the inheriting model is the top-level model
        #         and `model_id` and `model_tree` are automatically set
        case1 = 'model_name' in kwas and 'model_id' not in kwas and \
                'model_tree' not in kwas
        # case 2: `model_name`, `model_id` and `model_tree` are provided
        #         which is the case in the `is_child_model` decorator
        #         (i.e. this case ONLY appears in the initialization of the 
        #         decorated child models)
        case2 = 'model_name' in kwas and 'model_id' in kwas and \
                'model_tree' in kwas
        if case1:
            self._mname:str = kwargs['model_name']
            self._mid:str = kwargs['model_name']
            self._mtree:ModelTree = ModelTree(self)
        elif case2:
            self._mname:str = kwargs['model_name']
            self._mid:str = kwargs['model_id']
            self._mtree:ModelTree = kwargs['model_tree']
        else:
            msg = 'Supported cases to initialize a subclass of '
            msg += '`mef_agri.models.base.Model`: \n'
            msg += '- Specification of the `model_name` keyword argument by '
            msg += 'the developer in the super-call of the top-level model\n'
            msg += '- Automatic initialization done in the `is_child_model` '
            msg += 'decorator'
            raise ValueError(msg)
        
        self._epoch:date = None  # last processed epoch
        self._init:bool = False  # flag if model is initialized (i.e. `Model.initialize(epoch)` has been called)

        # initialize decorators
        if not hasattr(self, 'decorators'):
            self.decorators = MODEL_DECORATORS
        dms = self._get_decorated_methods()
        for dm in dms:
            getattr(self, dm)()
        self._cinit = True  # set flag to True such that the condition itself is called in the is_condition-decorator

    ############################################################################
    # Properties
    @property
    def requirements(self) -> dict:
        """
        :return: dictionary containing the required quantities from other models in the model tree (sorted according to the time of requirement - initialization or update)
        :rtype: dict
        """
        return self._requ
    
    @property
    def quantity_names(self) -> list[str]:
        """
        :return: names of all quantities contained in the model
        :rtype: list[str]
        """
        return list(self._qs.keys())

    @property
    def observation_names(self) -> list[str]:
        """
        :return: names of observations contained in the model
        :rtype: list[str]
        """
        return self._qnames[Quantities.OBS]
    
    @property
    def parameter_names(self) -> list[str]:
        """
        :return: names of parameters contained in the model
        :rtype: list[str]
        """
        return self._qnames[Quantities.PARAM]
    
    @property
    def state_names(self) -> list[str]:
        """
        :return: names of states contained in the model
        :rtype: list[str]
        """
        return self._qnames[Quantities.STATE]
    
    @property
    def random_output_names(self) -> list[str]:
        """
        :return: names of the random outputs in the model
        :rtype: list[str]
        """
        return self._qnames[Quantities.ROUT]
    
    @property
    def deterministic_output_names(self) -> list[str]:
        """
        :return: names of the deterministic outputs in the model
        :rtype: list[str]
        """
        self._qnames[Quantities.DOUT]
    
    @property
    def output_names(self) -> list[str]:
        """
        :return: names of other output quantities (random and deterministic) contained in the model
        :rtype: list[str]
        """
        return self._qnames[Quantities.ROUT] + self._qnames[Quantities.DOUT]
    
    @property
    def pfunction_names(self) -> list[str]:
        """
        :return: names of the functions acting as parameters in the model
        :rtype: list[str]
        """
        return self._qnames[Quantities.PFUNC]
    
    @property
    def model_name(self) -> str:
        """
        :return: name of the model
        :rtype: str
        """
        return self._mname
    
    @property
    def model_id(self) -> str:
        """
        :return: ID of the model in the ``model_tree``
        :rtype: str
        """
        return self._mid

    @property
    def model_tree(self):
        """
        :return: reference of the model-tree
        :rtype: ModelTree
        """
        return self._mtree
    
    @property
    def is_initialized(self) -> bool:
        """
        :return: flag if model is initialized (i.e. ``Model.initialize()`` has been called)
        :rtype: bool
        """
        return self._init
    
    @property
    def current_epoch(self) -> date:
        """
        :return: current epoch of the model (i.e. for which epoch, the computations in ``Model.update()`` have already been performed)
        :rtype: datetime.date
        """
        return self._epoch
    
    @property
    def conditions(self) -> list:
        """
        :return: list of method-names which represent conditions on model-quantities
        :rtype: list
        """
        return self._cs
    
    ############################################################################
    # class-private Methods
    def _get_all_quantities_from_model_tree(self, qtype:str) -> list:
        ret = []
        for mdl in self.model_tree.keys():
            obj = self.get_model_from_model_tree(mdl)
            for qname in getattr(obj, qtype):
                ret.append((qname, mdl, getattr(obj, qname)))

        return ret

    def _get_decorated_methods(self) -> str:
        def loop(cls, ret):
            ret += inspect.getsource(cls)
            for supcls in cls.__bases__:
                if issubclass(supcls, Model):
                    if supcls == Model:
                        pass
                    else:
                        ret = loop(supcls, ret)
            return ret

        sc = ''
        sc = loop(self.__class__, sc)
        decos = []
        for dec  in self.decorators:
            for part in sc.split(dec)[1:]:
                decos.append(part.split('def ')[1].split('(self')[0].strip())
        return decos
    
    ############################################################################
    # Methods which should/can be further implemented in child class
    def initialize(self, epoch:date) -> None:
        """
        Method which performs initial computations within the model before it 
        is being updated regularly with :func:`update`.
        This method can be implemented by child classes.
        In this case, the super call ``super().initialize(epoch)`` is mandatory.
        
        Here, the ``is_initialized`` flag as well as the initialization-epoch is 
        set, as well as the random outputs are initialized with nan-arrays (
        length according to ``self.model_tree.n_particles``).
        If this is not intended for certain random outputs, this has to be 
        overridden in the child class.

        States, hyper-parameters, pfunctions and observations are not 
        initialized with nan-arrays, because these quantities have to be 
        provided a priori respectively are set from "outside" of this class 

        :param epoch: intialization epoch
        :type epoch: datetime.date
        """
        # NOTE has to be called in the beginning of the initialize()-method of the child class
        self._init = True
        self._epoch = epoch

        # initialize all random outputs with nan-arrays
        self.reset_quantities(self.random_output_names)

    def update(self, epoch:date) -> None:
        """
        Method which does the computations on a daily interval.
        This method has to be implemented by child classes.
        The super call ``super().update(epoch)`` is mandatory.

        :param epoch: current epoch
        :type epoch: datetime.date
        """
        # NOTE has to be called in the beginning of the update()-method of the child class
        self._epoch = epoch

    def get_obs_epoch(self, obs_name:str) -> date:
        """
        Get the epoch related to the last observations. New observations are set 
        from outside together with the current epoch. Thus, this method can be 
        used to indicate, that new observations are available.

        :param obs_name: name of the observation in the model
        :type obs_name: str
        :return: epoch related to the last set observations
        :rtype: datetime.date
        """
        return self._qs[obs_name]['epoch']
    
    def reset_quantities(self, qs:list[str] | str, force:bool=False) -> None:
        """
        Reset specified quantities to nan-arrays.

        :param qs: model quantities which should be reset
        :type qs: list[str] | str
        """
        if isinstance(qs, str):
            qs = [qs,]
        nanarr = np.ones((self.model_tree.n_particles,))
        nanarr[:] = np.nan
        for q in qs:
            if (getattr(self, q) is not None) and (not force):
                continue
            setattr(self, q, nanarr.copy())

    ############################################################################
    # Decorator Methods
    @staticmethod
    def is_quantity(qd:str, unit:str, discrete:bool=False):
        """
        Decorator to define model quantities (see :class:`__QS__`).

        :param qd: type of model quantity (properties of :class:`__QS__`)
        :type qd: str
        :param unit: unit of the model quantity (see :class:`mef_agri.models.utils.__UNITS__`)
        :type unit: str
        :param discrete: specify if the quantity exhibits a discrete probability distribution (only matters for the TODO evalDB), defaults to False
        :type discrete: bool, optional
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(obj):
                distrtype = DISTRIBUTION_TYPE.DISCRETE if discrete \
                    else DISTRIBUTION_TYPE.CONTINUOUS
                obj._qs[func.__name__] = {
                    'qtype': qd, 'unit': unit, 'distr_type': distrtype
                }
                if qd == Quantities.OBS:
                    # Epochs are assigned to observations to know when they have
                    # been set.
                    # This is especially important when they do not appear in a 
                    # regular interval to trigger certain actions in the 
                    # evaluations.
                    # The assumption is that they are only set from 
                    # outside with values from the database (not manipulated by 
                    # other models in the model-tree)
                    obj._qs[func.__name__]['epoch'] = None
                obj._qnames[qd].append(func.__name__)
                setattr(obj, func.__name__, None)
            return wrapper
        return decorator
        
    @staticmethod
    def is_child_model(child_class):
        """
        Add another model (class which inherits from :class:`Model`) as 
        child-model. The provided ``child_class`` will be initialized in the 
        decorator and inserted into the model-tree (:func:`model_tree` property) 
        appropriately (see 
        :func:`mef_agri.models.tree.ModelTree.extend_model_tree` and 
        :func:`mef_agri.models.tree.ModelTree.add_model_reference`)

        :param child_class: class which represents the child model
        :type child_class: :class:`mef_agri.models.base.Model`
        """
        def decorator_is_child_model(func):
            @functools.wraps(func)
            def wrapper(parent_obj):
                model_id = parent_obj.model_tree.extend_model_tree(
                    parent_obj.model_id, func.__name__
                )
                child_obj = child_class(
                    model_name=func.__name__,
                    model_tree=parent_obj.model_tree,
                    model_id=model_id
                )
                parent_obj.model_tree.add_model_reference(child_obj)
                setattr(parent_obj, func.__name__, child_obj)
            return wrapper        
        return decorator_is_child_model

    @staticmethod
    def is_required(qname:str, model_id:str, unit:str):
        """
        Decorator to add required quantities which are part of another model in 
        the model-tree.
        The decorated method is converted to an attribute being an instance of 
        :class:`mef_agri.models.requ.Requirement` (i.e. the value of the 
        required quantity has to be accessed via the ``value`` property of the 
        attribute).

        :param qname: name of the required quantity in the corresponding model of the model-tree
        :type qname: str
        :param model_id: id of the model which contains the required quantity
        :type model_id: str
        :param unit: required unit of the quantity
        :type unit: str
        """
        def decorator_requ(func):
            @functools.wraps(func)
            def wrapper(obj):
                from .requ import Requirement
                if model_id.startswith('.'):
                    qmodel_id = obj.model_tree.get_model_id_from_relative_path(
                        obj.model_id, model_id
                    )
                else:
                    qmodel_id = model_id
                obj._requ[func.__name__] = {
                    'qname': qname, 'model_id': qmodel_id, 'unit': unit
                }
                rq = Requirement(qname, qmodel_id, obj, required_unit=unit)
                setattr(obj, func.__name__, rq)
            return wrapper
        return decorator_requ
    
    @staticmethod
    def is_condition(func):
        """
        Decorator to specify conditions which should be kept in the evaluation 
        of a model.

        The idea is to implement conditions on 
        quantities (e.g. lower and/or upper bounds) which should be kept, even 
        when the model is used in an estimation procedure where adding noise or 
        reseampling steps can "break" conditions on quantities and/or 
        dependencies between quantities.

        Arguments other than ``self`` are not supported in the decorated 
        methods.
        """
        functools.wraps(func)
        def wrapper(obj):
            if not obj._cinit:
                obj._cs.append(func.__name__)
            else:
                func(obj)
        return wrapper
