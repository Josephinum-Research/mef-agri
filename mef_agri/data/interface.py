import os
import functools
import dill
from geopandas import GeoDataFrame
from datetime import date
from multiprocessing import Process, Queue, cpu_count

from ..utils.raster import GeoRaster
from ..utils.misc import get_decorated_methods


class Worker(Process):
    """
    Class which is used when ``parallel`` of :func:`add_data_task`-decorator is 
    set to ``True``.
    It inherits from ``multiprocessing.Process``.
    """
    QFLAG_MESSAGE = '__MESSAGE__'
    QFLAG_FINISHED = '__FINISHED__'

    def __init__(self, task, settings):
        super().__init__(daemon=True)
        self._t = dill.dumps(task)
        self._s = settings

    def run(self):
        try:
            dill.loads(self._t)(None, **self._s)
        except Exception as exc:
            self._s['queue'].put((self.QFLAG_MESSAGE, str(exc)))
        finally:
            self._s['queue'].put((self.QFLAG_FINISHED, self._s['pid']))


class Interface(object):
    """
    Base class for data interfaces.

    By using the decorator :func:`add_data_task`, it is possible to specify 
    methods of a child-class as tasks which will be executed in the provided 
    order within :func:`prj_add_data`.
    The return-values of decorated methods have to be dictionaries which will 
    be passed to the next task in the specified order as ``kwargs``.
    The last task must return a list with epochs (``datetime.date`` or 
    iso-formatted strings ``YYYY-MM-dd``) for which data has been requested and 
    saved (Note: if :func:`static_data` is ``True`` nothing has to be returned).
    Through the decorator they are only appended to the evaluation order at the 
    time of initialization.
    Decorated methods can be used like other normal methods after 
    initialization - if ``parallel`` is not set to ``True`` in the decorator!
    
    When setting ``parallel`` to ``True`` in this decorator, the corresponding 
    task will be split into sub-processess.
    This means, that the corresponding method will be passed to :class:`Worker` 
    as target for the evaluation in sub-processes.
    Settings for the sub-processes have to be specified in the returned 
    dictionary of the previous task.
    The keys of this dictionary have to be equal to process-ids which are 
    available throuhg :func:`process_ids` and the values must be dictionaries 
    which are passed to the corresponding process (i.e. ``settings`` passed to 
    the constructor of :class:`Worker`).
    If results from the sub-processes should be passed to the next task in 
    order, it is possible to access a ``multiprocessing.Queue`` instance within 
    ``kwargs`` with key ``queue``.
    With ``kwargs['queue'].put(('key', value))`` the provided key-value pairs 
    will be collected in the main process in a dictionary which will be passed 
    to the next task.
    If the key in the queues' put method equals ``Worker.QFLAG_MESSAGE``, the 
    corresponding message will be appropriately handled by the project-instance 
    calling :func:`prj_add_data`.

    """
    class Errors:
        class FirstTaskParallel(Exception):
            def __init__(self, *args):
                super().__init__(*args)

        class OrderNotSpecified(Exception):
            def __init__(self, *args):
                super().__init__(*args)

        class TimerangeRequired(Exception):
            def __init__(self, *args):
                super().__init__(*args)

    def __init__(self):
        # private variables which belong to properties
        self._did:str = None  # data source id
        self._descr:str = None  # description of interface
        self._dtypes:list[str] = None  # specify multiple data types which could be available at one epoch
        self._dtypes_descr:dict[str] = None  # description of datatypes
        self._static:bool = False  # flag if data is static
        self._grclass = GeoRaster  # georaster-class which the interface represents
        self._dir:str = None  # absolute path of directory where georaster should be saved to
        self._aoi:GeoDataFrame = None  # aoi/field for which data should be provided/gathered
        self._tr:tuple[date, date] = None  # time-range for which data should be gathered
        self._ep:date = None  # epoch for which data should be derived
        self._nprcs:int = max(cpu_count() // 2, 2)  # number of cores to use for tasks
        self._pids:list[str] = None  # process ids
        
        # private variables which are related to execution
        self._init:bool = False  # flag if class is initialized
        self._add_tasks = {}  # dict containing functions and tasks which are performed in ``self.prj_add_data``
        self._get_tasks = {}  # dict containing functions and tasks which are performed in ``self.prj_get_data``

        # call decorated methods once initially to set everything up to add data
        dms = get_decorated_methods(
            self, ['@Interface.add_data_task', '@Interface.get_data_task']
        )
        for dm in dms:
            getattr(self, dm)()
        self._init = True

    ############################################################################
    # PROPERTIES DEFINING THE INTERFACE
    @property
    def data_source_id(self) -> str:
        """
        Settable

        Set it to a fixed value in the constructor of the child-class

        :return: id of the data source
        :rtype: str
        """
        return self._did
    
    @data_source_id.setter
    def data_source_id(self, val):
        self._did = val

    @property
    def static_data(self) -> bool:
        """
        Settable

        Set it to a fixed value in the constructor of the child-class

        Default value: ``False`` (i.e. data is available per epoch)

        :return: flag if data source provides static data (i.e. one time-independent dataset)
        :rtype: bool
        """
        return self._static
    
    @static_data.setter
    def static_data(self, val):
        self._static = val

    @property
    def data_types(self) -> list[str]:
        """
        Settable

        Set it to a fixed value in the constructor of the child-class

        Default value: ``None`` (i.e. only one data type or data)

        Users can leave it also unspecfied, if types/layers of georaster should 
        not be divided into different directories. A possible use-case is, if  
        types/layers should available in the georaster itself.

        :return: list of different data types which can be available for one epoch (e.g. weather data with preciptitation, air temperature, ... for each day)
        :rtype: list[str]
        """
        return self._dtypes
    
    @data_types.setter
    def data_types(self, dtypes):
        self._dtypes = dtypes

    @property
    def description(self) -> str:
        """
        Settable

        Set it to a fixed value in the constructor of the child-class

        :return: Description of the interface and corresponding data
        :rtype: str
        """
        return self._descr

    @description.setter
    def description(self, value):
        self._descr = value

    @property
    def data_types_description(self) -> dict[str]:
        """
        Settable

        Set it to a fixed value in the constructor of the child-class

        :return: description for each datatype belonging to the interface (if provided - see :func:`data_types`)
        :rtype: dict[str]
        """
        return self._dtypes_descr

    @data_types_description.setter
    def data_types_description(self, value):
        self._dtypes_descr = value

    @property
    def georaster_class(self):
        """
        Settable

        Set it to a fixed value in the constructor of the child-class

        :return: class which the interface represents (e.g. used by :func:`mef_agri.data.project.Project.get_data`)
        :rtype: class
        """
        return self._grclass

    @georaster_class.setter
    def georaster_class(self, value):
        self._grclass = value

    ############################################################################
    # PROPERTIES RELATED TO MULTIPROCESSING
    @property
    def n_processes(self) -> int:
        """
        Settable

        Default value: half the number of ``multiprocessing.cpu_count()`` but at least 2

        :return: number of processes which should be used to execute functions decorated with ``Interface.add_data_task``
        :rtype: int
        """
        return self._nprcs

    @n_processes.setter
    def n_processes(self, value):
        self._nprcs = value
        self._pids = None

    @property
    def process_ids(self) -> list[str]:
        """
        :return: list of process ids with length according to :func:`n_processes`
        :rtype: list[str]
        """
        if self._pids is None:
            self._pids = [
                'pid-{}'.format(i + 1) for i in range(self.n_processes)
            ]
        return self._pids
    
    ############################################################################
    # PROPERTIES WHICH PROVIDE INFORMATION FOR CHILD CLASSES
    @property
    def directory(self) -> str:
        """
        :return: absolute path to directory where data is located (depending on :func:`static_data` and :func:`data_types` it contains further folders or the geotiff(s) with metadata)
        :rtype: str
        """
        return self._dir
    
    @property
    def aoi(self) -> GeoDataFrame:
        """
        :return: one row of a GeoDataFrame representing the field for which data will be requested (see :func:`prj_add_data` and :func:`prj_get_data`)
        :rtype: geopandas.GeoDataFrame
        """
        return self._aoi

    @property
    def timerange(self) -> tuple[date, date]:
        """
        :return: timerange for which data will be requested (see :func:`prj_add_data` and :func:`prj_get_data`)
        :rtype: tuple[datetime.date, datetime.date]
        """
        return self._tr
    
    ############################################################################
    # METHODS
    def prj_add_data(
            self, ddir:str, aoi:GeoDataFrame, trange:list[date]=None
        ) -> list[date]:
        """
        Processes the methods which are decorated with :func:`add_data_task`.

        If georasters are saved, the project-data structure has to be satisfied, 
        because otherwise these data will not be considered in the 
        consistency checks within 
        :func:`mef_agri.data.project.Project.add_data`.
        The structure of data-folders is

        1. :func:`data_source_id`s
        2. name of the field
        3. epochs
        4. :func:`data_types`

        1.) and 2.) are available through :func:`directory`.
        Thus, it is only necessary to consider 3.) and 4.) when saving 
        georasters in the tasks.
        If :func:`static_data` is ``True``, level 3.) of the folder structure 
        is omitted.
        If :func:`data_types` is ``None``, level 4.) of the folder structure is 
        omitted.

        :param ddir: directory where data should be located, see :func:`directory`
        :type ddir: str
        :param aoi: one row of a GeoDataFrame, see :func:`aoi`
        :type aoi: GeoDataFrame
        :param trange: first and last epoch for which data should be gathered, defaults to None
        :type trange: list[datetime.date], optional
        :return: list of epochs for which data is indeed gathered (empty list or ``None`` if :func:`static_data` is ``True``)
        :rtype: list[datetime.date]
        """
        # set properties such that they are available in methods of child class
        self._dir = ddir
        self._aoi = aoi
        if trange is None and not self.static_data:
            msg = 'Data is not static for interface {} - thus, `trange` has to '
            msg += 'be provided for `Interface.prj_add_data`!'
            raise self.Errors.TimerangeRequired(msg.format(self.data_source_id))
        self._tr = trange

        # process add-data-tasks
        ret = {}
        for i in range(len(self._add_tasks)):
            task = self._add_tasks[str(i + 1)]
            # first step of add_data-steps has to be a function/method
            if (i == 0) and task['mp']:
                msg = 'First processing step of `Interface.prj_add_data` '
                msg += 'must not be a parallel task!'
                raise self.Errors.FirstTaskParallel(msg)
            
            # processing the add_data-steps
            if task['mp']:
                q, prcs = Queue(), []
                prcs, finish_flags = [], {}
                for i in range(self.n_processes):
                    settings = {'pid': self.process_ids[i], 'queue': q} | \
                        ret[self.process_ids[i]]
                    pi = Worker(task['f'], settings)
                    pi.start()
                    prcs.append(pi)
                    finish_flags[self.process_ids[i]] = False

                ret = {}
                while True:
                    key, val = q.get()
                    if key == Worker.QFLAG_FINISHED:
                        finish_flags[val] = True
                        if not (False in list(finish_flags.values())):
                            break
                    elif key == Worker.QFLAG_MESSAGE:
                        print(val)
                    else:
                        ret[key] = val
                
                for prc in prcs:
                    prc.join()
            else:
                ret = task['f'](self, **ret)
                if ret is None:
                    ret = {}
        return ret
    
    ############################################################################
    # DECORATORS
    @staticmethod
    def add_data_task(order:int=None, parallel:bool=False):
        """
        Decorate a method of a child-class of :class:`Interface` which 
        represents a task which will be evaluated in :func:`prj_add_data`.
        If ``order`` is not provided, it will be assumed, that there is only 
        one task for adding data.

        If ``parallel`` is set to ``True``, the decorated method will be passed 
        to :class:`Worker` as target which will be evaluated in sub-processs 
        which number is determined by :func:`n_processes`.
        In this case, the self-reference will be set to ``None`` within 
        :class:`Worker` and therefore it must not be used in the decorated 
        method.

        :param order: order in which tasks will be evaluated (has to start with 1!), defaults to None
        :type order: int, optional
        :param parallel: specify if the corresponding task should be split into processes and evaluated with pythons' multiprocessing package, defaults to False
        :type parallel: bool, optional
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(obj, **kwargs):
                if obj._init:
                    return func(obj, **kwargs)
                else:
                    if order is None and obj._add_tasks:
                        msg = 'No order provided but more than one '
                        msg += '`add_data_task` is specified!'
                        raise Interface.Errors.OrderNotSpecified(msg)
                    obj._add_tasks[str(order)] = {'f': func, 'mp': parallel}
            return wrapper
        return decorator
    