import datetime as dt
from pandas import DataFrame
from inspect import isclass
from copy import deepcopy

from .interfaces.tasks import TaskInterface


class EvalSetup(object):
    def __init__(self, edb, prj_dt, data_intfs, task_intfs):
        """
        TODO

        :param edb: _description_/
        :type edb: EvalDB
        :param prj_dt: _description_
        :type prj_dt: ProjectTasksExtension
        :param data_intfs: _description_
        :type data_intfs: list[Interface]
        :param task_intfs: _description_
        :type task_intfs: list[Interface]
        """
        from .interfaces import Interface
        from .db import EvalDB
        from .zoning import Zoning
        from ..farming.tasks import ProjectTasksExtension

        self._edb:EvalDB = edb
        self._prj:ProjectTasksExtension = prj_dt
        self._dintfs:dict[str, Interface] = {}
        for intf in data_intfs:
            self._dintfs[intf.data_source_id] = intf
        self._tintfs:dict[str, Interface] = {}
        for intf in task_intfs:
            self._tintfs[intf.data_source_id] = intf

        # other internal variables
        self._ins_cmds:list[str] = []
        self._field:str = None
        self._zoning:Zoning = None
        self._zinfos:DataFrame = None

    def set_up_eval(
            self, field_name, zoning_folder_id, zmodel, cmodels, info=None
        ):
        """
        TODO 

        The evaluation interval (i.e. first and last epoch) will be deduced from 
        the zoning specified with ``field_name`` and ``zoning_folder_id``.

        If ``cmodels`` is a dictionary, the keys have to match the sowing date 
        (from tasks) for which the corresponding model should be applied.
        If it is one class-definition/instance, the same crop model will be used 
        for all vegetation periods in the specified evaluation interval.

        :param field_name: _description_
        :type field_name: str
        :param zoning_folder_id: _description_
        :type zoning_folder_id: str
        :param zmodel: zone model which should be used for the evaluation
        :type zmodel: Model (class or instance)
        :param cmodels: crop model(s) which should be used for the evaluation
        :type cmodels: Model | dict[str, Model] (class or instance)
        :param info: _description_, defaults to None
        :type info: str, optional
        """
        from .interfaces.mdef import ModelDefinition

        self._ins_cmds = []  # clear already available insert commands
        self._field = field_name
        self._zoning = self._edb.zoning_data.get_zoning_by_id(
            field_name, zoning_folder_id
        )
        self._zinfos = self._edb.queries.zids_zones_by_field_zoning(
            self._field, zoning_folder_id
        )

        last_eid = self._edb.number_of_rows(self._edb.schema.table_evals.name)
        self._zinfos['eid'] = [
            last_eid + i + 1 for i in range(len(self._zinfos))
        ]
        # dictionary containing the model-structure for each zone in the field
        self._mdefs:dict[str, ModelDefinition] = None

        self._ins_eval(info, zmodel)
        self._process_tasks(cmodels)
        self._process_data()

    def _ins_eval(self, info:str, zmodel):
        from .interfaces.mdef import ModelDefinition

        if isclass(zmodel):
            zmodel = zmodel()
        zmname = zmodel.__class__.__name__
        zmmodule = zmodel.__class__.__module__

        znames, mdefs = [], []
        for tpl in self._zinfos.itertuples():
            self._ins_cmds.append(
                self._edb.schema.table_evals.sql_insert(
                    eid=getattr(tpl, 'eid'),
                    zid=getattr(tpl, 'zid'), 
                    epoch_start=self._zoning.valid_from.isoformat(),
                    epoch_end=self._zoning.valid_until.isoformat(),
                    zone_model=zmname,
                    zone_model_module=zmmodule,
                    info=info if info is not None else '---'
                )
            )
            znames.append(tpl.zone_name)
            mdefs.append(ModelDefinition(self._edb, zmodel))
        self._mdefs = dict(zip(znames, mdefs))

    def _process_tasks(self, cmodels):
        """
        ``cmodels`` can exhibit three structures

        * if it is provided as one crop-model, the same crop-model will be used for all vegetation periods in the desired evaluation-run
        * if it is proivded as dictionary the keys have to represent the sowing-dates and for the values two further possibilities exist

            * if the values are crop-models, they will be used in the corresponding vegetation periods (also if multiple crops are sown, e.g. catch crops)
            * if the values are arrays, the elements of such an array are again dictionaries with keys ``'crop'``, ``'cultivar'``, ``'cmodel'`` (class or instance)

        :param cmodels: crop models which should be used to evaluate crop related processes
        :type cmodels: Model | dict[str, Model] | dict[str, list[dict]]
        """
        from .zoning.utils import compute_zone_values
        from ..farming.tasks.sowing import Sowing
        from ..farming.tasks.harvest import Harvest

        tasks = self._prj.get_tasks(
            self._field, self._zoning.valid_from, self._zoning.valid_until
        )[self._field]

        zoned_tasks = {}
        sowinfo_init = {
            'epoch': None,
            'crops': [],
            'cultivars': [],
            'cmodels': [],
            'mids': []
        }
        nthcrop = 1
        for zcont in self._zinfos.itertuples():
            for intf in self._tintfs.values():
                intf.reset_quantity_defs()

            for etasks in tasks.values():
                for tname, task in etasks.items():
                    epoch = task.date_begin.isoformat()
                    # determine vegetation periods
                    if tname == Sowing.__name__:
                        if sowinfo['epoch'] is not None:
                            msg = 'There is already a sowing task done from '
                            msg += f'on field {self._field} from '
                            msg += f'{sowinfo['epoch']} with no associated '
                            msg += 'harvest-task - cannot add another '
                            msg += 'Sowing-task!'
                            raise ValueError(msg)

                        sowinfo = deepcopy(sowinfo_init)
                        sowinfo['epoch'] = deepcopy(epoch)
                        for appl in task.applications:
                            sowinfo['crops'].append(appl.crop.value)
                            sowinfo['cultivars'].append(appl.cultivar.value)
                            if isinstance(cmodels, dict):
                                cminfo = cmodels[epoch]
                                if isinstance(cminfo, list):
                                    for partcrop in cminfo:
                                        c1 = partcrop['crop'] == appl.crop.value
                                        c2 = partcrop['cultivar'] == \
                                             appl.cultivar.value
                                        if c1 and c2:
                                            cmodel = partcrop['cmodel']
                                    # TODO check if a model is provided for all 
                                    # TODO crops sown at a specific epoch
                                else:
                                    cmodel = cminfo
                            else:
                                cmodel = cmodels
                            if isclass(cmodel):
                                cmodel = cmodel()
                            mid = f'crop_{nthcrop}'
                            self._mdefs[zcont.zone_name].add_crop_model(
                                cmodel, mid
                            )
                            sowinfo['cmodels'].append(cmodel)
                            sowinfo['mids'].append(mid)
                            nthcrop += 1

                    elif tname == Harvest.__name__:
                        if sowinfo is None:
                            msg = 'No sowing-task has been performed on field '
                            msg += f'{self._field_} - '
                            msg += ' cannot add another Harvest-task!'
                            raise ValueError(msg)
                        cmnames = [cmdl.__class__.__name__ for \
                                   cmdl in sowinfo['cmodels']]
                        cmmodules = [cmdl.__class.__module__ for \
                                     cmdl in sowinfo['cmodels']]
                        eids = [zcont.eid for i in range(len(sowinfo['crops']))]
                        eps_start = [sowinfo['epoch'] for \
                                  i in range(len(sowinfo['crops']))]
                        eps_end = [epoch for i in range(len(sowinfo['crops']))]
                        self._ins_cmds.append(
                            self._edb.schema.table_croprot.sql_insert(
                                eid=eids,
                                crop=sowinfo['crops'],
                                cultivar=sowinfo['cultivars'],
                                crop_model=cmnames,
                                crop_model_module=cmmodules,
                                mid=sowinfo['mids'],
                                epoch_start=eps_start,
                                epoch_end=eps_end
                            )
                        )
                        sowinfo['epoch'] = None

                    # process tasks for zone
                    if not tname in zoned_tasks.keys():
                        zoned_tasks[tname] = {'numvals': {},'metadata': {}}
                    # TODO implement method for zone-value computation if 
                    # TODO something else than the median is required
                    zoned_tasks[tname]['numvals'][epoch] = compute_zone_values(
                        task, self._zoning, zone_name=zcont.zone_name
                    )
                    meta = deepcopy(task.metadata)
                    meta[TaskInterface.META_KEY_CMODEL] = sowinfo['cmodels']
                    meta[TaskInterface.META_KEY_CROP] = sowinfo['crops']
                    meta[TaskInterface.META_KEY_CULT] = sowinfo['cultivars']
                    zoned_tasks[tname]['metadata'][epoch] = deepcopy(meta)
            for tname, etasks in zoned_tasks.items():
                self._tintfs[tname].process_data(
                    etasks['numvals'], meta=etasks['metadata']
                )
            self._mdefs[zcont.zone_name].set_from_interfaces(
                list(self._tintfs.values())
            )

    def _process_data(self):
        from .zoning.utils import compute_zone_values
            
        data = self._prj.project.get_data(
            self._zoning.valid_from, self._zoning.valid_until, 
            dids=list(self._dintfs.keys()), fields=self._field
        )
            
        init_epoch = (
            self._zoning.valid_from - dt.timedelta(days=1.)
        ).isoformat()
        for zcont in self._zinfos.itertuples():
            # initialize reset mq-definitions of all data-interfaces
            for intf in self._dintfs.values():
                intf.initialize_model_quantities()

            for did, ifdata in data.items():
                dif = self._prj.project.interfaces[did]   # data interface
                eif = self._dintfs[did]   # evaluation interface

                # process data depending on if it is static and if it is 
                # separated into further data-types
                zoned_data = None
                if dif.static_data and (dif.data_types is None):
                    zdata = compute_zone_values(
                        ifdata[self._field], self._zoning, 
                        zone_name=zcont.zone_name, method=eif.zone_value_method
                    )
                    zoned_data = {init_epoch: zdata}
    
                elif dif.static_data and isinstance(dif.data_types, list):
                    zoned_data = dict(zip(
                        ifdata[self._field].keys(), 
                        [None for i in range(len(ifdata[self._field]))]
                    ))
                    for dtype, dtdata in ifdata[self._field].items():
                        zdata = compute_zone_values(
                            dtdata, self._zoning, zone_name=zcont.zone_name,
                            method=eif.zone_value_method
                        )
                        zoned_data[dtype] = zdata
                    zoned_data = {init_epoch: zoned_data}
    
                else:
                    zoned_data = dict(zip(
                        ifdata[self._field].keys(),  # epochs
                        [None for i in range(len(ifdata[self._field]))]
                    ))
                    for epoch, edata in ifdata[self._field].items():
                        if dif.data_types is None:
                            zoned_data[epoch] = compute_zone_values(
                                edata, self._zoning, 
                                zone_name=zcont.zone_name,
                                method=eif.zone_value_method
                            )
                        else:
                            zoned_data[epoch] = dict(zip(
                                edata.keys(),
                                [None for i in range(len(edata))]
                            ))
                            for dtype, dtdata in edata.items():
                                zoned_data[epoch][dtype] = \
                                compute_zone_values(
                                    dtdata, self._zoning, 
                                    zone_name=zcont.zone_name,
                                    method=eif.zone_value_method
                                )
    
                eif.process_data(zoned_data)
            self._mdefs[zcont.zone_name].set_from_interfaces(
                list(self._dintfs.values())
            )

    def save_eval_defs(self):
        # TODO
        self._edb.execute_many(self._ins_cmds)
