import datetime as dt
from pandas import DataFrame

from .interfaces import Interface
from .db import EvalDB
from .zoning import Zoning
from .zoning.utils import compute_zone_values
from ..models.base import Quantities
from ..farming.tasks import ProjectTasksExtension
from ..farming.tasks.sowing import Sowing
from ..farming.tasks.harvest import Harvest


class EvalSetup(object):
    def __init__(
            self, edb:EvalDB, prj_dt:ProjectTasksExtension, 
            intfs:list[Interface]
        ):
        self._edb:EvalDB = edb
        self._prj:ProjectTasksExtension = prj_dt
        self._intfs:dict[str, Interface] = {}
        for intf in intfs:
            self._intfs[intf.data_source_id] = intf
        self._ins_cmds:list[str] = []
        self._field:str = None
        self._zoning:Zoning = None
        self._zinfos:DataFrame = None

    def set_up_eval(
            self, field_name:str, zoning_folder_id:str, zmodel, cmodel, 
            info:str=None
        ):
        self._field = field_name
        last_eid = self._edb.number_of_rows(self._edb.schema.table_evals.name)
        self._zoning = self._edb.zoning_data.get_zoning_by_id(
            field_name, zoning_folder_id
        )
        self._zinfos = self._edb.queries.zids_zones_by_field_zoning(
            self._field, zoning_folder_id
        )
        self._zinfos['eid'] = [
            last_eid + i + 1 for i in range(len(self._zinfos))
        ]

        self._ins_cmds = []
        self._ins_eval(zoning_folder_id, last_eid, info, zmodel)
        self._process_tasks(last_eid, cmodel)
        self._process_data(last_eid)
        self._edb.execute_many(self._ins_cmds)

    def _process_data(self, last_eid:int):
        dbs = self._edb.schema
        data = self._prj.project.get_data(
            self._zoning.valid_from, self._zoning.valid_until, 
            dids=list(self._intfs.keys()), fields=self._field
        )
        
        init_epoch = (
            self._zoning.valid_from - dt.timedelta(days=1.)
        ).isoformat()
        for did, ifdata in data.items():
            dif = self._prj.project.interfaces[did]  # instance of data-interface
            for zname in self._zoning.layer_ids:
                # process data depending on if it is static and if it is 
                # separated into further data-types
                zoned_data = None
                if dif.static_data and (dif.data_types is None):
                    zdata = compute_zone_values(
                        ifdata[self._field], self._zoning, zone_name=zname
                    )
                    zoned_data = {init_epoch: zdata}

                elif dif.static_data and isinstance(dif.data_types, list):
                    zoned_data = dict(zip(
                        ifdata[self._field].keys(), 
                        [None for i in range(len(ifdata[self._field]))]
                    ))
                    for dtype, dtdata in ifdata[self._field].items():
                        zdata = compute_zone_values(
                            dtdata, zoning, zone_name=zname
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
                                edata, self._zoning, zone_name=zname
                            )
                        else:
                            zoned_data[epoch] = dict(zip(
                                edata.keys(),
                                [None for i in range(len(edata))]
                            ))
                            for dtype, dtdata in edata.items():
                                zoned_data[epoch][dtype] = compute_zone_values(
                                    dtdata, self._zoning, zone_name=zname
                                )

                self._intfs[did].process_data(zoned_data)
                for mqname in self._intfs[did].model_quantities:
                    mq:Interface.MQDefinition = \
                        getattr(self._intfs[did], mqname)
                    if mq.qtype == Quantities.PFUNC:
                        # TODO
                        pass
                    else:
                        ins_data = dict(
                            eid=[... for i in range(len(mq.values))],
                            name=[mqname for i in range(len(mq.values))],
                            model=[mq.model for i in range(len(mq.values))],
                            epoch=mq.epochs,
                            value=mq.values,
                            distr=mq.distributions
                        )
                        get_sql = None
                        if mq.qtype == Quantities.STATE:
                            get_sql = dbs.table_states_def.sql_insert
                        elif mq.qtype == Quantities.PARAM:
                            get_sql = dbs.table_params_def.sql_insert
                        elif mq.qtype == Quantities.OBS:
                            get_sql = dbs.table_obs_def.sql_insert
                        self._ins_cmds.append(get_sql(**ins_data))

    def _ins_eval(self, info:str, zmodel):
        zmname = zmodel.__name__
        zmmodule = zmodel.__module__
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

    def _process_tasks(self, last_eid:int, cmodel):
        cmname = cmodel.__name__
        cmmodule = cmodel.__module__
        tasks = self._prj.get_tasks(
            self._field, self._zoning.valid_from, self._zoning.valid_until
        )[self._field]

        epsow = None
        for epoch, etasks in tasks.items():
            for tname, task in etasks.items():
                # determine vegetation periods
                if tname == Sowing.__name__:
                    if epsow is not None:
                        msg = f'There is already a sown crop from {epsow} '
                        msg += f'present on field {self._field} which has not '
                        msg += 'been harvested yet - cannot add another '
                        msg += 'Sowing-task!'
                        raise ValueError(msg)
                    epsow = task.date_begin.isoformat()
                elif tname == Harvest.__name__:
                    if epsow is None:
                        msg = f'No crop has been sown on field {self._field_} -'
                        msg += ' cannot add another Harvest-task!'
                        raise ValueError(msg)

                # compute zoned values from tasks
