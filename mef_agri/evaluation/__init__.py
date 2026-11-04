import datetime as dt

from .interfaces import Interface
from .db import EvalDB
from .zoning.utils import compute_zone_values
from ..models.base import Quantities
from ..farming.tasks import ProjectTasksExtension


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

    def set_up_eval(
            self, field_name:str, zoning_folder_id:str, zmodel, cmodel, 
            info:str=None
        ):
        dbs, dbq = self._edb.schema, self._edb.queries
        
        new_eid = self._edb.number_of_rows(dbs.table_evals.name)
        zmname = zmodel.__name__
        zmmodule = zmodel.__module__
        cmname = cmodel.__name__
        cmmodule = cmodel.__module__
        zoning = self._edb.zoning_data.get_zoning_by_id(
            field_name, zoning_folder_id
        )

        # process task data
        tasks = self._prj.get_tasks(
            field_name, zoning.valid_from, zoning.valid_until
        )

        # process other data
        data = self._prj.project.get_data(
            zoning.valid_from, zoning.valid_until, 
            dids=list(self._intfs.keys()), fields=field_name
        )
        zone_data = dbq.zids_zones_by_field_zoning(field_name, zoning_folder_id)
        ins_cmds = []
        init_epoch = (zoning.valid_from - dt.timedelta(days=1.)).isoformat()
        for did, ifdata in data.items():
            dif = self._prj.project.interfaces[did]  # instance of data-interface
            for zname in zoning.layer_ids:
                new_eid += 1
                zid = zone_data[
                    zone_data['zone_name'] == zname
                ]['zid'].values[0]
                ins_cmds.append(
                    dbs.table_evals.sql_insert(
                        eid=new_eid, 
                        zid=zid, 
                        epoch_start=zoning.valid_from.isoformat(),
                        epoch_end=zoning.valid_until.isoformat(),
                        zone_model=zmname,
                        zone_model_module=zmmodule,
                        info=info if info is not None else '---'
                    )
                )

                # process data depending on if it is static and if it is 
                # separated into further data-types
                zoned_data = None
                if dif.static_data and (dif.data_types is None):
                    zdata = compute_zone_values(
                        ifdata[field_name], zoning, zone_name=zname
                    )
                    zoned_data = {init_epoch: zdata}

                elif dif.static_data and isinstance(dif.data_types, list):
                    zoned_data = dict(zip(
                        ifdata[field_name].keys(), 
                        [None for i in range(len(ifdata[field_name]))]
                    ))
                    for dtype, dtdata in ifdata[field_name].items():
                        zdata = compute_zone_values(
                            dtdata, zoning, zone_name=zname
                        )
                        zoned_data[dtype] = zdata
                    zoned_data = {init_epoch: zoned_data}

                else:
                    zoned_data = dict(zip(
                        ifdata[field_name].keys(),  # epochs
                        [None for i in range(len(ifdata[field_name]))]
                    ))
                    for epoch, edata in ifdata[field_name].items():
                        if dif.data_types is None:
                            zoned_data[epoch] = compute_zone_values(
                                edata, zoning, zone_name=zname
                            )
                        else:
                            zoned_data[epoch] = dict(zip(
                                edata.keys(),
                                [None for i in range(len(edata))]
                            ))
                            for dtype, dtdata in edata.items():
                                zoned_data[epoch][dtype] = compute_zone_values(
                                    dtdata, zoning, zone_name=zname
                                )

                self._intfs[did].process_data(zoned_data)
                for mqname in self._intfs[did].model_quantities:
                    mq:Interface.MQDefinition = \
                        getattr(self._intfs[did], mqname)
                    if mq.qtype == Quantities.PFUNC:
                        pass
                    else:
                        ins_data = dict(
                            eid=[new_eid for i in range(len(mq.values))],
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
                        ins_cmds.append(get_sql(**ins_data))

        self._edb.execute_many(ins_cmds)