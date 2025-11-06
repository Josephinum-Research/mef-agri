import os
import datetime as dt
from copy import deepcopy
from inspect import isclass
from pandas import date_range

from ..interfaces.base import EvalInterface
from ..eval_def import EvaluationDefinitions
from ...data.project import Project
from ...models.base import Model
from ..interfaces.jr_management.interface import Management_JRV01


class Zoning(object):
    def __init__(self, prj:Project, zmodel:Model):
        self._prj:Project = prj
        self._ifs:dict = {}
        self._zm:Model = zmodel

    @property
    def project(self) -> Project:
        return self._prj
    
    @property
    def interfaces(self) -> dict:
        return self._ifs
    
    def add_interface(self, ei:EvalInterface) -> None:
        if isclass(ei):
            ei = ei()
        if ei.data_source_id in self._ifs.keys():
            return
        self._ifs[ei.data_source_id] = ei

    def prepare_data(
            self, tstart:dt.date, tstop:dt.date, field_name:str, 
            save_edefs:bool=False
        ) -> EvaluationDefinitions:
        ed = EvaluationDefinitions(self._zm)
        ed.set_epoch_start_end(tstart, tstop)

        zones = self.determine_zones(field_name)
        _, epsg = self._prj.get_field_geodata(field_name)
        height = self._prj.get_field_height(field_name, 'height')
        ed.provide_field_info(field_name, epsg, height, zones)
        for intf in self._ifs.values():
            for zname, zinfo in zones.items():
                for day in date_range(tstart, tstop):
                    epoch = day.date()
                    rasters = self._prj.get_data(
                        epoch, dids=intf.data_source_id, fields=field_name
                    )[field_name][intf.data_source_id]
                    if intf.time_independent:
                        epoch -= dt.timedelta(days=1)
                    ed = intf.process_data(
                        ed, rasters, zinfo['gcs'], epoch, zname
                    )
                    if intf.time_independent:
                        break

        init_epoch = tstart - dt.timedelta(days=1)
        ed.set_zone_states_init_epoch(init_epoch)
        ed.set_zone_params_init_epoch(init_epoch)

        if save_edefs:
            self.save_edefs(ed, field_name)
        return ed
    
    def save_edefs(self, ed:EvaluationDefinitions, field_name:str) -> None:
        fpath = os.path.join(self._prj.project_path, ed.EVAL_FOLDER_NAME)
        if not os.path.exists(fpath):
            os.mkdir(fpath)
        fpath = os.path.join(fpath, field_name)
        if not os.path.exists(fpath):
            os.mkdir(fpath)
        fn = 'eid_'
        if os.path.exists(os.path.join(fpath, fn + '.json')):
            fn += dt.datetime.now().isoformat()
        ed.save(fpath, fn)
    
    def determine_zones(self, field_name:str) -> dict:
        # NOTE implement in child class
        pass
