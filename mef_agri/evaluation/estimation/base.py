import os
import json
import numpy as np
import pandas as pd
from importlib import import_module
from datetime import date, timedelta

from ...models.zone.base import Zone
from ..db import EvalDB_Quantiles
from ..stats_utils import RVSampler


class Estimator(object):
    TMPFOLDER = 'tmp_est'

    def __init__(self, nps:int, edb:EvalDB_Quantiles) -> None:
        """
        Class which does the overall setup of an evaluation based on sequential
        Monte-Carlo simulation.

        :param nps: number of particles
        :type nps: int
        :param edb: instance of evaluation database
        :type edb: mef_agri.evaluation.db.EvalDB_Quantiles

        """
        self._nps:int = nps  # number of particles representing distribution of state vector
        self._db:EvalDB_Quantiles = edb
        self._rvs = RVSampler()  # sampler object to get RVs
        self._zmdl = None
        ########################################################################
        # GET AND SETUP EVALUATION DATA
        # get required data for evaluation from database
        eval_data = self._db.get_eval_data(self._db.evaluation_id)
        self._ep_1 = date.fromisoformat(eval_data['epoch_start'].values[0])
        self._ep_n = date.fromisoformat(eval_data['epoch_end'].values[0])
        self._ep_0 = self._ep_1 - timedelta(days=1)  # initialization epoch of states and parameters is one day before the start of the evaluation
        self._zmcl = getattr(
            import_module(eval_data['zmodel_module'].values[0]), 
            eval_data['zmodel'].values[0]
        )
        self._zd = self._db.get_zones_data(self._db.evaluation_id)

    @property
    def n_particles(self) -> int:
        """
        :return: number of particles
        :rtype: int
        """
        return self._nps

    @property
    def database(self) -> EvalDB_Quantiles:
        """
        :return: object which represents database containing definitions and results of an evaluation
        :rtype: EvaluationDB
        """
        return self._db
    
    @property
    def epoch_init(self) -> date:
        return self._ep_0
    
    @property
    def epoch_start(self) -> date:
        return self._ep_1
    
    @property
    def epoch_end(self) -> date:
        return self._ep_n
    
    @property
    def zone_model_class(self):
        return self._zmcl
    
    @property
    def zones(self) -> pd.DataFrame:
        return self._zd
    
    @property
    def zids(self) -> list[str]:
        return list(self._zd['zid'].unique())

    def process(self, wdir:str) -> None:
        ########################################################################
        # initialize sql script
        tmpp = os.path.join(wdir, self.TMPFOLDER)
        if not os.path.exists(tmpp):
            os.mkdir(tmpp)
        self._db.create_sql_script(tmpp, 'sqlscr.txt')

        ########################################################################
        # Evaluation
        for zid in self.zids:
            zdata = self.zones[self.zones['zid'] == zid]
            zname = zdata['zname'].values[0]
            ####################################################################
            # GET AND SETUP ZONE DATA
            zone_gcs = self._db.get_zone_gcs(zid).values

            ####################################################################
            # PROCESS MODEL
            self._zmdl:Zone = self.zone_model_class()
            self._zmdl.gcs = zone_gcs
            self._zmdl.latitude = zdata['latitude'].values[0]
            self._zmdl.height = zdata['height'].values[0]
            self._zmdl.model_tree.n_particles = self._nps

            ####################################################################
            # CROP ROTATION
            self._zmdl.crop_rotation.add_data(
                self._db.get_crop_rotation(zid)
            )

            ################################################################
            # SET INITIAL STATES, PARAMS, PFUNCS
            self._set_states(zid, self.epoch_init)
            self._set_params(zid, self.epoch_init)
            self._set_pfuncs(zid, self.epoch_init)

            ################################################################
            # INITIALIZE ZONE MODEL AND START PROCESSING FOR EACH DAY
            self._zmdl.initialize(self.epoch_init)
            for pd_epoch in pd.date_range(self.epoch_start, self.epoch_end):
                epoch = pd_epoch.date()
                print(zname + ' - processing epoch: ' + epoch.isoformat())
                ############################################################
                # GET OBSERVATIONS FROM DATABASE AND SET THEM IN THE MODEL
                self._set_obs(zid, epoch)
                ############################################################
                # STATE PROPAGATION
                self.propagate(epoch)
                ############################################################
                # PF-UPDATE / WEIGHT COMPUTATION
                self.update(epoch)
                ############################################################
                # SAVE STATES, OUTPUTS AND EVALUATED PFUNCTIONS
                for model in self._zmdl.model_tree.models:
                    for sname in model.state_names:
                        self._db.add_states_eval(
                            zid, epoch, sname, model.model_id, 
                            getattr(model, sname),
                            discrete=self._zmdl.model_tree.is_q_discrete(
                                sname, model.model_id
                            )
                        )
                    for oname in model.random_output_names:
                        self._db.add_out_eval(
                            zid, epoch, oname, model.model_id,
                            getattr(model, oname), 
                            discrete=self._zmdl.model_tree.is_q_discrete(
                                oname, model.model_id
                            )
                        )
                    for fname in model.pfunction_names:
                        pf = getattr(model, fname)
                        if pf is None:
                            continue
                        if pf.is_sampled:
                            self._db.add_pfuncs_eval(
                                zid, epoch, fname, model.model_id,
                                pf.current_value,
                                discrete=self._zmdl.model_tree.is_q_discrete(
                                    fname, model.model_id
                                )
                            )

                # Trigger setting of initial states, params and pfuncs in the 
                # currently sown crop 
                if self._zmdl.crop_rotation.crop_sown:
                    self._set_states(zid, epoch)
                    self._set_params(zid, epoch)
                    self._set_pfuncs(zid, epoch)
                    self._zmdl.crop_rotation.current_crop.initialize(epoch)
                
                self._db.add_cmd_to_script(self._db.insert_states_eval_cmd())
                self._db.add_cmd_to_script(self._db.insert_out_eval_cmd())
                self._db.add_cmd_to_script(self._db.insert_pfuncs_eval_cmd())

        self._db.close_script()
        self._db.execute_script()

    def propagate(self, epoch) -> None:
        pass

    def update(self, epoch) -> None:
        pass

    def _set_states(self, zid:int, epoch:date) -> None:
        for tpl in self._db.get_states_def(zid, epoch).itertuples():
            dinfo = json.loads(tpl.distr)
            val = self._rvs.get_sampled_values(tpl.value, dinfo, self._nps)
            self._zmdl.model_tree.set_quantity(tpl.name, tpl.model, val)
            self._db.add_states_eval(
                zid, epoch, tpl.name, tpl.model, val,
                discrete=self._zmdl.model_tree.is_q_discrete(
                    tpl.name, tpl.model
                )
            )
        self._db.add_cmd_to_script(self._db.insert_states_eval_cmd())

    def _set_params(self, zid:int, epoch:date) -> None:
        for tpl in self._db.get_params_def(zid, epoch).itertuples():
            dinfo = json.loads(tpl.distr)
            if dinfo['sample']:
                val = self._rvs.get_sampled_values(tpl.value, dinfo, self._nps)
                self._db.add_params_eval(
                    zid, epoch, tpl.name, tpl.model, val,
                    discrete=self._zmdl.model_tree.is_q_discrete(
                        tpl.name, tpl.model
                    )
                )
            else:
                val = tpl.value * np.ones((self._nps,))
            self._zmdl.model_tree.set_quantity(tpl.name, tpl.model, val)
        self._db.add_cmd_to_script(self._db.insert_params_eval_cmd())

    def _set_pfuncs(self, zid:int, epoch:date) -> None:
        for tpl in self._db.get_pfuncs_def(zid, epoch).itertuples():
            fdef = json.loads(tpl.fdef)
            self._zmdl.model_tree.set_quantity(tpl.name, tpl.model, fdef)
            if fdef['sample']:
                self._zmdl.model_tree.get_quantity(tpl.name, tpl.model).sample(
                    self._rvs, self._nps
                )

    def _set_obs(self, zid:str, epoch:date) -> None:
        for tpl in self._db.get_obs_def(zid, epoch).itertuples():
            dinfo = json.loads(tpl.distr)
            if dinfo['sample']:
                val = self._rvs.get_sampled_values(tpl.value, dinfo, self._nps)
                self._db.add_obs_eval(
                    zid, epoch, tpl.name, tpl.model, val,
                    discrete=self._zmdl.model_tree.is_q_discrete(
                        tpl.name, tpl.model
                    )
                )
            else:
                val = tpl.value * np.ones((self._nps,))
            self._zmdl.model_tree.set_quantity(
                tpl.name, tpl.model, val, epoch=epoch
            )
        self._db.add_cmd_to_script(self._db.insert_obs_eval_cmd())
