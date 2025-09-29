import os
import json
import numpy as np
import pandas as pd
from importlib import import_module
from datetime import date, timedelta

from mef_agri.utils.misc import set_attributes

from ...models.zone.base import Zone
from ..db import EvalDB_Quantiles
from ..stats_utils import RVSampler


TMPFOLDER = 'tmp_est'


class Estimator(object):
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

    def process(self, wdir:str) -> None:
        # initialize sql script
        tmpp = os.path.join(wdir, TMPFOLDER)
        if not os.path.exists(tmpp):
            os.mkdir(tmpp)
        self._db.create_sql_script(tmpp, 'sqlscr.txt')

        ########################################################################
        # GET AND SETUP EVALUATION DATA
        # get required data for evaluation from database
        eid = self._db.evaluation_id
        eval_data = self._db.get_eval_data(eid)
        epoch_start = date.fromisoformat(eval_data['epoch_start'].values[0])
        epoch_end = date.fromisoformat(eval_data['epoch_end'].values[0])
        zmdl_class = getattr(
            import_module(eval_data['zmodel_module'].values[0]), 
            eval_data['zmodel'].values[0]
        )

        # get zones and corresponding data
        zone_data = self._db.get_zones_data(eid)
        for zid in zone_data['zid'].unique():
            zname = zone_data[zone_data['zid'] == zid]['zname'].values[0]
            ####################################################################
            # GET AND SETUP ZONE DATA
            zone_gcs = self._db.get_zone_gcs(zid).values
            # initial values of states and hyper parameters
            # epoch has to be one day before start of the evaluation
            epoch_init = epoch_start - timedelta(days=1)

            ####################################################################
            # PROCESS MODEL
            self._zmdl:Zone = zmdl_class()
            self._zmdl.gcs = zone_gcs
            self._zmdl.latitude = zone_data['latitude'].values[0]
            self._zmdl.height = zone_data['height'].values[0]
            self._zmdl.model_tree.n_particles = self._nps

            ####################################################################
            # CROP ROTATION
            self._zmdl.crop_rotation.add_data(
                self._db.get_crop_rotation(zid)
            )

            ################################################################
            # SET INITIAL STATES, HPARAMS, HPFUNCS
            self._set_states(zid, epoch_init)
            self._set_hparams(zid, epoch_init)
            self._set_hpfuncs(zid, epoch_init)

            ################################################################
            # INITIALIZE ZONE MODEL AND START PROCESSING FOR EACH DAY
            self._zmdl.initialize(epoch_init)
            for pd_epoch in pd.date_range(epoch_start, epoch_end):
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
                # SAVE STATES, OUTPUTS AND EVALUATED HPFUNCTIONS
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
                        if oname == 'finished':  # TODO check if this is still necessary
                            pass
                        self._db.add_out_eval(
                            zid, epoch, oname, model.model_id,
                            getattr(model, oname), 
                            discrete=self._zmdl.model_tree.is_q_discrete(
                                oname, model.model_id
                            )
                        )
                    for fname in model.hp_function_names:
                        hpf = getattr(model, fname)
                        if hpf.is_sampled:
                            self._db.add_hpfuncs_eval(
                                zid, epoch, fname, model.model_id,
                                hpf.current_value,
                                discrete=self._zmdl.model_tree.is_q_discrete(
                                    fname, model.model_id
                                )
                            )

                # Trigger setting of initial states, hparams and hpfuncs in the 
                # currently sown crop 
                if self._zmdl.crop_rotation.crop_sown:
                    self._set_states(zid, epoch)
                    self._set_hparams(zid, epoch)
                    self._set_hpfuncs(zid, epoch)
                    self._zmdl.crop_rotation.current_crop.initialize(epoch)
                
                self._db.add_cmd_to_script(self._db.insert_states_eval_cmd())
                self._db.add_cmd_to_script(self._db.insert_out_eval_cmd())
                self._db.add_cmd_to_script(self._db.insert_hpfuncs_eval_cmd())

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

    def _set_hparams(self, zid:int, epoch:date) -> None:
        for tpl in self._db.get_hparams_def(zid, epoch).itertuples():
            dinfo = json.loads(tpl.distr)
            if dinfo['sample']:
                val = self._rvs.get_sampled_values(tpl.value, dinfo, self._nps)
                self._db.add_hparams_eval(
                    zid, epoch, tpl.name, tpl.model, val,
                    discrete=self._zmdl.model_tree.is_q_discrete(
                        tpl.name, tpl.model
                    )
                )
            else:
                val = tpl.value * np.ones((self._nps,))
            self._zmdl.model_tree.set_quantity(tpl.name, tpl.model, val)
        self._db.add_cmd_to_script(self._db.insert_hparams_eval_cmd())

    def _set_hpfuncs(self, zid:int, epoch:date) -> None:
        for tpl in self._db.get_hpfuncs_def(zid, epoch).itertuples():
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
