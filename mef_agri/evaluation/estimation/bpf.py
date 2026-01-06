import numpy as np

from ...models.utils import Units as U
from .pf_utils import effective_sample_size_choice, resampling_systematic
from .base import Estimator


class BootstrapParticleFilter(Estimator):
    def __init__(self, nps, edb):
        super().__init__(nps, edb)
        # initialize weights of particles
        self._wi = np.ones((self._nps,)) / self._nps
        # resampling method
        self._rm = resampling_systematic
        # propagation noise definitions
        self._pn = {}
        # default system noise as fraction of corresponding value
        self._dsn = 0.001

    @property
    def resampling_method(self):
        """
        Resampling method from ``mef_agri.evaluation.estimation.pf_utils``
        """
        return self._rm
    
    @resampling_method.setter
    def resampling_method(self, func):
        if not callable(func):
            msg = 'Provided argument is not callable!'
            raise ValueError(msg)
        self._rm = func

    @property
    def default_system_noise(self) -> float:
        return self._dsn
    
    @default_system_noise.setter
    def default_system_noise(self, val):
        self._dsn = val

    def set_propagation_noise(
            self, state_name:str, model_id:str, value:float
        ) -> None:
        if not model_id in self._pn.keys():
            self._pn[model_id] = {}
        self._pn[model_id][state_name] = value

    def propagate(self, epoch):
        self._zmdl.update(epoch)
        zipped = zip(
            self._zmdl.model_tree.models, self._zmdl.model_tree.model_ids
        )
        for mdl, mid in zipped:
            for stn in mdl.state_names:
                q = getattr(mdl, stn)
                if q is None:
                    # this is the case for crop-model states at the day of 
                    # sowing > model-trees of the zone model and the crop model 
                    # are already connected but values are not set in the models
                    # > does not need to be changed because adding noise at the 
                    # day of sowing would not make sense (initial states are 
                    # already sampled)
                    continue

                if (mid in self._pn.keys()) and (stn in self._pn[mid].keys()):
                    std = self._pn[mid][stn] * np.ones((self._nps,))
                else:
                    std = np.abs(q) * self._dsn
                sv = q + np.random.normal(loc=0.0, scale=std)
                setattr(mdl, stn, sv)
        if self._zmdl.crop_rotation.crop_sown:
            self._zmdl.model_tree.check_conditions(ignore_connected=True)
        else:
            self._zmdl.model_tree.check_conditions()



class BPF_EPIC_Obs_LAI(BootstrapParticleFilter):
    def __init__(self, nps, edb):
        super().__init__(nps, edb)
        # standard deviation of measurement noise (LAI from Sentinel-2)
        self._std = 0.3

    @property
    def std_lai_obs(self) -> float:
        """
        :return: standard deviation of observation noise
        :rtype: float
        """
        return self._std
    
    @std_lai_obs.setter
    def std_lai_obs(self, val):
        self._std = val

    def update(self, epoch):
        # get observed lai and check if it is up to date
        lai_obs = self._zmdl.model_tree.get_quantity(
            'lai', 'zone.sentinel2_lai', unit=U.none
        )
        if lai_obs is None:
            # no lai observations available yet
            return
        if not False in np.isnan(lai_obs):
            # only nans in lai obs => no new sentinel-2 image
            return
        lai_mdl = self._zmdl.model_tree.get_quantity(
            'lai', 'crop.leaves', unit=U.none
        )

        # compute weights
        exparg = -0.5 * np.power((lai_obs - lai_mdl) / self._std, 2.0)
        self._wi *= np.exp(exparg)

        # resampling if necessary
        if not effective_sample_size_choice(self._wi):
            return
        rix = self._rm(self._wi)  # get indices of resampled values
        self._wi = np.ones((self._nps,)) / self._nps  # resetting the weights
        for mdl in self._zmdl.model_tree.models:
            for stn in mdl.state_names:
                setattr(mdl, stn, getattr(mdl, stn)[rix])
        self._zmdl.model_tree.check_conditions()
