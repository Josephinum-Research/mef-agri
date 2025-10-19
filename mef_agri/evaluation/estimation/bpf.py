import numpy as np

from .pf import ParticleFilter
from ...models.utils import Units as U
from .pf_utils import effective_sample_size_choice


class BPF_EPIC_Obs_LAI(ParticleFilter):
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
