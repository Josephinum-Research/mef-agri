import numpy as np
from datetime import date, timedelta

from .base import Estimator
from ...models.utils import Units as U


def effective_sample_size_choice(wi:np.ndarray, nthresh:float=None) -> bool:
    if nthresh is None:
        # default threshold wil be half the number of particles
        nthresh = 0.5 * wi.shape[0]
    neff = 1. / np.sum(np.power(wi, 2.))
    if neff <= nthresh:
        return True
    else:
        return False


class BPF_EPIC_Obs_LAI(Estimator):
    def __init__(self, edb, **kwargs):
        super().__init__(edb, **kwargs)
        # initialize weights of particles
        self._wi = np.ones((self.n_particles,)) / self.n_particles
        # standard deviation of measurement noise (LAI from Sentinel-2)
        self._std = 0.3

    @property
    def std_lai_obs(self) -> float:
        return self._std
    
    @std_lai_obs.setter
    def std_lai_obs(self, val):
        self._std = val

    def propagate(self, epoch):
        self._zmdl.update(epoch)

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
        const = 1. / (self._std * np.sqrt(2. * np.pi))
        exparg = np.power((lai_obs - lai_mdl) / self._std, 2.0)
        self._wi *= const * np.exp(exparg)

        # resampling if necessary
        if not effective_sample_size_choice(self._wi):
            return
        
        # TODO
