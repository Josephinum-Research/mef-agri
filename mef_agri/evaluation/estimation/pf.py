import numpy as np

from .base import Estimator
from .pf_utils import resampling_systematic


class ParticleFilter(Estimator):
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
