import numpy as np

from .base import Estimator
from ...models.base import Model


class UnscentedParticleFilter(Estimator):
    def __init__(self, nps, edb):
        super().__init__(nps, edb)

        # get lists of states with corresponding model ids for the zone model to 
        # ensure a fixed order in the VCM and other quantities necessary for the 
        # unscented particle filter (only states which directly belong to the 
        # zone model, i.e. states from possibly connected trees are ignored)
        zm:Model = self.zone_model_class()
        self._slist:list = []
        for mid in zm.model_tree.model_ids_intern:
            mdl:Model = zm.model_tree.get_model(mid)
            for stn in mdl.state_names:
                self._slist.append((stn, mid))
        
        # create VCMs of states for each zone
        self._vcms = {}
        for zid in self.zids:
            sd = self.database.get_states_def(zid, self.epoch_init)

        
