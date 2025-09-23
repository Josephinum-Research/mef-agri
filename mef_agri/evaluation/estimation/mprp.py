from .base import Estimator


class ModelPropagation(Estimator):
    def propagate(self, epoch):
        self._zmdl.update(epoch)
