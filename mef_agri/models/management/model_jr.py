from ..base import Model
from .sowing.model_jr import Sowing
from .harvest.model_jr import Harvest
from .growth_regulation.model_jr import GrowthRegulation
from .fertilization.model_jr import Mineral_N_Fertilization


class Management(Model):
    @Model.is_child_model(Sowing)
    def sowing(self) -> Sowing:
        """
        :return: Model representing the sowing task
        :rtype: Sowing
        """

    @Model.is_child_model(Harvest)
    def harvest(self) -> Harvest:
        """
        :return: Model representing the harvest task
        :rtype: Harvest
        """

    @Model.is_child_model(Mineral_N_Fertilization)
    def fertilization(self) -> Mineral_N_Fertilization:
        """
        :return: Model representing the mineral fertilization task
        :rtype: Mineral_N_Fertilization
        """

    def initialize(self, epoch=None):
        super().initialize(epoch)
        self.sowing.initialize(epoch)
        self.fertilization.initialize(epoch)
        self.harvest.initialize(epoch)

    def update(self, epoch=None):
        super().update(epoch)
        self.sowing.update(epoch)
        self.fertilization.update(epoch)
        self.harvest.update(epoch)
