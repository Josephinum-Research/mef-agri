from .base import Zoning
from ..interfaces.geosphere_austria.inca_interfaces import (
    INCA_AtmosphereSWAT_JRV1
)
from ..interfaces.ebod_austria.ebod_interfaces import EBOD_SoilSWAT_JRV1
from ..interfaces.planetary_computer.lai.interfaces_inrae import (
    PCSentinel2_NNET10
)


class ZoningJR_V01(Zoning):
    def __init__(self, prj, zmodel):
        super().__init__(prj, zmodel)
        self.add_interface(INCA_AtmosphereSWAT_JRV1)
        self.add_interface(EBOD_SoilSWAT_JRV1)
        self.add_interface(PCSentinel2_NNET10)

    def determine_zones(self, field_name):
        pass