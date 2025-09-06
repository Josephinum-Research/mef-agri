import numpy as np

from ...base import Model, Quantities as Q, Units as U


class Sowing(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
