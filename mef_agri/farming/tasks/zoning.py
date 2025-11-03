from datetime import date

from .task import Task


class Zoning(Task):
    def __init__(self):
        super().__init__()
        self._meta['valid_until'] = None

    @property
    def valid_until(self) -> date:
        return self._meta['valid_until']
    
    @valid_until.setter
    def valid_until(self, val):
        self._meta['valid_until'] = self._check_date(val)
