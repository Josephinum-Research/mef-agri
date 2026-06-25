import os
from multiprocessing import cpu_count
from datetime import date
from inspect import isclass

from mef_agri.data.project import ProjectData
from mef_agri.data.geosphere_austria.inca.interface import INCAInterface


class Parent(object):
    def __init__(self):
        self._a = 1
        self._b = 10

    @property
    def avalue(self) -> int:
        return self._a
    
    @avalue.setter
    def avalue(self, val):
        self._a = val

    @property
    def bvalue(self) -> int:
        return self._b
    
    @bvalue.setter
    def bvalue(self, val):
        self._b = val

    def get_sum_ab(self):
        return self._a + self._b


class Child(Parent):
    @Parent.bvalue.setter
    def bvalue(self, val):
        self._b = 3 * val



if __name__ == '__main__':
    par = Parent()
    par.bvalue = 3
    print(par.bvalue)
    print(par.get_sum_ab())

    cld = Child()
    cld.bvalue = 3
    print(cld.bvalue)
    print(cld.get_sum_ab())
