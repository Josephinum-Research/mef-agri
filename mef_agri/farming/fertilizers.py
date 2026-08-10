import json


class Fertilizer(object):
    """
    Class representing a fertilizer. If a specific fertilizer needs to be 
    created, just initialize this class and set the nutrient properties being 
    fractions (i.e. range of possible values is [0, 1]).
    """
    def __init__(self):
        self._name = 'TBD'
        self._no3 = 0.0
        self._nh4 = 0.0
        self._cao = 0.0
        self._cao_sol = 0.0
        self._p2o5 = 0.0
        self._p2o5_sol = 0.0
        self._k2o_sol = 0.0
        self._so3_sol = 0.0
        self._zn = 0.0

    @property
    def name(self) -> str:
        """
        :return: name of fertilizer
        :rtype: str
        """
        return self._name
    
    @name.setter
    def name(self, name):
        self._name = name

    @property
    def NO3(self) -> float:
        r"""
        :return: fraction of nitrate :math:`NO_{3}^{-}` within a specified amount of fertilizer
        :rtype: float
        """
        return self._no3

    @NO3.setter
    def NO3(self, value:float):
        self._no3 = value

    @property
    def NH4(self) -> float:
        r"""
        :return: fraction of ammonia :math:`NH_{4}^{+}` within a specified amount of fertilizer
        :rtype: float
        """
        return self._nh4

    @NH4.setter
    def NH4(self, value:float):
        self._nh4 = value

    @property
    def P2O5(self) -> float:
        r"""
        :return: fraction of diphosphoruspentoxide ("phosphate") :math:`P_{2}O_{5}` within a specified amount of fertilizer
        :rtype: float
        """
        return self._p2o5

    @P2O5.setter
    def P2O5(self, value:float):
        self._p2o5 = value

    @property
    def P2O5_sol(self) -> float:
        r"""
        :return: fraction of water-soluble :math:`P_{2}O_{5}` within a specified amount of fertilizer
        :rtype: float
        """
        return self._p2o5_sol

    @P2O5_sol.setter
    def P2O5_sol(self, value:float):
        self._p2o5_sol = value

    @property
    def K2O_sol(self) -> float:
        r"""
        :return: fraction of water-soluble potassium oxide :math:`K_{2}O` within a specified amount of fertilizer
        :rtype: float
        """
        return self._k2o_sol

    @K2O_sol.setter
    def K2O_sol(self, value:float):
        self._k2o_sol = value

    @property
    def CaO(self) -> float:
        r"""
        :return: fraction of calcium oxide :math:`CaO` within a specified amount of fertilizer
        :rtype: float
        """
        return self._cao

    @CaO.setter
    def CaO(self, value:float):
        self._cao = value

    @property
    def CaO_sol(self) -> float:
        r"""
        :return: fraction of water-soluble calcium oxide :math:`CaO` within a specified amount of fertilizer
        :rtype: float
        """
        return self._cao_sol
    
    @CaO_sol.setter
    def CaO_sol(self, value:float):
        self._cao_sol = value

    @property
    def SO3(self) -> float:
        r"""
        :return: fraction of sulphur trioxide :math:`SO_{3}` within a specified amount of fertilizer
        :rtype: float
        """
        return self._so3_sol

    @SO3.setter
    def SO3(self, value:float):
        self._so3_sol = value

    @property
    def SO3_sol(self) -> float:
        r"""
        :return: fraction of water-soluble sulphur trioxide :math:`SO_{3}` within a specified amount of fertilizer
        :rtype: float
        """
        return self._so3_sol

    @SO3_sol.setter
    def SO3_sol(self, value:float):
        self._so3_sol = value

    @property
    def Zn(self) -> float:
        r"""
        :return: fraction of zinc :math:`Zn` within a specified amount of fertilizer
        :rtype: float
        """

    @Zn.setter
    def Zn(self, value:float):
        self._zn = value

    def get_dict_repr(self) -> dict:
        di = {}
        for prop in self.get_properties():
            di[prop] = getattr(self, prop)
        return di

    @classmethod
    def get_properties(cls) -> list:
        """
        Classmethod
        
        :return: names of all methods decorated with ``@property``
        :rtype: list
        """
        props = []
        for key, val in vars(cls).items():
            if isinstance(val, property):
                props.append(key)
        return props
    
    @classmethod
    def from_json(cls, jstr:str):
        di = json.loads(jstr)
        fert = cls()
        for key, val in di:
            setattr(fert, key, val)
        return fert
    

class DBIntegration(object):
    COL_FERT_NAME = 'fname'
    COL_FERT_DEF = 'fdef'

    def __init__(self, table_name:str):
        self._tname:str = table_name

    @property
    def sql_table_exists(self) -> str:
        sql = 'SELECT name FROM sqlite_master WHERE type=\'table\' AND '
        sql += f'name=\'{self._tname}\';'
        return sql

    @property
    def sql_create(self) -> str:
        sql = f'CREATE TABLE {self._tname} ({self.COL_FERT_NAME} TEXT, '
        sql += f'{self.COL_FERT_DEF} TEXT, PRIMARY KEY ({self.COL_FERT_NAME}));'
        return sql

    @property
    def sql_insert_defaults(self) -> str:
        sql = f'INSERT INTO {self._tname} '
        sql += f'({self.COL_FERT_NAME}, {self.COL_FERT_DEF}) VALUES '
        for fert in self.default_fertilizers():
            sql += self.sql_insert_tuple(fert) + ', '
        return sql[:-2] + ';'
    
    def sql_insert(self, fert:Fertilizer) -> str:
        sql = f'INSERT INTO {self._tname} '
        sql += f'({self.COL_FERT_NAME}, {self.COL_FERT_DEF}) VALUES '
        return sql + self.sql_insert_tuple(fert) + ';'

    def sql_insert_tuple(self, fert:Fertilizer) -> str:
        fstr = json.dumps(fert.get_dict_repr())
        return f'(\'{fert.name}\', \'{fstr}\')'

    def default_fertilizers(self) -> list[Fertilizer]:
        nac = Fertilizer()
        nac.name = 'NAC'
        nac.NO3 = 0.135
        nac.NH4 = 0.135
        nac.CaO = 0.115
        nac.CaO_sol = 0.065

        complex_01 = Fertilizer()
        complex_01.name = 'Complex 15/15/15 + 8S + Zn'
        complex_01.NO3 = 0.06
        complex_01.NH4 = 0.09
        complex_01.P2O5 = 0.15
        complex_01.P2O5_sol = 0.135
        complex_01.K2O_sol = 0.15
        complex_01.SO3_sol = 0.08
        complex_01.Zn = 1e-4

        urea = Fertilizer()
        urea.name = 'Urea 30N'
        urea.NH4 = 0.3

        return [nac, complex_01, urea]
