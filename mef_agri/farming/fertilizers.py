

class Fertilizer(object):
    """
    Class representing a fertilizer. If a specific fertilizer needs to be 
    created, just initialize this class and set the nutrient properties being 
    fractions (i.e. range of possible values is [0, 1]).
    """
    def __init__(self):
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
    def N_total(self) -> float:
        """
        :return: overall fraction of nitrogen within a specified amount of fertilizer (NO3 + NH4)
        :rtype: float
        """
        return self._no3 + self._nh4

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

    
NAC = Fertilizer()
NAC.NO3 = 0.135
NAC.NH4 = 0.135
NAC.CaO = 0.115
NAC.CaO_sol = 0.065


Complex_15_15_15_8S_Zn = Fertilizer()
Complex_15_15_15_8S_Zn.NO3 = 0.06
Complex_15_15_15_8S_Zn.NH4 = 0.09
Complex_15_15_15_8S_Zn.P2O5 = 0.15
Complex_15_15_15_8S_Zn.P2O5_sol = 0.135
Complex_15_15_15_8S_Zn.K2O_sol = 0.15
Complex_15_15_15_8S_Zn.SO3_sol = 0.08
Complex_15_15_15_8S_Zn.Zn = 1e-4


Petiso = Fertilizer()
Petiso.NH4 = 0.12
Petiso.NO3 = 0.12
Petiso.SO3 = 0.12
Petiso.SO3_sol = 0.12
Petiso.CaO = 0.09
Petiso.CaO_sol = 0.09


Ensin = Fertilizer()
Ensin.NO3 = 0.075
Ensin.NH4 = 0.185
Ensin.SO3 = 0.325
Ensin.SO3_sol = 0.325


DusLAS = Fertilizer()
DusLAS.NO3 = 0.12
DusLAS.NH4 = 0.12
DusLAS.SO3 = 0.15
DusLAS.CaO = 0.11


Cultan_17N = Fertilizer
Cultan_17N.NH4 = 0.17
Cultan_17N.NO3 = 0.0
