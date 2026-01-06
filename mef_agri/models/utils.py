import numpy as np
from datetime import date

from ..evaluation.stats_utils import RVSampler


def day_of_year(epoch:date) -> int:
    """
    :param epoch: current epoch
    :type epoch: datetime.date
    :return: n-th day of the year
    :rtype: int
    """
    return epoch.timetuple().tm_yday


class __UNITS__:
    """
    Helper class which holds the string values defining a certain unit. It is 
    recommended to use/import ``Units`` from this module - being an instance of 
    this class - which holds the initialized properties.
    """

    @property
    def boolean(self) -> str:
        """
        :return: boolean value
        :rtype: str
        """
        return '[ bool ]'

    @property
    def s(self) -> str:
        r"""
        :return: seconds :math:`s`
        :rtype: str
        """
        return '[ s ]'
    
    @property
    def h(self) -> str:
        r"""
        :return: hours :math:`h`
        :rtype: str
        """
        return '[ h ]'
    
    @property
    def day(self) -> str:
        r"""
        :return: days :math:`day`
        :rtype: str
        """
        return '[ day ]'
    
    @property
    def m(self) -> str:
        r"""
        :return: meters :math:`m`
        :rtype: str
        """
        return '[ m ]'

    @property
    def cm(self) -> str:
        r"""
        :return: centimeters :math:`cm`
        :rtype: str
        """
        return '[ cm ]'

    @property
    def mm(self) -> str:
        r"""
        :return: millimeters :math:`mm`
        :rtype: str
        """
        return '[ mm ]'
    
    @property
    def g(self) -> str:
        """
        :return: gram (mass) :math:`g`
        :rtype: str
        """
        return '[ g ]'
    
    @property
    def kg(self) -> str:
        """
        :return: kilogram (mass) :math:`kg`
        :rtype: str
        """
        return '[ kg ]'
    
    @property
    def t(self) -> str:
        """
        :return: ton (mass) :math:`t`
        :rtype: str
        """
        return '[ t ]'

    @property
    def rad(self) -> str:
        r"""
        :return: angular radians :math:`rad`
        :rtype: str
        """
        return '[ rad ]'
    
    @property
    def deg(self) -> str:
        r"""
        :return: angular degrees :math:`^\circ`
        :rtype: str
        """
        return '[ deg ]'

    @property
    def degC(self) -> str:
        r"""
        :return: degree Celsius :math:`^\circ C`
        :rtype: str
        """
        return '[ degC ]'
    
    @property
    def degK(self) -> str:
        r"""
        :return: degree Kelvin :math:`^\circ K`
        :rtype: str
        """
        return '[ degK ]'
    
    @property
    def n_degC(self) -> str:
        r"""
        :return: per degree Celsius :math:`\frac{1}{^\circ C}`
        :rtype: str
        """
        return '[ ( 1 ) / ( degC ) ]'
    
    @property
    def degCday(self) -> str:
        r"""
        :return: thermal time :math:`^\circ C\cdot day`
        :rtype: str
        """
        return '[ degC x day ]'
    
    @property
    def mm_day(self) -> str:
        r"""
        :return: mm per day (e.g. hydraulic conductivity) :math:`\frac{mm}{day}`
        :rtype: str
        """
        return '[ ( mm ) / ( day ) ]'

    @property
    def g_cm3(self) -> str:
        r"""
        :return: bulk density :math:`\frac{g}{cm^3}`
        :rtype: str
        """
        return '[ ( g ) / ( cm3 ) ]'
    
    @property
    def kg_m3(self) -> str:
        r"""
        :return: bulk density :math:`\frac{kg}{m^3}`
        :rtype: str
        """
        return '[ ( kg ) / ( m3 ) ]'
    
    @property
    def mm_m2xday(self) -> str:
        r"""
        :return: mm per square meter and day (e.g. precipitation) :math:`\frac{mm}{m^2\cdot day}`
        :rtype: str
        """
        return '[ ( mm ) / ( m2 x day ) ]'
    
    @property
    def m_s(self) -> str:
        r"""
        :return: velocity in meters per second :math:`\frac{m}{s}`
        :rtype: str
        """
        return '[ ( m ) / ( s ) ]'
    
    @property
    def s_m(self) -> str:
        r"""
        :return: used for resistance term in evapotranspiration :math:`\frac{s}{m}`
        :rtype: str
        """
        return '[ ( s ) / ( m ) ]'
    
    @property
    def MJ_m2(self) -> str:
        r"""
        :return: Mega Joule per square meter :math:`\frac{MJ}{m^2}`
        :rtype: str
        """
        return '[ ( MJ ) / ( m2 ) ]'
    
    @property
    def MJ_m2xday(self) -> str:
        r"""
        :return: Mega Joule per square meter :math:`\frac{MJ}{m^2\cdot day}`
        :rtype: str
        """
        return '[ ( MJ ) / ( m2 x day ) ]'
    
    @property
    def MJ_kg(self) -> str:
        r"""
        :return: Mega Joule per kg :math:`\frac{MJ}{kg}`
        :rtype: str
        """
        return '[ ( MJ ) / ( kg ) ]'
    
    @property
    def W_m2(self) -> str:
        r"""
        :return: Watt per square meter :math:`\frac{W}{m^2}`
        :rtype: str
        """
        return '[ ( W ) / ( m2 ) ]'
    
    @property
    def Wh_m2xday(self) -> str:
        r"""
        :return: Watt hours per square meter and day :math:`\frac{Wh}{m^2\cdot day}`
        :rtype: str
        """
        return '[ ( Wh ) / ( m2 x day ) ]'
    
    @property
    def kgxm2_haxMJxday(self) -> str:
        r"""
        :return: unit used for factor to convert energy to biomass :math:`\frac{kg\cdot m^2}{MJ\cdot ha\cdot day}`
        :rtype: str
        """
    
    @property
    def kPa(self) -> str:
        r"""
        :return: pressure in kilo Pascal :math:`kPa`
        :rtype: str
        """
        return '[ kPa ]'
    
    @property
    def kPa_degC(self) -> str:
        r"""
        :return: kilo Pascal per degree Celsius (e.g. psychrometric constant) :math:`\frac{kPa}{^\circ C}`
        :rtype: str
        """
        return '[ ( kPa ) / ( degC ) ]'
    
    @property
    def kg_ha(self) -> str:
        r"""
        :return: kilogram per hectar :math:`\frac{kg}{ha}`
        :rtype: str
        """
        return '[ ( kg ) / ( ha ) ]'
    
    @property
    def kg_haxday(self) -> str:
        r"""
        :return: kilogram per hectar and day :math:`\frac{kg}{ha\cdot day}`
        :rtype: str
        """
        return '[ ( kg ) / ( ha x day ) ]'
    
    @property
    def t_ha(self) -> str:
        r"""
        :return: tons per hectar :math:`\frac{kg}{ha}`
        :rtype: str
        """
        return '[ ( t ) / ( ha ) ]'
    
    @property
    def g_m2(self) -> str:
        r"""
        :return: gram per square meter :math:`\frac{g}{m^2}`
        :rtype: str
        """
        return '[ ( g ) / ( m2 ) ]'
    
    @property
    def kg_m2(self) -> str:
        r"""
        :return: kilogram per square meter :math:`\frac{g}{m^2}`
        :rtype: str
        """
        return '[ ( kg ) / ( m2 ) ]'
    
    @property
    def kg_t(self) -> str:
        r"""
        :return: kg nutrient per ton biomass :math:`\frac{kg}{t}`
        :rtype: str
        """
        return '[ ( kg ) / ( t ) ]'
    
    @property
    def t_haxday(self) -> str:
        r"""
        :return: tons per hectar and day :math:`\frac{t}{ha\cdot day}`
        :rtype: str
        """
        return '[ ( t ) / ( ha x day ) ]'

    @property
    def n_m2(self) -> str:
        r"""
        :return: amount/number per square meter :math:`\frac{1}{m^2}`
        :rtype: str
        """
        return '[ ( 1 ) / ( m2 ) ]'
    
    @property
    def n_ha(self) -> str:
        r"""
        :return: amount/number per hectar :math:`\frac{1}{ha}`
        :rtype: str
        """
        return '[ ( 1 ) / ( ha ) ]'
    
    @property
    def per_day(self) -> str:
        r"""
        :return: daily unitless rates :math:`\frac{1}{day}`
        :rtype: str
        """
        return '[ ( 1 ) / ( day ) ]'
    
    @property
    def g_ha(self) -> str:
        r"""
        :return: gram per hectar :math:`\frac{g}{ha}`
        :rtype: str
        """
        return '[ ( g ) / ( ha ) ]'
    
    @property
    def l_ha(self) -> str:
        r"""
        :return: liters per hectar :math:`\frac{l}{ha}`
        :rtype: str
        """
        return '[ ( l ) / ( ha ) ]'
    
    @property
    def frac(self) -> str:
        r"""
        :return: fraction (usally corresponds to percent, in the range [0, 1])
        :rtype: str
        """
        return '[ ]'
    
    @property
    def perc(self) -> str:
        r"""
        :return: percent
        :rtype: str
        """
        return '[ & ]'
    
    @property
    def none(self) -> str:
        """
        :return: no units, i.e. unitless
        :rtype: str
        """
        return '[ ]'
    
    @property
    def undef(self) -> str:
        """
        :return: unit not defined
        :rtype: str
        """
        return 'undefined'
    
    def convert(self, value, u_source:str, u_target:str):
        """
        Method to convert physical units.

        :param value: value which should be converted
        :type value: float or numpy.ndarray
        :param u_source: unit of ``value``
        :type u_source: str
        :param u_target: target unit, to which the value should be converted
        :type u_target: str
        :return: converted value
        :rtype: float or numpy.ndarray
        """
        if u_source == u_target:
            return value
        ########################################################################
        # DISTANCE CONVERSIONS
        elif u_source == self.m and u_target == self.cm:
            return value * 100.0
        elif u_source == self.cm and u_target == self.m:
            return value  / 100.0
        elif u_source == self.m and u_target == self.mm:
            return value * 1000.0
        elif u_source == self.mm and u_target == self.m:
            return value / 1000.0
        elif u_source == self.cm and u_target == self.mm:
            return value * 10.0
        elif u_source == self.mm and u_target == self.cm:
            return value / 10.0
        ########################################################################
        # AREA CONVERSIONS
        elif u_source == self.n_m2 and u_target == self.n_ha:
            return value * 1e4
        elif u_source == self.n_ha and u_target == self.n_m2:
            return value / 1e4
        ########################################################################
        # MASS CONVERSIONS
        elif u_source == self.g and u_target == self.kg:
            return value * 1e-3
        elif u_source == self.g and u_target == self.t:
            return value * 1e-6
        elif u_source == self.kg and u_target == self.g:
            return value * 1e3
        elif u_source == self.kg and u_target == self.t:
            return value * 1e-3
        elif u_source == self.t and u_target == self.g:
            return value * 1e6
        elif u_source == self.t and u_target == self.kg:
            return value * 1e3
        elif u_source == self.g_ha and u_target == self.kg_ha:
            return value * 1e-3
        elif u_source == self.g_ha and u_target == self.t_ha:
            return value * 1e-6
        elif u_source == self.kg_ha and u_target == self.g_ha:
            return value * 1e3
        elif u_source == self.kg_ha and u_target == self.t_ha:
            return value * 1e-3
        elif u_source == self.t_ha and u_target == self.g_ha:
            return value * 1e6
        elif u_source == self.t_ha and u_target == self.kg_ha:
            return value * 1e3
        elif u_source == self.g_m2 and u_target == self.kg_m2:
            return value * 1e-3
        elif u_source == self.kg_m2 and u_target == self.g_m2:
            return value * 1e3
        elif u_source == self.g_m2 and u_target == self.g_ha:
            return value * 1e4
        elif u_source == self.g_m2 and u_target == self.kg_ha:
            return value * 1e4 * 1e-3
        elif u_source == self.g_m2 and u_target == self.t_ha:
            return value * 1e4 * 1e-6
        elif u_source == self.kg_m2 and u_target == self.g_ha:
            return value * 1e4 * 1e3
        elif u_source == self.kg_m2 and u_target == self.kg_ha:
            return value * 1e4
        elif u_source == self.kg_m2 and u_target == self.t_ha:
            return value * 1e4 * 1e-3
        elif u_source == self.g_ha and u_target == self.g_m2:
            return value * 1e-4
        elif u_source == self.g_ha and u_target == self.kg_m2:
            return value * 1e-4 * 1e-3
        elif u_source == self.kg_ha and u_target == self.g_m2:
            return value * 1e-4 * 1e3
        elif u_source == self.kg_ha and u_target == self.kg_m2:
            return value * 1e-4
        elif u_source == self.t_ha and u_target == self.g_m2:
            return value * 1e-4 * 1e6
        elif u_source == self.t_ha and u_target == self.kg_m2:
            return value * 1e-4 * 1e3
        ########################################################################
        # TIME CONVERSIONS
        elif u_source == self.s and u_target == self.h:
            return value / 3600.0
        elif u_source == self.h and u_target == self.s:
            return value * 3600.0
        elif u_source == self.s and u_target == self.day:
            return value / 86400.0
        elif u_source == self.day and u_target == self.s:
            return value * 86400.0
        elif u_source == self.h and u_target == self.day:
            return value / 24.0
        elif u_source == self.day and u_target == self.h:
            return value * 24.0
        ########################################################################
        # (BULK) DENSITY CONVERSIONS
        elif u_source == self.g_cm3 and u_target == self.kg_m3:
            return value * 1e3
        elif u_source == self.kg_m3 and u_target == self.g_cm3:
            return value * 1e-3
        ########################################################################
        # ENERGY CONVERSIONS
        elif u_source in (self.MJ_m2, self.MJ_m2xday) and u_target == self.W_m2:
            return value / 0.0864
        elif u_source == self.W_m2 and u_target in (self.MJ_m2, self.MJ_m2xday):
            return value * 0.0864
        elif u_source == self.Wh_m2xday and u_target == self.MJ_m2xday:
            return value * 0.0036
        elif u_source == self.MJ_m2xday and u_target == self.Wxh_m2xday:
            return value / 0.0036
        elif u_source == self.Wh_m2xday and u_target == self.W_m2:
            return value / 24.0
        elif u_source == self.W_m2 and u_target == self.Wh_m2xday:
            return value * 24.0
        ########################################################################
        # TEMPERATURE CONVERSIONS
        elif u_source == self.degC and u_target == self.degK:
            return value + 273.15
        elif u_source == self.degK and u_target == self.degC:
            return value - 273.15
        ########################################################################
        # (BIO)MASS CONVERSIONS
        elif u_source == self.kg_ha and u_target == self.t_ha:
            return value * 1e-3
        elif u_source == self.t_ha and u_target == self.kg_ha:
            return value * 1e3
        elif u_source == self.kg_haxday and u_target == self.t_haxday:
            return value  * 1e-3
        elif u_source == self.t_haxday and u_target == self.kg_haxday:
            return value * 1e3
        ########################################################################
        # PERCENT STUFF
        elif u_source == self.perc and u_target == self.frac:
            return value / 100.0
        elif u_source == self.frac and u_target == self.perc:
            return value * 100.0
        # CONVERSION NOT POSSIBLE
        else:
            msg = 'Conversion not possible (or not implemented) for the '
            msg += 'provided units {} => {}!'.format(u_source, u_target)
            raise ValueError(msg)
        
# initialization of __UNITS__ class such that properties are available
Units = __UNITS__()
################################################################################

class PFunction(object):
    """
    This class represents a function which acts as parameter in the 
    models. The object is callable and returns the output value based on the 
    provided input value. The input value can be a numeric value, a 1D 
    numpy.ndarray with length 1 or with length equal to the number of 
    particles representing the distributions.

    The following function types are supported:

    **PIECEWISE_LINEAR**

    Is defined by a set of x- and y-values (equal length!). 
    The ``fdef`` dictionary of :func:`define` contains the 
    following keys

    - **ftype** - ``str`` specifying the function type, see :class:`FTYPE`
    - **values-x** - array of values on the x-axis of the function (i.e. the input values), used as mean/mode value for the distribution
    - **values-y** - array of values on the y-axis of the function (i.e. the output values), used as mean/mode value for the distribution
    - **distr-x** - distribution of the x-values, dict which contains the `distr_id`, and further distribution parameters according to `stats_utils`, the assumption is, that the distribution is equal for all x-values
    - **distr-y** - distribution of the y-values, dict which contains the `distr_id`, and further distribution parameters according to `stats_utils`, the assumption is, that the distribution is equal for all y-values
    - **sample** - specifies if pfunction should be sampled or not

    **values-x** and **values-y** are necessary, the other ones are optional 

    """
    class FTYPE:
        """
        Helper class which holds the string values (class variables) to define 
        the type of a hyper parametric function

        * *PIECEWISE_LINEAR* = ``'piecewise-linear'``

        """
        PIECEWISE_LINEAR = 'piecewise-linear'

    class PiecewiseLinear(object):
        """
        Class representing a piecewise linear function. Based on the provided 
        x- and y-values, intervals are defined, in which a linear function is 
        used to interpolate values.

        :param fdef: dictionary containing information about the function (see :class:`PFunction`)
        :type fdef: dict
        """
        def __init__(self, fdef:dict) -> None:
            if not 'values-x' in fdef.keys() or not 'values-y' in fdef.keys():
                msg = 'PFunction.PiecewiseLinear: '
                msg += '`values-x` and `values-y` have to be provided in the '
                msg += '`fdef`-dictionary!'
                raise ValueError(msg)
            
            self._x = np.atleast_2d(np.array(fdef['values-x']))
            self._y = np.atleast_2d(np.array(fdef['values-y']))
            self._dx = fdef['distr-x']
            self._dy = fdef['distr-y']
            self._sampled:bool = False

        def sample(self, rvs:RVSampler, nsamples:int) -> None:
            r"""
            If the pfunction is sampled, the arrays of x- and y-values with 
            length :math:`n` (values in ``fdef`` provided to ``__init__()``) are 
            sampled as often as specified by ``nsamples``. 
            Thus, the arrays of x- and y-values (representing the supporting 
            points) exhibit the dimensions :math:`nsamples\times n` instead of 
            :math:`1\times n`.

            :param rvs: TODO
            :type rvs: RVSampler
            :param nsamples: number of samples
            :type nsamples: int
            """
            xrv = np.zeros((nsamples, self._x.shape[1]), dtype=float)
            yrv = np.zeros((nsamples, self._y.shape[1]), dtype=float)
            for i in range(self._x.shape[1]):
                xrv[:, i] = rvs.get_sampled_values(
                    self._x[0, i], self._dx, nsamples
                )
                yrv[:, i] = rvs.get_sampled_values(
                    self._y[0, i], self._dy, nsamples
                )
            self._x = xrv
            self._y = yrv
            self._sampled = True

        def compute(self, value:np.ndarray) -> np.ndarray:
            r"""
            Compute interpolated values. In the case, that the pfunction is 
            sampled, each value in ``value`` is used for interpolation together 
            with one sample of supporting points (i.e. one row of the 
            :math:`nsampels\times n` arrays representing the x- and y-values).

            :param value: x-values for which y-values should be interpolated - should be a one-dimensional array
            :type value: numpy.ndarray
            :raises ValueError: if ``value`` has more than one dimension
            :raises ValueError: if ``len(value)`` does not match ``nsamples`` of :func:`sample`
            :return: interpolated values
            :rtype: numpy.ndarray
            """
            if len(value.shape) > 1:
                msg = 'Input value to pfunction should be a 1D numpy.ndarray!'
                raise ValueError(msg)
            
            if self._sampled:
                if self._x.shape[0] != value.shape[0]:
                    msg = 'Number of samples of pfunction does not match the '
                    msg += 'number of samples in the provided array of values!'
                    raise ValueError(msg)

                xvs = np.zeros(value.shape, dtype=float)
                for i in range(self._x.shape[0]):
                    xvs[i] = np.interp(value[i], self._x[i, :], self._y[i, :])
            else:
                xvs = np.interp(value, self._x[0, :], self._y[0, :])
            return xvs

    def __init__(self) -> None:
        self._func = None
        self._val:np.ndarray = None
        self._sampled:bool = False

    @property
    def current_value(self) -> np.ndarray:
        """
        :return: current values computed with PFunction definition and input values
        :rtype: numpy.ndarray
        """
        return self._val
    
    @property
    def is_sampled(self) -> bool:
        """
        :return: flag if pfunction is sampled
        :rtype: bool
        """
        return self._sampled

    def define(self, fdef:dict) -> None:
        """
        Define the hyper-parametric function.

        :param fdef: dictionary containing information about the function
        :type fdef: dict
        :raises ValueError: if ``'ftype'`` in ``fdef`` is not supporte by this class
        """
        if fdef['ftype'] == self.FTYPE.PIECEWISE_LINEAR:
            self._func = self.PiecewiseLinear(fdef)
        else:
            msg = 'PFunction: provided function-type not supported!'
            raise ValueError(msg)
    
    def sample(self, rvs:RVSampler, nsamples:int) -> None:
        """
        TODO

        :param rvs: _description_
        :type rvs: RVSampler
        :param nsamples: _description_
        :type nsamples: int
        """
        self._sampled = True
        self._func.sample(rvs, nsamples)
        
    def __call__(self, value) -> np.ndarray:
        self._val = self._func.compute(value)
        return self._val
    