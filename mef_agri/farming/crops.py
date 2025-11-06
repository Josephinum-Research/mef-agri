from functools import wraps

def cultivar(func):
    """
    Decorator to specify cultivars in child classes of :class:`crop`
    """
    @wraps(func)
    def wrapper(obj):
        if not func.__name__ in obj._cs:
            obj._cs.append(func.__name__)
            setattr(obj, func.__name__ + '__info__', {})
        return func(obj)
    return wrapper


class Crop(object):
    """
    Crop base class. If specifying a new crop, the corresponding class has to 
    inherit from this one.
    Cultivars are introduced as methods decorated with :func:`cultivar` which
    adds an attribute composed of the method's name and the string 
    ``'__info__'`` (e.g. for the generic cultivar, the attribute is named 
    ``'generic__info__'``). 
    The cultivar methods have to return this attribute being an empty dictionary 
    by default. 
    If there are cultivar informations or parameters available from external 
    data sources, these can be added to these dictionaries. 
    """
    def __init__(self):
        self._cs = []

    @property
    def cultivars(self) -> list[str]:
        """
        :return: list of available cultivars
        :rtype: list[str]
        """
        return self._cs


class winter_wheat(Crop):
    def __init__(self):
        super().__init__()

    @cultivar
    def generic(self):
        """
        Unspecified cultivar triggering the usage of default values for crop
        input parameters in the evaluation of models
        """
        return self.generic__info__


class winter_barley(Crop):
    def __init__(self):
        super().__init__()

    @cultivar
    def generic(self):
        """
        Unspecified cultivar triggering the usage of default values for crop
        input parameters in the evaluation of models
        """
        return self.generic__info__


class maize(Crop):
    def __init__(self):
        super().__init__()

    @cultivar
    def generic(self):
        """
        Unspecified cultivar triggering the usage of default values for crop
        input parameters in the evaluation of models
        """
        return self.generic__info__


class soybean(Crop):
    def __init__(self):
        super().__init__()

    @cultivar
    def generic(self):
        """
        Unspecified cultivar triggering the usage of default values for crop
        input parameters in the evaluation of models
        """
        return self.generic__info__
