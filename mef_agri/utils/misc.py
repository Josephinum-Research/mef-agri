import os
import inspect
import datetime

class PixelUnits:
    DIFF = 'different'  # if there are different types of values in the layers/channels of a GeoRaster/Image
    UINT8 = 'uint8'  # [0, 255]
    UINT16 = 'uint16'  # [0, 65535]
    UINT32 = 'uint32'
    INT8 = 'int8'
    INT16 = 'int16'
    INT32 = 'int32'
    FLOAT32 = 'float32'
    FLOAT64 = 'float64'
    REFL = 'reflectance'  # ratio of reflected/incident and total/emitted energy - normally [0, 1]
    REFL_INT = 'refl_int'  # e.g. from sentinelhub: reflactance * 10000


def set_attributes(obj, attr_dict):
    for key, val in attr_dict.items():
        if hasattr(obj, key):
            setattr(obj, key, val)


def get_decorated_methods(
        obj:object, decorators:list[str], iterate_super_class=None,
        exclude_intermediate_classes:list=None
    ):
    """
    Function which returns the names of methods which are decorated with a 
    decorator contained in ``decorators``. 
    ``iterate_super_class`` specifies the base-class of ``obj`` - if provided, 
    also the intermediate classes will be screened for the provided 
    ``decorators``.
    For clarification:
    
    * base-class (i.e. ``iterate_super_class``)

        * NOT screened

    * intermediate class 1 (inherits from base-class)

        * screened 

    * ...

        *  all screened

    * intermediate class n (inherits from intermediate class n-1)

        * screened

    * derived class (i.e. ``obj``)

        * screened

    :param obj: object from which class-definition (i.e. source code) the decorated methods should be retrieved
    :type obj: object
    :param decorators: decorator names to search for
    :type decorators: list[str]
    :param iterate_super_class: base-class which childs should be also screened, defaults to None
    :type iterate_super_class: class, optional
    :param exclude_intermediate_classes: intermediate classes which should not be screened, defaults to None
    :type exclude_intermediate_classes: tuple, optional
    """
    def loop(cls, ret):
        ret += inspect.getsource(cls)
        if iterate_super_class is None:
            return ret
        else:
            if exclude_intermediate_classes is None:
                noscreen = [iterate_super_class]
            else:
                noscreen = [iterate_super_class] + exclude_intermediate_classes
        
        for supcls in cls.__bases__:
            if issubclass(supcls, iterate_super_class):
                if supcls in noscreen:
                    pass
                else:
                    ret = loop(supcls, ret)
        return ret

    sc = ''
    sc = loop(obj.__class__, sc)
    decos = []
    for dec  in decorators:
        for part in sc.split(dec)[1:]:
            decos.append(part.split('def ')[1].split('(self')[0].strip())
    return decos


def search_files(directory:str, fformat:str) -> list[str]:
    if fformat[0] != '.':
        fformat = '.' + fformat
    lenff = len(fformat)
    files = []
    for cont in os.listdir(directory):
        if len(cont) <= lenff:
            continue
        if cont[-lenff:] == fformat:
            files.append(cont)
    return files


def search_file(directory:str, fformat:str, which:str='first') -> str | None:
    files = search_files(directory, fformat)
    if not files:
        return None
    if which == 'first':
        return files[0]
    

def check_date(val) -> datetime.date:
    """
    Ensure, that the returned value is of type ``datetime.date``.
    Possible data-types for ``val``

    * ``datetime.date`` - ``val`` will be returned as is
    * ``datetime.datetime`` - date-part will be returned from ``val``
    * iso-formatted string - ``val`` will be converted to ``datetime.date``

    :param val: value which should be checked
    :type val: any
    :raises ValueError: if ``val`` cannot be converted to ``dateitime.date``
    :return: ``datetime.date`` instance
    :rtype: datetime.date
    """
    if isinstance(val, datetime.datetime):
        return val.date()
    elif isinstance(val, str):
        return datetime.date.fromisoformat(val)
    elif isinstance(val, datetime.date):
        return val
    else:
        msg = 'Provided value cannot be converted to datetime.date!'
        raise ValueError(msg)
    

def check_time(val) -> datetime.time:
    """
    Ensure, that the returned value is of type ``datetime.time``.
    Possible data-types for ``val``

    * ``datetime.time`` - ``val`` will be returned as is
    * ``datetime.datetime`` - time-part will be returned from ``val``
    * iso-formatted string - ``val`` will be converted to ``datetime.time``

    :param val: value which should be checked
    :type val: any
    :raises ValueError: if ``val`` cannot be converted to ``dateitime.time``
    :return: ``datetime.time`` instance
    :rtype: datetime.time
    """
    if isinstance(val, datetime.datetime):
        return val.time()
    elif isinstance(val, str):
        return datetime.time.fromisoformat(val)
    elif isinstance(val, datetime.time):
        return val
    else:
        msg = 'Provided value cannot be converted to datetime.time!'
        raise ValueError(msg)
