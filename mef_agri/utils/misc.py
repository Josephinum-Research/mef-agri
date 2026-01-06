import inspect

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
            return
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
