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
