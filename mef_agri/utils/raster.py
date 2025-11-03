import os
import json
import datetime
import numpy as np
import rasterio as rio
import rasterio.warp as riowrp

from copy import deepcopy
from shapely.geometry import Polygon
from geopandas import GeoDataFrame

from .misc import PixelUnits
from .gis import (
    affine_from_trafo, trafo_from_affine, get_epsg_from_rio_crs, 
    imgshape_to_rasterio, bbox_from_gdf
)


class ReferencePoint:
    CENTER = 0
    UPPERLEFT = 1
    # UPPERRIGHT = 2
    # LOWERLEFT = 3
    # LOWERRIGHT = 4
    # MIDLEFT = 5  # only for hexagon as raster element
    # MIDRIGHT = 6  # only for hexagon as raster element


class RasterElement(object):
    def __init__(self) -> None:
        self.reference_point = ReferencePoint.UPPERLEFT
        self.edge_length = 1.0  # [m]

    def get_polygon(self, rpx:float, rpy:float) -> Polygon:
        NotImplementedError()


# only square and hexagon for now because they are "fully symmetric" and can be 
# determined from one reference point and one edge length
class Square(RasterElement):
    def __init__(self) -> None:
        super().__init__()

    def get_polygon(self, rpx: float, rpy: float) -> Polygon:
        plgn = None
        if self.reference_point == ReferencePoint.CENTER:
            addd = self.edge_length / 2.0
            plgn = Polygon(shell=(
                [rpx - addd, rpy + addd],
                [rpx + addd, rpy + addd],
                [rpx + addd, rpy - addd],
                [rpx - addd, rpy - addd]
            ))
        elif self.reference_point == ReferencePoint.UPPERLEFT:
            plgn = Polygon(shell=(
                [rpx, rpy],
                [rpx + self.edge_length, rpy],
                [rpx + self.edge_length, rpy - self.edge_length],
                [rpx, rpy - self.edge_length]
            ))
        else:
            NotImplementedError()
        return plgn


class Hexagon(RasterElement):
    """
    Assumption: angles are counted positive in clockwise direction from north/
    y-axis
    """
    def __init__(self) -> None:
        super().__init__()

    def get_polygon(self, rpx: float, rpy: float) -> Polygon:
        plgn = None
        if self.reference_point == ReferencePoint.CENTER:
            hel = 0.5 * self.edge_length
            icr = np.sqrt(3.0) * hel  # inner-circle radius
            plgn = Polygon(shell=(
                [rpx + hel, rpy + icr],
                [rpx + self.edge_length, rpy],
                [rpx + hel, rpy - icr],
                [rpx - hel, rpy - icr],
                [rpx - self.edge_length, rpy],
                [rpx - hel, rpy + icr]
            ))
        else:
            NotImplementedError()
        return plgn


class GeoRaster(object):
    META_FILENAME = 'metadata'
    META_LAYERID_KEY = 'georaster-layer-ids'
    META_LAYERINFO_KEY = 'georaster-layer-infos'
    META_DATAFILES_KEY = 'georaster-data-files'
    META_LAYERIX_KEY = 'georaster-layer-index'

    ERR_INDEX_TYPE = 'Layers are only accessable/indexable by an integer, '
    ERR_INDEX_TYPE += 'a string contained in `layer_ids`, a slice or a tuple '
    ERR_INDEX_TYPE += 'containing integers or strings!'

    ERR_IXTPLVAL_TYPE = 'Values in tuple to access/index layers have to be '
    ERR_IXTPLVAL_TYPE += 'integers or strings!'
    
    def __init__(self) -> None:
        """
        Base class for all raster data in `mef_agri` package.
        """
        self._rstr:np.ndarray = None
        self._rstr_el:RasterElement = None
        self._shp:tuple = None
        self._bbox:tuple = None
        self._nodata = None
        self._units:PixelUnits = None
        self._trf:np.ndarray = None
        self._trf_inv:np.ndarray = None
        self._meta = {}

        self._reset_iteration()

    ############################################################################
    # ITERATOR ATTRIBUTES
    @property
    def current_row(self) -> int:
        if self._crst is None:
            raise ValueError('This value is only available in an iteration!')
        return self._crst[1, self._i1]
    
    @property
    def current_column(self) -> int:
        if self._crst is None:
            raise ValueError('This value is only available in an iteration!')
        return self._crst[0, self._i1]

    @property
    def current_object_coordinates(self) -> tuple[float, float]:
        if self._crst is None:
            raise ValueError('This value is only available in an iteration!')
        return self.get_object_coordinates(
            self._crst[1, self._i1], self._crst[0, self._i1]
        )

    @property
    def nextcol_object_coordinates(self) -> tuple[float, float]:
        if self._crst is None:
            raise ValueError('This value is only available in an iteration!')
        return self.get_object_coordinates(
            self._crst[1, self._i2], self._crst[0, self._i2]
        )

    @property
    def nextrow_object_coordinates(self) -> tuple[float, float]:
        if self._crst is None:
            raise ValueError('This value is only available in an iteration!')
        return self.get_object_coordinates(
            self._crst[1, self._i3], self._crst[0, self._i3]
        )

    @property
    def diagonal_object_coordinates(self) -> tuple[float, float]:
        if self._crst is None:
            raise ValueError('This value is only available in an iteration!')
        return self.get_object_coordinates(
            self._crst[1, self._i4], self._crst[0, self._i4]
        )

    @property
    def current_polygon(self) -> Polygon:
        ret = None
        if self.raster_element is None:
            ret = Polygon((
                self.current_object_coordinates,
                self.nextcol_object_coordinates,
                self.diagonal_object_coordinates,
                self.nextrow_object_coordinates
            ))
        else:
            ret = self.raster_element.get_polygon(
                *self.current_object_coordinates
            )
        return ret
    
    ############################################################################
    # ATTRIBUTES with raster and data infos
    @property
    def raster(self) -> np.ndarray:
        """
        Returns the array representing the raster data.

        :return: raster data (settable)
        :rtype: numpy.ndarray
        """
        return self._rstr
    
    @raster.setter
    def raster(self, val):
        if not isinstance(val, np.ndarray):
            raise ValueError(
                'Provided raster data is not of type numpy.ndarray!'
            )
        elif len(val.shape) != 3:
            raise ValueError(
                'Provided raster data has to have three dimensions!'
            )
        self._rstr = val
    
    @property
    def raster_element(self) -> RasterElement:
        """
        Returns an instance of a `RasterElement` child class (Square, Rectangle, 
        Hexagon) which can be used to derive the corresponding shapely geometry 
        (i.e. a Polygon). If `None`, no raster element is defined and user has 
        to decide how ot proceed.

        :return: child class of `RasterElement` (settable)
        :rtype: RasterElement
        """
        return self._rstr_el
    
    @raster_element.setter
    def raster_element(self, val):
        if not isinstance(val, RasterElement):
            raise ValueError(
                '`raster_element` has to be an instance of `RasterElement`'
            )
        self._rstr_el = val

    @property
    def crs(self) -> int:
        """
        Coordinate reference system of the raster.

        :return: EPSG Code (settable)
        :rtype: int
        """
        return self._crs
    
    @crs.setter
    def crs(self, val):
        if not isinstance(val, int):
            raise ValueError(
                '`crs` has to be an integer representing a valid EPSG-code!'
            )
        self._crs = val

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """
        Minimum and maximum of x- and y-coordinates.

        :return: bounds or bbox with order: minx, miny, maxx, maxy (settable)
        :rtype: tuple[float, float, float, float]
        """
        return self._bbox
    
    @bounds.setter
    def bounds(self, val):
        if not isinstance(val, tuple) and not isinstance(val, list):
            msg = '`bounds` has to be a tuple or list containing min. and max. '
            msg += 'coordinate values representing the bounding box!'
            raise ValueError(msg)
        elif len(val) != 4:
            msg = '`bounds` has to contain the min. and max. values in both '
            msg += 'coordinate-axes (i.e. four values)!'
            raise ValueError(msg)
        elif val[0] > val[2]:
            msg = 'Provided minimum x-coordinate is bigger than the '
            msg += 'maximum x-coordinate'
            raise ValueError(msg)
        elif val[1] > val[3]:
            msg = 'Provided minimum y-coordinate is bigger than the '
            msg += 'maximum y-coordinate'
            raise ValueError(msg)
        self._bbox = val
    
    @property
    def extent(self) -> tuple[float, float, float, float]:
        """
        Minimum and maximum of x- and y-coordinates with different order 
        compared to `self.bounds`.

        :return: bounds or bbox with order: minx, maxx, miny, maxy
        :rtype: tuple[float, float, float, float]
        """
        return (self.bounds[0], self.bounds[2], self.bounds[1], self.bounds[3])
    
    @property
    def units(self) -> PixelUnits:
        """
        The units of the pixels.

        :return: pixel units (settable)
        :rtype: mef_agri.utils.misc.PixelUnits
        """
        return self._units
    
    @units.setter
    def units(self, val):
        self._units = val
    
    @property
    def metadata(self) -> dict:
        """
        Additional metadata about the raster.

        :return: dictionary containing additional metadata 
        :rtype: dict
        """
        if self._meta is None:
            self._meta = {}
        return self._meta
    
    @property
    def transformation(self) -> np.ndarray:
        """
        3x3 transformation-matrix for georeferencing the image/raster containing 
        scale, offset and rotation.

        :return: 3x3 matrix describing georeference, i.e. transformation from image to object space (settable)
        :rtype: np.ndarray
        """
        return self._trf
    
    @transformation.setter
    def transformation(self, val):
        if not isinstance(val, np.ndarray):
            msg = '`transformation` has to be of type `numpy.ndarray`!'
            raise ValueError(msg)
        self._trf = val
    
    @property
    def nodata_value(self) -> any:
        """
        The nodata-value for geotiff raster

        :return: nodata value (settable)
        :rtype: any
        """
        return self._nodata
    
    @nodata_value.setter
    def nodata_value(self, val):
        self._nodata = val
    
    @property
    def raster_shape(self) -> tuple[int, int, int]:
        """
        The shape of the raster. Has to be 3D! This attribute has to be provided 
        by the user and is not derived from the data array, because the shape is 
        a tag in the metadata of the GeoTIFF specification.

        :return: rster shape (settable)
        :rtype: tuple
        """
        return self._shp
    
    @raster_shape.setter
    def raster_shape(self, val):
        if not isinstance(val, tuple) and not isinstance(val, list):
            msg = 'Shape of raster has to be a tuple or list containing the '
            msg += 'raster dimensions (number of layers, number of pixels in '
            msg += 'x- and y-direction)!'
            raise ValueError(msg)
        elif len(val) != 3:
            msg = 'Shape of raster has to be a tuple or list containing the '
            msg += 'raster dimensions (number of layers, number of pixels in '
            msg += 'x- and y-direction)!'
            raise ValueError(msg)
        self._shp = val
    
    @property
    def layer_index(self) -> int:
        """
        Index at which the channels of the raster are available. `rasterio` 
        convention is that the first dimension (i.e. 0) of the raster 
        corresponds to the channels. Can be also the third dimension (i.e. 2).

        :return: channel index (settable)
        :rtype: int
        """
        if not self.META_LAYERIX_KEY in self._meta.keys():
            self._meta[self.META_LAYERIX_KEY] = None
        return self._meta[self.META_LAYERIX_KEY]
    
    @layer_index.setter
    def layer_index(self, val):
        if not val in (0, 2):
            msg = 'Channel/Layer index has to be 0 or 2!'
            raise ValueError(msg)
        self._meta[self.META_LAYERIX_KEY] = val

    @property
    def layer_ids(self) -> list[str]:
        """
        Identfiers/names of the individual layers of the GeoRaster.

        :return: list containing layer ids/names (settable)
        :rtype: list[str]
        """
        if not self.META_LAYERID_KEY in self._meta.keys():
            self._meta[self.META_LAYERID_KEY] = []
        return self._meta[self.META_LAYERID_KEY]
    
    @layer_ids.setter
    def layer_ids(self, val):
        if not isinstance(val, list):
            msg = '`layer_ids` has to be a list containing strings!'
            raise ValueError(msg)        
        self._meta[self.META_LAYERID_KEY] = val

    @property
    def layer_infos(self) -> dict:
        """
        Dictionary containing the information about the layers which have been 
        provided by the user.

        :return: dictionary with information of layers
        :rtype: dict
        """
        if not self.META_LAYERINFO_KEY in self._meta.keys():
            self._meta[self.META_LAYERINFO_KEY] = {}
        return self._meta[self.META_LAYERINFO_KEY]
    
    # no implementation in child class ATTRIBUTES
    @property
    def transformation_inv(self) -> np.ndarray:
        """
        Inverse of the transformation matrix which specifies the georeference. 
        It is not necessary to implement it in a child class.

        :return: 3x3 matrix describing georeferenc (object -> image)
        :rtype: np.ndarray
        """
        if self._trf_inv is None:
            self._trf_inv = np.linalg.inv(self.transformation)
        return self._trf_inv
    
    ############################################################################
    # METHODS
    ############################################################################
    def get_raster_coordinates(self, xc:float, yc:float) -> tuple[int, int]:
        """
        Compute the raster/image coordinates (i.e. row and column index of 
        array) from the provided geographic/object coordinates.

        :param xc: x-coordinate (east- or right-direction)
        :type xc: float
        :param yc: y-coordinate (north- or up-direction)
        :type yc: float
        :return: row- and column-index (is also the order in the result tuple)
        :rtype: tuple[int, int]
        """
        ji = self.transformation_inv @ np.array([[xc, yc, 1.0]]).T
        j, i = ji.astype('int')[:2]
        return i, j
    
    def get_raster_coordinates_array(
            self, xc:np.ndarray, yc:np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the raster/image coordinates (i.e. row and column index of 
        array) from the provided geographic/object coordinates.
        Contrary to `get_raster_coordinates()`, this method accepts arrays of 
        geo/object coordinates which are transformed into the image coordinate
        system. If the resolution of the geo/object coordinates is higher than 
        the resolution of this `GeoRaster`, the resulting duplicate image 
        coordinates will be removed.

        :param xc: x-coordinate (east- or right-direction)
        :type xc: np.ndarray
        :param yc: y-coordinate (north- or up-direction)
        :type yc: np.ndarray
        :return: row- and column-indices (is also the order in the result tuple)
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        xy = np.vstack((
            np.atleast_2d(xc), np.atleast_2d(yc), np.ones((1, xc.shape[0]))
        ))
        ji = np.unique(
            (self.transformation_inv @ xy).astype('int')[:2, :], 
            axis=1
        )
        return ji[1, :], ji[0, :]
    
    def get_object_coordinates(self, ri:int, ci:int) -> tuple[float, float]:
        """
        Compute geographic/object coordinates (i.e. x/east/right- 
        and y/north/up-coordinate) from the provided raster/image coordinates.

        :param ri: row index
        :type ri: int
        :param ci: column index
        :type ci: int
        :return: x- and y-coordinate (is also the order of the result tuple)
        :rtype: tuple[float, float]
        """
        xy = self.transformation @ np.array([[ci, ri, 1]]).astype('float').T
        return tuple(xy[:2])
    
    def get_rasterio_dataset(self, filename:str, count:int=1):
        """
        Get a rasterio dataset which can be used to save an image to a geotiff 
        file.

        :param filename: name of the geotiff
        :type filename: str
        :param count: number of channels in the geotiff, defaults to 1
        :type count: int, optional
        :return: file object returned by `rasterio.open(mode='w', ...)`
        :rtype: file-like object
        """
        if self.layer_index == 0:
            ixw, ixh = 2, 1
        elif self.layer_index == 2:
            ixw, ixh = 1, 0
        else:
            raise ValueError('Channel index has to be 0 or 2!')
        return rio.open(
            filename,
            mode='w',
            driver='GTiff',
            count=count,
            dtype=self.units,
            nodata=self.nodata_value,
            height=self.raster_shape[ixh],
            width=self.raster_shape[ixw],
            crs=rio.CRS.from_epsg(self.crs),
            transform=affine_from_trafo(self.transformation)
        )
    
    def save_geotiff(
            self, savepath:str, overwrite:bool=False, compress:bool=False, 
            filename:str=None
        ) -> None:
        """
        Save the raster data to GeoTIFF. Additionally the metadata will be 
        written to .json.

        :param savepath: path where to write file(s)
        :type savepath: str
        :param overwrite: specify if already available files should be overwritten, defaults to False
        :type overwrite: bool, optional
        :param compress: specify if all layers should be written to one file or each layer is written into a separate file, defaults to False
        :type compress: bool, optional
        :param filename: name of the geotiff if all layers are written to one file (i.e. `compress=True`), defaults to None
        :type filename: str, optional
        """
        def _write(data:np.ndarray, nlayers:int, filepath:str, overwrite:bool):
            if os.path.exists(filepath) and not overwrite:
                return
            fio = self.get_rasterio_dataset(filepath, count=nlayers)
            fio.write(data)
            fio.close()

        if not os.path.exists(savepath):
            msg = 'Provided path to save file to does not exist!'
            raise ValueError(msg)
        if self.layer_index is None:
            msg = '`layer_index`-attribute has to be set when writing GeoTIFF!'
            raise ValueError(msg)
        
        # save raster data to geotiff
        self._meta[self.META_DATAFILES_KEY] = []
        if compress:
            if filename is None:
                msg = '`filename` has to be provided if all layers should be '
                msg += 'saved to one geotiff!'
                raise ValueError(msg)
            
            if not filename[-4:] == '.tif':
                filename += '.tif'
            _write(
                self.raster, self.raster_shape[self.layer_index], 
                os.path.join(savepath, filename), overwrite
            )
            self._meta[self.META_DATAFILES_KEY].append(filename)
        else:
            if self.layer_ids is None:
                msg = '`layer_ids`-attribute has to be set when writing each '
                msg += 'layer into a separate file!'
            for i in range(self.raster_shape[self.layer_index]):
                fn = self.layer_ids[i] + '.tif'
                _write(self[i], 1, os.path.join(savepath, fn), overwrite)
                self._meta[self.META_DATAFILES_KEY].append(fn)

        # save metadata to json
        fmeta = os.path.join(savepath, self.META_FILENAME + '.json')
        if os.path.exists(fmeta) and not overwrite:
            return 
        fio = open(fmeta, 'w')
        json.dump(self.metadata, fio, indent=2)
        fio.close()

    def load_geotiff(self, savepath:str) -> None:
        """
        Load raster data from GeoTIFF.

        :param savepath: folder where metadata and raster data is located
        :type savepath: str
        """
        # load metadata
        fmeta = os.path.join(savepath, self.META_FILENAME + '.json')
        if not os.path.exists(fmeta):
            msg = 'No {}.json file in the provided folder!'.format(
                self.META_FILENAME
            )
            raise ValueError(msg)
        fio = open(fmeta, 'r')
        self._meta = json.load(fio)
        fio.close()

        # set properties from metadata
        self._lids = self._meta[self.META_LAYERID_KEY]
        self._lix = self._meta[self.META_LAYERIX_KEY]

        # load raster data
        arr = None
        for fn in self._meta[self.META_DATAFILES_KEY]:
            sdata = rio.open(os.path.join(savepath, fn))
            if arr is None:
                arr = sdata.read()
            else:
                arr = np.concatenate((arr, sdata.read()), axis=self.layer_index)

        # set properties from geotiff information
        self._units = str(arr.dtype)
        self._crs = get_epsg_from_rio_crs(sdata.crs)
        self._nodata = sdata.nodata
        self._trf = trafo_from_affine(sdata.transform)
        self._bbox = list(sdata.bounds)
        self._shp = arr.shape
        self._rstr = arr

    def reproject(self, target_epsg:int, inplace:bool=True):
        """
        Reproject raster to the desired crs represented by `target_epsg`.

        :param target_epsg: epsg coder of the target crs
        :type target_epsg: int
        :param inplace: specify if reprojection should be applied to the instance itself or if a reprojected copy should be returned, defaults to True
        :type inplace: bool, optional
        :return: reprojected copy of this instance if `inplace=False`
        :rtype: GeoRaster
        """
        if inplace:
            obj:GeoRaster = self
        else:
            obj:GeoRaster = deepcopy(self)
            
        scrs = rio.CRS.from_epsg(self.crs)
        strf = affine_from_trafo(self.transformation)
        tcrs = rio.CRS.from_epsg(target_epsg)

        tarr = None
        sarr = self.raster.copy()
        if self.layer_index == 2:
            sarr = imgshape_to_rasterio(sarr)
        for i in range(self.raster_shape[self.layer_index]):
            tlay, ttrf = riowrp.reproject(
                source=sarr[i:i + 1, :, :], src_crs=scrs, src_transform=strf, 
                dst_crs=tcrs, dst_nodata=self.nodata_value
            )
            if tarr is None:
                tarr = np.zeros((sarr.shape[0], tlay.shape[1], tlay.shape[2]))
            tarr[i:i + 1, :, :] = tlay
            
        obj.raster = tarr
        obj.crs = target_epsg
        obj.bounds = riowrp.transform_bounds(scrs, tcrs, *self.bounds)
        obj.transformation = trafo_from_affine(ttrf)
        if not inplace:
            return obj
    
    def _check_impl(self, attr_name:str) -> None:
        if getattr(self, attr_name) is None:
            raise ValueError(attr_name + ' has not been provided or set yet!')
    
    def _reset_iteration(self) -> None:
        # class-private iterator attributes
        self._crst:np.ndarray = None  # (2, n) numpy ndarray - each column contains the raster/image coordinates of the raster points
        self._cobj:np.ndarray = None  # (2, n) numpy ndarray - each column contains the object/geo coordinates of the raster points
        self._i1, self._i2, self._i3, self._i4 = None, None, None, None
        self._nc = None  # number of columns of raster

    def _set_metadata(self, avlbl_keys:list[str], kwargs:dict) -> None:
        for key, val in kwargs.items():
            if key in avlbl_keys:
                try:
                    json.dumps(val)
                    self._meta[key] = val
                except:
                    is_date = isinstance(val, datetime.date)
                    is_datetime = isinstance(val, datetime.datetime)
                    is_time = isinstance(val, datetime.time)
                    if is_date or is_datetime or is_time:
                        self._meta[key] = val.isoformat()
                        
                    wrn = 'Provided value at dict-key {} is not '
                    wrn += 'json-serializable!'
                    print(wrn.format(key))
                    continue

    def _get_layers(self, ix) -> np.ndarray:
        """
        :param ix: slice or tuple/list of integers to access layers of `GeoRaster`
        :type ix: slice or tuple or list
        :return: three-dimensional array representing the chosen layers
        :rtype: numpy.ndarray
        """
        if self.layer_index == 0:
            return self.raster[ix, :, :]
        elif self.layer_index == 2:
            return self.raster[:, :, ix]
        
    def _set_layers(self, ix, value:np.ndarray) -> None:
        """
        :param ix: slice or tuple/list of integers to access layers of `GeoRaster`
        :type ix: slice or tuple or list
        :param value: array with values which should be set
        :type value: np.ndarray
        """
        if self.layer_index == 0:
            self.raster[ix, :, :] = value
        elif self.layer_index == 2:
            self.raster[:, :, ix] = value

    def _prc_ix_tpl(self, ixtpl:tuple) -> list[int]:
        ixs = []
        for val in ixtpl:
            if isinstance(val, int):
                ixs.append(val)
            elif isinstance(val, str):
                ixs.append(self.layer_ids.index(val))
            else:
                raise ValueError(self.ERR_IXTPLVAL_TYPE)
        return ixs

    def __getitem__(self, key) -> np.ndarray:
        if isinstance(key, int):
            return self._get_layers(slice(key, key + 1))
        elif isinstance(key, str):
            ix = self.layer_ids.index(key)
            return self._get_layers(slice(ix, ix + 1))
        elif isinstance(key, slice):
            return self._get_layers(key)
        elif isinstance(key, tuple):
            return self._get_layers(self._prc_ix_tpl(key))
        else:
            raise ValueError(self.ERR_INDEX_TYPE)
        
    def __setitem__(self, key, value):
        if isinstance(key, int):
            self._set_layers(slice(key, key + 1), value)
        elif isinstance(key, str):
            ix = self.layer_ids.index(key)
            self._set_layers(slice(ix, ix + 1), value)
        elif isinstance(key, slice):
            self._set_layers(key, value)
        elif isinstance(key, tuple):
            self._set_layers(self._prc_ix_tpl(key), value)
        else:
            raise ValueError(self.ERR_INDEX_TYPE)

    def __iter__(self):
        if self.raster_shape is None:
            raise ValueError('No raster data respectively its shape available!')
        if self.layer_index == 0:
            nr, nc = self.raster_shape[1:]
        else:
            nr, nc = self.raster_shape[:2]

        self._crst = np.vstack((
            np.atleast_2d(np.tile(np.arange(0, nc + 1), nr + 1)),
            np.atleast_2d(np.repeat(np.arange(0, nr + 1), nc + 1)),
        ))
        self._cobj = (
            self.transformation @ np.vstack((
                self._crst, np.ones((1, (nr + 1) * (nc + 1)), dtype=int)
            ))
        )[:2, :]

        self._nc = nc
        return self

    def __next__(self):
        ret = None

        if self._i1 is None:
            self._i1 = 0
        else:
            self._i1 += 1
        self._i2, self._i3, self._i4 = self._update_is(self._i1, self._nc)
        if self._i4 >= self._crst.shape[1]:
            self._reset_iteration()
            raise StopIteration
        if self._crst[0, self._i4] == 0:
            self._i1 += 1
            self._i2, self._i3, self._i4 = self._update_is(self._i1, self._nc)

        if self.raster is not None:
            if self.layer_index == 0:
                ret = self.raster[:, self.current_row, self.current_column]
            elif self.channel_index == 2:
                ret = self.raster[self.current_row, self.current_column, :]
            else:
                raise ValueError('Invalid channel index!')

        return ret
    
    @staticmethod
    def _update_is(i1, nc) -> tuple[int, int, int]:
        """
        Method used in the iterator implementation of `GeoRaster`
        """
        i2 = i1 + 1
        i3 = i2 + nc
        i4 = i3 + 1
        return i2, i3, i4
    
    @classmethod
    def from_gdf_and_objres(
        cls, gdf:GeoDataFrame, objres:float, nlayers:int=1, 
        rtype=PixelUnits.FLOAT32, nodataval=np.nan, lix:int=0
    ):
        """
        Create instance of :class:`GeoRaster` from given input.

        :param gdf: GeoDataFrame specifying the areo of interest
        :type gdf: geopandas.GeoDataFrame
        :param objres: object resolution of resulting :class:`GeoRaster` instance
        :type objres: float
        :param nlayers: number of layers, defaults to 1
        :type nlayers: int, optional
        :param rtype: data type of raster, defaults to PixelUnits.FLOAT32
        :type rtype: str, optional
        :param nodataval: no-data-value (GeoTIFF), defaults to np.nan
        :type nodataval: any, optional
        :param lix: index which spedifies the axis/dimension which contains the layers/channels of the raster, defaults to 0
        :type lix: int, optional
        :return: instance of :class:`GeoRaster`
        :rtype: mef_agri.utils.raster.GeoRaster
        """
        # preliminary computations to get raster bounds which exactly fit the 
        # rows and columns in object space
        bb1 = bbox_from_gdf(gdf)
        dx, dy = bb1[2] - bb1[0], bb1[3] - bb1[1]
        rr, rc = dy // objres + 2, dx // objres + 2

        # initializing GeoRaster class
        rstr = cls()
        rstr.crs = gdf.crs.to_epsg()
        rstr.bounds = (
            bb1[0] - objres, 
            bb1[1] - objres, 
            bb1[0] - objres + rc * objres,
            bb1[1] - objres + rr * objres
        )
        rstr.transformation = np.array([
            [objres, 0., rstr.bounds[0]],
            [0., -objres, rstr.bounds[-1]],
            [0., 0., 1.]
        ])
        rstr.layer_index = lix
        rstr.raster_shape = (nlayers, int(rr), int(rc))
        rstr.units = rtype
        rstr.raster = np.zeros(rstr.raster_shape, dtype=rtype)
        rstr.raster[:] = nodataval

        return rstr
