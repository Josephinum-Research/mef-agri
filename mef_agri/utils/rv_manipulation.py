import numpy as np
from geopandas import GeoDataFrame

from .raster import GeoRaster
from .misc import set_attributes


class IntersectionBaseClass(object):
    def __init__(self, rstr:GeoRaster, **kwargs) -> None:
        self.nodata_value = np.nan
        self.fraction = 1.0
        set_attributes(self, kwargs)

        self._rstr:GeoRaster = rstr
        self._ixnan:np.ndarray = None
        self._ixins:np.ndarray = None
        self._wghts:np.ndarray = None

    @property
    def processed_raster(self) -> np.ndarray:
        """
        :return: Raster which contains no-data-values outside of the intersecting area
        :rtype: np.ndarray
        """
        if self._ixnan is None:
            self.compute()
        rstr = self._rstr.raster.copy()
        if self._rstr.layer_index == 0:
            rstr[:, self._ixnan[0, :], self._ixnan[1, :]] = self.nodata_value
        else:
            rstr[self._ixnan[0, :], self._ixnan[1, :], :] = self.nodata_value
        return rstr
    
    @property
    def nodata_indices(self) -> np.ndarray:
        """
        :return: (2, n) numpy array where first row corresponds to the row indices and second row to the column indices of pixels outside of the intersecting area
        :rtype: np.ndarray
        """
        if self._ixnan is None:
            self.compute()
        return self._ixnan
    
    @property
    def inside_indices(self) -> np.ndarray:
        """
        :return: (2, n) numpy array where first row corresponds to the row indices and second row to the column indices of pixels inside of the intersecting area
        :rtype: np.ndarray
        """
        if self._ixins is None:
            self.compute()
        return self._ixins
    
    @property
    def pixel_weights(self) -> np.ndarray:
        """
        :return: (1, n_rows, n_cols) numpy array containing the fraction of intersection area (i.e. intersecting area divided by the pixel area)
        :rtype: np.ndarray
        """
        if self._wghts is None:
            self.compute()
        return self._wghts
    
    def compute(self) -> None:
        raise NotImplementedError()


class RasterVectorIntersection(IntersectionBaseClass):
    def __init__(self, geom, rstr:GeoRaster, **kwargs) -> None:
        """
        Class which performs the intersection of a `GeoRaster` object with a 
        vector/geometry object.

        :param geom: vector feature which will be used for intersection with the raster
        :type geom: Polygon or GeoDataFrame
        :param rstr: raster which will be intersected with `geom`
        :type rstr: GeoRaster
        """
        super().__init__(rstr, **kwargs)
        self._geom = geom
        self._clsss:np.ndarray = None

    @property
    def assignment(self) -> np.ndarray:
        """
        If `geom` is a polygon, this array contains ones where a pixel is inside 
        the polygon, otherwise zeros (i.e. it is a mask in this case). If `geom` 
        is a GeoDataFrame, this array contains zeros for pixels which are in 
        none of the geometries, and a continuous integer if a pixel is in the 
        corresponding geoemetry.

        :return: array containing labels for pixels which are inside the provided geometry
        :rtype: np.ndarray
        """
        return self._clsss
    
    def compute(self) -> None:
        """
        Compute intersecting pixels fo the provided `GeoRaster`. `GeoRaster` is 
        iterated and pixels are created as polygons which are intersected with 
        the provided `geom` of the __init__-method. If `geom` is a polygon, the 
        intersecting area is divided by the pixel area to get the percentage. If 
        `geom` is a GeoDataFrame (containing polygons), the maximum intersecting 
        area is used for the computation of the perecentage of intersection.
        If the percentage exceeds the `fraction` attribute of this class, the 
        pixel will be assumed to be inside the provided geometry.
        """
        if self._rstr.layer_index == 0:
            nr, nc = self._rstr.raster_shape[1:]
        else:
            nr, nc = self._rstr.raster_shape[:2]
        self._ixnan = np.zeros((2, 0), dtype=int)
        self._ixins = np.zeros((2, 0), dtype=int)
        self._wghts = np.zeros((1, nr, nc), dtype=float)
        self._clsss = np.zeros((1, nr, nc), dtype=int)
        fldlbl = 1
        for _ in self._rstr:
            pixl = self._rstr.current_polygon
            # compute intersection and corresponding area for the decision
            iprc = self._geom.intersection(pixl).area / pixl.area
            if isinstance(self._geom, GeoDataFrame):
                fldlbl = iprc.values.argmax() + 1  # exclude zero as this is for the "outside pixels"
                iprc = iprc.values.max()
            crst = np.array([
                [self._rstr.current_row, self._rstr.current_column]
            ]).T
            if iprc < self.fraction:
                self._ixnan = np.hstack((self._ixnan, crst))
            else:
                self._ixins = np.hstack((self._ixins, crst))
                self._clsss[0, crst[0, 0], crst[1, 0]] = fldlbl
            self._wghts[0, crst[0, 0], crst[1, 0]] = iprc


def georaster2geodataframe(
        rstr:GeoRaster, column_labels:list[str]=None, column_geom:str=None
    ) -> GeoDataFrame:
    if column_labels is None:
        if rstr.channel_index == 0:
            nlayers = rstr.raster_shape[0]
        elif rstr.channel_index == 2:
            nlayers = rstr.raster_shape[2]
        column_labels = ['layer-' + str(i) for i in range(nlayers)]
    if column_geom is None:
        column_geom = 'geometry'

    data = []
    for di in rstr:
        drow = di.tolist() + [rstr.current_polygon]
        data.append(drow)
    return GeoDataFrame(
        data=data, columns=column_labels + [column_geom], geometry=column_geom
    )
