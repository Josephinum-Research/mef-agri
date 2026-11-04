import numpy as np

from ...utils.raster import GeoRaster
from . import Zoning


def pixels_in_zone(
        zarr:np.ndarray, trafo_z_i2g:np.ndarray, trafo_d_g2i:np.ndarray, 
        dshape:tuple[int, int], corder='img'
    ) -> np.ndarray:
    """
    Determine the image coordinates (i.e. column and row indices) of pixels from 
    a data-array which touch at least on pixel of a zoning-array.
    ``corder`` specifies the order of the returned image coordinates

    * ``'img'`` (default) - first coordinate = row index, second coordinate = column index
    * ``'geo'`` - first coordinate = column index, second coordinate = row index

    :param zarr: 2D-array containing 1's for pixels which belong to a zone and otherwise 0's
    :type zarr: numpy.ndarray
    :param trafo_z_i2g: matrix to transform from the image coordinate system into geo-coordinates
    :type trafo_z_i2g: numpy.ndarray
    :param trafo_d_g2i: matrix to transform geo-coordinates into image coordinate system of the data-array
    :type trafo_d_g2i: numpy.ndarray
    :param dshape: shape of the data-array (rows, columns)
    :type dshape: tuple[int, int]
    :param corder: order of image coordinates, defaults to ``'img'``
    :type corder: str, optional
    :return: image coordinates of pixels which touch at least one zoning-pixel (first row are the column-indices, second row are the row-indices)
    :rtype: (2, n) numpy.ndarray
    """
    # get image coordinates of pixels belonging to current zone
    ics_z = np.where(zarr)
    # convert the image coordinates in zoning-domain to homogeneous vectors 
    # i.e. 1 at third position
    # important NOTE 
    # first coordinate represents the column indices and the second ones 
    # the row indices
    ics_z = np.vstack((
        np.atleast_2d(ics_z[1]),
        np.atleast_2d(ics_z[0]),
        np.ones((1, len(ics_z[0])))
    ))
    # extend image coordinates representing upper-left corner of the pixels
    # by the image coordinates of the other three corners of the pixels
    ics_z_4 = np.hstack((
        ics_z,
        ics_z + np.array([[1, 0, 0]]).T,
        ics_z + np.array([[0, 1, 0]]).T,
        ics_z + np.array([[1, 1, 0]]).T
    ))
    # transform zone image coordinates into geo-coordinates
    gcs_z_4 = trafo_z_i2g @ ics_z_4
    # transform zone geo-coordinates into the image-coordinates of the 
    # georaster `r`, i.e. the data-domain
    # NOTE `ics_d_4` contains float-values and is not homogenous anymore
    ics_d_4_float = (trafo_d_g2i @ gcs_z_4)[:2, :]
    # consider the edge case, that zoning and data exhibit the same object
    # resolution and do exactly overlay because in this case 4 data pixels
    # would be used to compute the zone value but only the one data pixel 
    # which coincides with the zone is necessary
    decimal_digits = ics_d_4_float - np.floor(ics_d_4_float)
    ixs_corr = np.where((ics_d_4_float > 0.) & (decimal_digits == 0.))
    ics_d_4_float[ixs_corr] -= 1.
    # convert float to integers and restrict the values to be in the range 
    # 0 <= row-index < number-of-rows
    # 0 <= column-index < number-of-columns
    ics_d_4 = np.floor(ics_d_4_float).astype(int)
    ics_d_4[np.where(ics_d_4 < 0)] = 0
    ics_d_4[0, :][np.where(ics_d_4[0, :] >= dshape[1])[0]] = dshape[1] - 1
    ics_d_4[1, :][np.where(ics_d_4[1, :] >= dshape[0])[0]] = dshape[0] - 1
    # check if zone-pixels touch more than 4 data-pixels (i.e. object 
    # resolution of data is smaller then that of zoning)
    ncs = ics_z.shape[1]  # number of pixels in current zone
    d41 = ics_d_4[:, 3 * ncs:] - ics_d_4[:, :ncs]
    ixs_fill = np.where(np.max(d41, axis=0) > 1)[0]
    # if zone-pixels do not touch more than 4 data-pixels nothing has to be 
    # done anymore and possibly duplicate image coordinates are removed
    if len(ixs_fill) == 0:
        if corder == 'img':
            ics_d_4 = np.flipud(ics_d_4)
        return np.unique(ics_d_4, axis=1)
    # if zone-pixels touch more than 4 data-pixels, additional image 
    # coordinates have to be added 
    # example (image coordinates are (col, row)):
    #
    # upper-left corner of zone-pixel maps to (0, 0) of data
    # lower-left corner of zone-pixel maps to (0, 2) of data
    # upper-right corner of zone-pixel maps to (2, 0) of data
    # lower-right corner of zone-pixel maps to (2, 2) of data
    #
    # in this case, also data-pixels with image coordinates 
    # (0, 1), (1, 0), (1, 1), (1, 2) and (2, 1)
    # are located in the zone-pixel
    for ix in ixs_fill:
        dc, dr = d41[:, ix]
        ics_d_4 = np.hstack((
            ics_d_4,
            np.flipud(
                np.indices((dr + 1, dc + 1)).reshape(2, (dr + 1) * (dc + 1))
            ) + ics_d_4[:, ix:ix + 1]
        ))
    if corder == 'img':
        ics_d_4 = np.flipud(ics_d_4)
    return np.unique(ics_d_4, axis=1)


def compute_zone_values(r:GeoRaster, z:Zoning, zone_name:str=None) -> dict:
    """
    Compute the median of all pixels from ``r`` which touch a zone of ``z``.
    The returned value is a dictionary where the first level of keys 
    corresponds to the zone-names (i.e. ``z.layer_ids``) with values being again 
    dictionaries.
    The second level of keys corresponds to the data-layers (i.e. 
    ``r.layer_ids``) with values being the median of the pixels which touch the 
    zone from the first level of keys.
    If ``zname`` is specified, only one dictionary with keys representing the 
    data-layers is returned.

    :param r: georaster containing the data
    :type r: mef_agri.utils.raster.GeoRaster
    :param z: zoning
    :type z: mef_agri.evaluation.zoning.Zoning
    :return: median values of data-pixels which touch the zones
    :rtype: dict
    """
    ret = {}
    dshape = r.raster_shape[1:] if r.layer_index == 0 else r.raster_shape[:2]

    zones = z.layer_ids if zone_name is None else [zone_name,]
    for zname in zones:
        zone = z[zname][0, :, :] if z.layer_index == 0 else z[zname][:, :, 0]
        # get image coordinates of data-pixels which touch a pixel of a zone
        ics = pixels_in_zone(
            zone, z.transformation, r.transformation_inv, dshape, corder='img'
        )
        # zvals contains the data-layers of `r` on the row-index and the 
        # corresponding values which are located in the current zone on the 
        # column-index
        if r.layer_index == 0:
            zvals = r.raster[:, ics[0, :], ics[1, :]]
        elif r.layer_index == 2:
            zvals = r.raster[ics[0, :], ics[1, :], :].T
        ret[zname] = dict(zip(r.layer_ids, np.nanmedian(zvals, axis=1)))

    if zone_name is None:
        return ret
    else:
        ret[zone_name]

