import numpy as np

from ...utils.raster import GeoRaster
from . import Zoning


def compute_zone_values(r:GeoRaster, z:Zoning) -> dict:
    ret = {}
    for zname in z.layer_ids:
        # get image coordinates of pixels belonging to current zone
        ics_z = np.where(z[zname])
        ics_z = ics_z[1:] if z.layer_index == 0 else ics_z[:2]
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
        gcs_z_4 = z.transformation @ ics_z_4
        # transform zone geo-coordinates into the image-coordinates of the 
        # georaster `r`, i.e. the data-domain
        # NOTE `ics_d_4` contains float-values and is not homogenous anymore
        ics_d_4_float = (r.transformation_inv @ gcs_z_4)[:2, :]
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
        # TODO restrict to number of rows and columns
