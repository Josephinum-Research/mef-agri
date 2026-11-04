import numpy as np
import matplotlib.pyplot as plt

from mef_agri.evaluation.zoning.utils import pixels_in_zone
from mef_agri.evaluation.analysis.utils import plot_raster


def image2geo(arr_shape:tuple, trafo:np.ndarray) -> np.ndarray:
    ics = homogeneous_coords(
        np.flipud(np.indices(arr_shape).reshape(2, arr_shape[0] * arr_shape[1]))
    )
    return (trafo @ ics)[:2, :]


def homogeneous_coords(arr:np.ndarray) -> np.ndarray:
    return np.vstack((arr, np.ones((1, arr.shape[1]))))


if __name__ == '__main__':
    zarr = np.array([
        [ 1, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0],
        [ 1, 0, 0, 0, 0],
        [ 1, 1, 1, 0, 1],
    ])
    or_z = 50.
    or_d = 15.
    tz_i2g = np.array([
        [or_z, 0., -1000.],
        [0., -or_z, 250],
        [0., 0., 1.]
    ])
    td_i2g = np.array([
        [or_d, 0., -1013.],
        [0., -or_d, 231.],
        [0., 0., 1.]
    ])
    td_g2i = np.linalg.inv(td_i2g)
    d_shape = (
        int(zarr.shape[0] * (or_z / or_d) + 1),
        int(zarr.shape[1] * (or_z / or_d) + 1)
    )

    gcs_z = image2geo(zarr.shape, tz_i2g)
    gcs_d = image2geo(d_shape, td_i2g)

    ics_z1 = np.where(zarr)
    ics_z1 = homogeneous_coords(
        np.vstack((np.atleast_2d(ics_z1[1]), np.atleast_2d(ics_z1[0])))
    )
    gcs_z1 = (tz_i2g @ ics_z1)[:2, :]

    ics_dz1 = homogeneous_coords(
        pixels_in_zone(zarr, tz_i2g, td_g2i, d_shape, corder='geo')
    )
    gcs_dz1 = (td_i2g @ ics_dz1)[:2, :]

    fig, axs = plt.subplots(1, 1)
    axs.axis('equal')
    axs = plot_raster(axs, gcs_d, or_d, 'forestgreen', 'lightgreen', lw=3, ms=10)
    #axs = plot_raster(axs, gcs_z, or_z, 'cornflowerblue', 'lightblue', lw=3, ms=10)
    axs = plot_raster(axs, gcs_z1, or_z, 'darkorange', 'orange')
    axs = plot_raster(axs, gcs_dz1, or_d, 'orangered', 'salmon')

    plt.show()
