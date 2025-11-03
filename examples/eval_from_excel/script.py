import os
import pandas as pd
import numpy as np
import geopandas as gpd
import datetime as dt
import matplotlib.pyplot as plt

from mef_agri.utils.raster import GeoRaster
from mef_agri.utils.rv_manipulation import RasterVectorIntersection

from mef_agri.data.project import Project
from mef_agri.data.jr_management.interface import ManagementInterface


def check_gpkg():
    objres = 10.
    flds = gpd.read_file(os.path.join(wdir, 'testdb.gpkg'), layer='fields')
    zns = gpd.read_file(os.path.join(wdir, 'testdb.gpkg'), layer='parcels')
    fx, fy = flds['geometry'].values[0].exterior.coords.xy
    z1x, z1y = zns[zns['pid'] == 1]['geometry'].values[0].exterior.coords.xy
    z2x, z2y = zns[zns['pid'] == 2]['geometry'].values[0].exterior.coords.xy

    frstr = GeoRaster.from_gdf_and_objres(flds, objres)
    frstr.raster = np.random.normal(size=frstr.raster_shape)

    rvi = RasterVectorIntersection(zns, frstr)
    rvi.fraction = 1e-4
    rvi.compute()

    f1, a1 = plt.subplots(1, 1)
    a1.imshow(rvi.processed_raster[0, :, :], extent=frstr.extent)
    a1.plot(fx, fy, color='red')
    a1.plot(z1x, z1y, color='gold')
    a1.plot(z2x, z2y, color='gold')
    a1.axis('equal')


if __name__ == '__main__':
    wdir = os.path.join('/', 'home', 'andreas', 'Downloads', 'eval_from_excel')

    prj = Project(wdir, 'fields', 'fname')
    prj.add_data_interface(ManagementInterface)
    prj.add_data(dt.date(2024, 1, 1), dt.date(2025, 12, 31))

    prj.quit_project()
    plt.show()
