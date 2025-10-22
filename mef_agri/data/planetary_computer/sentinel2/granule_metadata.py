import dask
import xml.etree.ElementTree as ETree
from urllib.request import urlopen
from datetime import datetime
from pystac import Item
from pystac.extensions.projection import ProjectionExtension

import numpy as np
import rioxarray  # noqa # pylint: disable=unused-import
import xarray as xr


"""
This module is designed to parse an MTD_TL.xml file of a Sentinel2 L2A product.
The XML file is fully processed and the contained grid information aka.: sun and viewing incidence angle
will be returned as xarray.Dataset.
Please note that no interpretation of the data takes place. Therefore, you will need to know how interpret these values.
see: https://forum.step.esa.int/t/generate-view-angles-from-metadata-sentinel-2/5598/1
another parser: https://towardsdatascience.com/how-to-implement-sunglint-detection-for-sentinel-2-images-in-python-using-metadata-info-155e683d50
understanding detectors: https://sentiwiki.copernicus.eu/web/s2-mission#S2Mission-MSIInstrumentS2-Mission-MSI-Instrumenttrue
"""


GRID_GSD_X, GRID_GSD_Y = 5000, 5000
GRID_SIZE_X, GRID_SIZE_Y = 23, 23


def parse_granule_metadata_lazy(norm_item:Item, donor_band:str=None) -> xr.Dataset:
    """
    this parsing function uses the metadata provided via STAC-Item
    to allow lazy-loading of granule_metadata.
    :param norm_item: a normalized stac item providing the required metadata. must provide:
    * projection extension: https://github.com/stac-extensions/projection
    * B4 (will be used as template)
    * granule_metadata
    :return: a lazy loaded Dataset containing DataArray sun_angles & view_angles
    :raises ValueError: if projection extension is not provided or important information is missing
    :raises urllib.error.HTTPError: lazy if the provided granule_metadata href is not accessible
    """
    if not ProjectionExtension.has_extension(norm_item):
        raise ValueError("only items with projection extension are supported")
    proj_item = ProjectionExtension.ext(norm_item)
    if not proj_item.epsg:
        raise ValueError("item must provide an EPSG code")
    # we use B4 as donor for projection metadata
    # this is possible because the upper left corner of all sentinel 2 images align
    # see 4.9.2.1: https://sentinel.esa.int/documents/247904/349490/S2_MSI_Product_Specification.pdf
    dband = 'B04' if donor_band is None else donor_band
    template_asset = norm_item.assets[dband]
    proj_asset = ProjectionExtension.ext(template_asset)
    if not proj_asset.transform:
        raise ValueError("asset B4 must provide projection transform")
    epsg_code = proj_item.epsg
    # see https://github.com/stac-extensions/projection?tab=readme-ov-file#projtransform
    # this is a shortcut for [[a0, a1, a2],[a3, a4, a5],[0,0,1]] * [0,0,1] = [a2,a5,1]
    geo_pos_x_min = proj_asset.transform[2]
    geo_pos_y_max = proj_asset.transform[5]

    lazy_et = dask.delayed(load_etree_from_url)(norm_item.assets['granule-metadata'].href)
    lazy_et_root = lazy_et.getroot()
    delayed_sun_angles = dask.delayed(_extract_sun_angles)(lazy_et_root)
    delayed_view_angles = dask.delayed(_extract_view_angles)(lazy_et_root)
    lazy_sun_angels = dask.array.from_delayed(delayed_sun_angles, shape=(2, 23, 23), dtype=np.float32)
    lazy_view_angels = dask.array.from_delayed(delayed_view_angles, shape=(2, 23, 23, 13, 12), dtype=np.float32)

    return _assemble_granule_dataset(
        sun_angles=lazy_sun_angels,
        view_angles=lazy_view_angels,
        epsg_code=epsg_code,
        geo_pos_x=geo_pos_x_min,
        geo_pos_y=geo_pos_y_max,
        sensing_time=norm_item.datetime
    )


def load_etree_from_url(url: str) -> ETree:
    with urlopen(url) as xml_file:
        return ETree.parse(xml_file)



def simplified_metadata(metadata: xr.Dataset) -> xr.Dataset:
    """
    combine the viewing angles as described in here:
    https://forum.sentinel-hub.com/t/sunazimuthangles-and-viewazimuthmean/1479/2
    https://forum.step.esa.int/t/generate-view-angles-from-metadata-sentinel-2/5598/2
    :param metadata: the already parsed grid data contained in MTD_TL.xml file
    :return: a simplified xarray.Dataset
    """
    viewing_angle: xr.DataArray = metadata.view_angles
    # as described in the linked step forum post, detector of the same band are combined by replacing the NaN values
    # of the detector with the highest id by values from the lower ones.
    # ffill dose exactly that. this function requires the dependency Bottleneck
    viewing_angle_stitched = viewing_angle.ffill(dim="detector").isel(detector=-1)
    viewing_angle_mean = viewing_angle_stitched.mean(dim="band")
    simpl_metadata = xr.Dataset({
        "sun_angles": metadata.sun_angles,
        "view_angles_mean": viewing_angle_mean
    })
    simpl_metadata.rio.write_crs(metadata.rio.crs, inplace=True)
    simpl_metadata.rio.set_spatial_dims(
        x_dim="x",
        y_dim="y",
        inplace=True,
    )
    simpl_metadata.rio.write_transform(inplace=True) # forces rioxarray to compute the transformation now.
    return simpl_metadata


def parse_granule_metadata(mtd_tl_source) -> xr.Dataset:
    element_tree = ETree.parse(mtd_tl_source)
    return __parse_tile_angles(element_tree)


def __parse_tile_angles(tree: ETree.ElementTree) -> xr.Dataset:
    root = tree.getroot()

    epsg_str = root.find('{*}Geometric_Info/Tile_Geocoding/HORIZONTAL_CS_CODE').text
    epsg_code = int(epsg_str[5:])
    geo_pos_x = float(root.find('{*}Geometric_Info/Tile_Geocoding/Geoposition/ULX').text)
    geo_pos_y = float(root.find('{*}Geometric_Info/Tile_Geocoding/Geoposition/ULY').text)
    sensing_time = datetime.fromisoformat(root.find('{*}General_Info/SENSING_TIME').text)

    sun_angles = _extract_sun_angles(root)
    view_angles = _extract_view_angles(root)

    return _assemble_granule_dataset(
        sun_angles=sun_angles,
        view_angles=view_angles,
        epsg_code=epsg_code,
        geo_pos_x=geo_pos_x,
        geo_pos_y=geo_pos_y,
        sensing_time=sensing_time
    )


def _assemble_granule_dataset(
        sun_angles,
        view_angles,
        epsg_code: int,
        geo_pos_x: float,
        geo_pos_y: float,
        sensing_time: datetime
) -> xr.Dataset:

    geo_pos_x_center = geo_pos_x + GRID_GSD_X / 2
    geo_pos_y_center = geo_pos_y - GRID_GSD_Y / 2

    # converting to ns to prevent the xarray UserWarning
    np_sensing_time = np.datetime64(int(sensing_time.timestamp() * 1_000), 'ms')

    sun_angles_reshaped = np.expand_dims(sun_angles, axis=0)
    xr_sun_angles = xr.DataArray(
        data=sun_angles_reshaped,
        dims=['time', 'spherical', 'y', 'x'],
        coords={
            'time': [np_sensing_time],
            'spherical': ['zenith', 'azimuth'],
            'x': np.linspace(start=geo_pos_x_center, stop=(geo_pos_x_center + GRID_GSD_X * (GRID_SIZE_X - 1)),
                             num=GRID_SIZE_X),
            'y': np.linspace(start=geo_pos_y_center, stop=(geo_pos_y_center - GRID_GSD_Y * (GRID_SIZE_Y - 1)),
                             num=GRID_SIZE_Y),
        }
    )

    view_angles_reshaped = np.expand_dims(view_angles, axis=0)
    xr_view_angles = xr.DataArray(
        data=view_angles_reshaped,
        dims=['time', 'spherical', 'y', 'x', 'band', 'detector'],
        coords={
            'time': [np_sensing_time],
            'spherical': ['zenith', 'azimuth'],
            'x': np.linspace(start=geo_pos_x_center, stop=(geo_pos_x_center + GRID_GSD_X * (GRID_SIZE_X - 1)),
                             num=GRID_SIZE_X),
            'y': np.linspace(start=geo_pos_y_center, stop=(geo_pos_y_center - GRID_GSD_Y * (GRID_SIZE_Y - 1)),
                             num=GRID_SIZE_Y),
            'band': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12'],
            'detector': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        }
    )

    ds_granule_metadata = xr.Dataset({"sun_angles": xr_sun_angles, "view_angles": xr_view_angles})

    ds_granule_metadata.rio.write_crs(epsg_code, inplace=True)
    ds_granule_metadata.rio.set_spatial_dims(
        x_dim="x",
        y_dim="y",
        inplace=True,
    )
    ds_granule_metadata.rio.write_transform(inplace=True)
    return ds_granule_metadata


def _extract_sun_angles(root: ETree.Element) -> np.ndarray:
    """
    will extract the sun angle grid out of a given ETree element.
    dimension order will be ['spherical', 'y', 'x']
    """
    et_sun_angels = root.find('{*}Geometric_Info/Tile_Angles/Sun_Angles_Grid')
    sun_angles_zenith = __parse_value_list(et_sun_angels.find('./Zenith/Values_List'))
    sun_angles_azimuth = __parse_value_list(et_sun_angels.find('./Azimuth/Values_List'))
    sun_angles = np.stack([sun_angles_zenith, sun_angles_azimuth], axis=0)
    return sun_angles


def _extract_view_angles(root: ETree.Element) -> np.ndarray:
    """
    will extract the view angles grid out of a given ETree element.
    dimension order will be ['spherical', 'y', 'x', 'band', 'detector']
    coords wil be:
    {
        'spherical': ['zenith', 'azimuth'],
        'band': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12'],
        'detector': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    }
    note: not all MSI detectors contribute to each S2 product (tile).
    depending on the position of the product in a swath, only ~5-7 detectors contribute measurements.
    """
    view_angles = np.full((2, 23, 23, 13, 12), fill_value=np.nan, dtype=np.float32)
    et_view_angles = root.findall('{*}Geometric_Info/Tile_Angles/Viewing_Incidence_Angles_Grids')
    if not et_view_angles:
        raise AttributeError('provided XML did not contain Geometric_Info/Tile_Angles/Viewing_Incidence_Angles_Grids')
    for et_view_angle in et_view_angles:
        angles_zenith = __parse_value_list(et_view_angle.find('./Zenith/Values_List'))
        angles_azimuth = __parse_value_list(et_view_angle.find('./Azimuth/Values_List'))
        band_id = int(et_view_angle.attrib['bandId'])
        detector_id = int(et_view_angle.attrib['detectorId'])
        detector_idx = detector_id - 1  # detector_id starts with 1
        view_angles[0, :, :, band_id, detector_idx] = angles_zenith
        view_angles[1, :, :, band_id, detector_idx] = angles_azimuth
    return view_angles


def __parse_value_list(value_list_elem: ETree.Element) -> np.ndarray:
    value_lines = value_list_elem.findall('VALUES')
    h = len(value_list_elem)
    w = len(value_lines[0].text.split(' '))
    grid = np.empty((h, w), dtype=np.float32)
    for idx, value_line in enumerate(value_lines):
        values = [float(x) for x in value_line.text.split(' ')]
        grid[idx, :] = values
    return grid
