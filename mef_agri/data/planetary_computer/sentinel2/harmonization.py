from pystac import Item
from pystac.extensions.raster import RasterExtension


default_raster_meta = {
    "nodata": 0,
    "data_type": "uint16",
    "bits_per_sample": 15,
    "scale": 0.0001,
    "offset": -0.1
}

asset_raster_meta_10m = {
    "raster:bands": [
        default_raster_meta | {"spatial_resolution": 10}
    ]
}

asset_raster_meta_20m = {
    "raster:bands": [
        default_raster_meta | {"spatial_resolution": 20}
    ]
}

asset_raster_meta_60m = {
    "raster:bands": [
        default_raster_meta | {"spatial_resolution": 60}
    ]
}

asset_raster_metadata = {
    'B02': asset_raster_meta_10m,
    'B01': asset_raster_meta_60m,
    'B03': asset_raster_meta_10m,
    'B08': asset_raster_meta_10m,
    'B8A': asset_raster_meta_20m,
    'B09': asset_raster_meta_60m,
    'B04': asset_raster_meta_10m,
    'B05': asset_raster_meta_20m,
    'B06': asset_raster_meta_20m,
    'B07': asset_raster_meta_20m,
    'SCL': {
        "raster:bands": [
            {
                "nodata": 0,
                "data_type": "uint8",
                "spatial_resolution": 20
            }
        ]
    },
    'B11': asset_raster_meta_20m,
    'B12': asset_raster_meta_20m,
}

def harmonization(item:Item) -> Item:
    """
    BOA correction and reflectance mapping information is added to the provided 
    item see: 
    https://sentinels.copernicus.eu/web/sentinel/-/copernicus-sentinel-2-major-products-upgrade-upcoming
    """
    def extract_processing_version(item: Item) -> str:
        """
        Extract the version number of the used processing software.
        currently there are two STAC extensions for that:
            
        https://github.com/stac-extensions/sentinel-2
        s2:processing_baseline property is deprecated but data-providable 
        still use it.
            
        https://github.com/stac-extensions/processing
        processing:version is the successor, but data-provider have not 
        switched to it yet.
        """
        for key in ('s2:processing_baseline', 'processing:version'):
            try:
                return item.properties[key]
            except KeyError:
                continue
        raise AttributeError('the given STAC-Item did not contain any processing version')    
        
    # get baseline to choose correct offset
    try:
        baseline = float(extract_processing_version(item))
    except:
        msg = f'unable to determine baseline of item: {item.id}, scale and '
        msg += 'offset not predictable'
        raise ValueError(msg)
    
    # apply correct offset according to baseline
    raster_meta = asset_raster_metadata.copy()
    if baseline < 4.0:
        for key in raster_meta:
            for band in raster_meta[key]['raster:bands']:
                band['offset'] = 0.

    d_item = item.to_dict()
    for key in raster_meta:
        d_item['assets'][key] = d_item['assets'][key] | raster_meta[key]

    new_item = Item.from_dict(d_item)
    #new_item.ext.add('raster')
    RasterExtension.add_to(new_item)
    return new_item