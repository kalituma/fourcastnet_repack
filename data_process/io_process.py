import numpy as np
import gzip
from osgeo import gdal
from pathlib import Path

def read_radar(file_path, height, width, crop=False):
    """
    read bin.gz rar file in as numerical arrays composed of radar values, radar boundary, map info.
    :param file_path: bin.gz file path
    :param height: height of radar
    :param width: width of radar
    :return: radar, radar boundary, map info
    """

    with gzip.open(file_path, 'r') as f:
        content = f.read()

        radar_values = np.flip(np.reshape(np.frombuffer(content[:height * width * 4], dtype=np.float32),
                                  newshape=(height, width)), axis=0)

        radar_bound = np.flip(np.reshape(np.frombuffer(content[height * width * 4:height * width * 5], dtype=np.int8),
                           newshape=(height, width)), axis=0)
        map_info = np.flip(np.reshape(np.frombuffer(content[height * width * 5:height * width * 6], dtype=np.int8),
                           newshape=(height, width)), axis=0)

    if crop:
        bounds = np.where(radar_bound > 0)
        height_min, height_max = bounds[0].min(), bounds[0].max()
        width_min, width_max = bounds[1].min(), bounds[1].max()

        return radar_values[height_min:height_max + 1, width_min:width_max + 1].copy(), \
               radar_bound[height_min:height_max + 1,width_min:width_max + 1].copy(), \
               map_info[height_min:height_max + 1,width_min:width_max + 1].copy()
    else:
        return radar_values.copy(), radar_bound.copy(), map_info.copy()

def read_array_from_raster(file_path):
    ds = gdal.Open(file_path)
    data_array = ds.ReadAsArray()
    ds = None
    return data_array

def get_files_recursive(root:str, pattern):
    p = Path(root).rglob(pattern)
    file_list = [str(x) for x in p if x.is_file()]
    return file_list

def write_tif(to_path, data_array, gdal_type='float'):
    data_array = data_array.squeeze()

    gdal_type_map = {
        'float': gdal.GDT_Float32,
        'uint': gdal.GDT_UInt16
    }

    driver = gdal.GetDriverByName('gtiff')
    data_array = np.expand_dims(data_array, axis=2)
    # data_array = np.transpose(data_array, axes=(1, 2, 0))
    channel = 1

    rad_ds = driver.Create(str(to_path), data_array.shape[1], data_array.shape[0], channel,
                           gdal_type_map[gdal_type])

    for i in range(1, data_array.shape[2] + 1):
        rad_ds.GetRasterBand(i).WriteArray(data_array[:, :, i - 1])
        # rad_ds.GetRasterBand(1).SetNoDataValue(-9999)
    rad_ds.FlushCache()  # Write to disk.
    rad_ds = None
