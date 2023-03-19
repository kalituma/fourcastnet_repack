import cdsapi
import numpy as np
from pathlib import Path

def build_request_params(year:int, variables:list, area:list, **kwargs):

    assert len(variables) > 0
    assert len(area) == 4

    year = str(year)
    area = [str(boundary) for boundary in area]

    params = {
        'product_type' : 'reanalysis',
        'format' : 'netcdf',
        'variable' : variables,
        'year' : year,
        'area' : area,
        'month' : [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'day' : [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': [
            '00:00', '01:00', '02:00', '03:00',
            '04:00', '05:00', '06:00', '07:00',
            '08:00', '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00', '15:00',
            '16:00', '17:00', '18:00', '19:00',
            '20:00', '21:00', '22:00', '23:00',
        ],
    }

    if 'pressure_level' in kwargs:
        params['pressure_level'] = str(kwargs['pressure_level'])

    return params


def request_download_era5(url, save_path, client, params):
    client.retrieve(
        url,
        params,
        save_path
    )

def download_1hr_data():
    base_path = Path('/home/khw/mount/_docandria1/data/ERA5/plevel/1hr/')
    base_path.mkdir(exist_ok=True, parents=True)

    variables = ['u_component_of_wind', 'v_component_of_wind', 'geopotential', 'temperature', 'relative_humidity']
    years = np.arange(2001, 2016)
    pressure_levels = [1000, 850, 750, 500]
    area = [47, 120, 28, 138]  # north, east, south, west
    url = "reanalysis-era5-pressure-levels"
    client = cdsapi.Client()

    for year in years:
        for pressure_level in pressure_levels:
            nc_file = f"u_v_z_t_rh_pressure_level_{pressure_level}_{year}.nc"
            save_path = base_path / nc_file
            params = build_request_params(year, variables=variables, area=area, pressure_level=pressure_level)
            request_download_era5(url=url, save_path=save_path, client=client, params=params)
            print(f'downloaded in {str(save_path)}')

def download_single_data():
    base_path = Path('/home/khw/mount/_docandria1/data/ERA5/plevel/single/')
    base_path.mkdir(exist_ok=True, parents=True)

    variables = ['total_column_water_vapour', 'total_precipitation']
    years = np.arange(2001, 2016)
    area = [47, 120, 28, 138]  # north, east, south, west
    url = "reanalysis-era5-single-levels"
    client = cdsapi.Client()

    for year in years:
            nc_file = f"tcwv_{year}.nc"
            save_path = base_path / nc_file
            params = build_request_params(year, variables=variables, area=area)
            request_download_era5(url=url, save_path=save_path, client=client, params=params)
            print(f'downloaded in {str(save_path)}')

if __name__ == '__main__':
    download_1hr_data()
    download_single_data()



