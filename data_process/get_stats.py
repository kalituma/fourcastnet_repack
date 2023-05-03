import os
import re
from pathlib import Path
import numpy as np
import h5py
from data_process.io_process import read_array_from_raster, write_tif, get_files_recursive

def export_h5_to_tif(train_path, years, out_root):
    Path(out_root).mkdir(parents=True, exist_ok=True)

    for ii, year in enumerate(years):
        with h5py.File(os.path.join(train_path, f'{year}.h5'), 'r') as f:
            time, var_size = f['fields'].shape[0:2]
            for t in range(30):
                for v in range(var_size):
                    out_path = os.path.join(out_root, f'{year}_{t}_{v}.tif')
                    write_tif(out_path, f['fields'][t][v])
                print()


def get_stats(train_path, years):

    global_means = np.zeros((1, 21, 1, 1))
    global_stds = np.zeros((1, 21, 1, 1))
    # time_means = np.zeros((1, 21, 721, 1440))

    for ii, year in enumerate(years):
        with h5py.File(os.path.join(train_path, f'{year}.h5'), 'r') as f:
            rnd_idx = np.random.randint(0, 1460 - 500)
            global_means += np.mean(f['fields'][rnd_idx:rnd_idx + 500], keepdims=True, axis=(0, 2, 3))
            global_stds += np.var(f['fields'][rnd_idx:rnd_idx + 500], keepdims=True, axis=(0, 2, 3))

    global_means = global_means / len(years)
    global_stds = np.sqrt(global_stds / len(years))
    # time_means = time_means / len(years)

    np.save(f'{os.path.basename(train_path)}/global_means.npy', global_means)
    np.save(f'{os.path.basename(train_path)}/global_stds.npy', global_stds)
    # np.save(f'{os.path.basename(train_path)}/time_means.npy', time_means)

    print("means: ", global_means)
    print("stds: ", global_stds)

def get_stats_tif(train_path):
    global_means = np.zeros((1, 21, 1, 1))
    global_stds = np.zeros((1, 21, 1, 1))

    tif_list = sorted(get_files_recursive(train_path, '*.tif'))
    w, h = 0, 0
    for f in tif_list:
        basename_stem = os.path.splitext(os.path.basename(f))[0]
        _, time, var = basename_stem.split("_")[0:3]
        var = int(var)

        arr = read_array_from_raster(f)
        h, w = arr.shape
        global_means[0][var][0][0] += np.sum(arr)

    global_means /= 30 * w * h

    for f in tif_list:
        basename_stem = os.path.splitext(os.path.basename(f))[0]
        _, time, var = basename_stem.split("_")[0:3]
        var = int(var)

        arr = read_array_from_raster(f)
        global_stds[0][var][0][0] += np.sum(np.power(arr - np.squeeze(global_means)[var], 2))

    global_stds /= 30 * w * h
    global_stds = np.sqrt(global_stds)

    np.save(f'{Path(train_path).parent.absolute()}/global_means.npy', global_means)
    np.save(f'{Path(train_path).parent.absolute()}/global_stds.npy', global_stds)
    # np.save(f'{os.path.basename(train_path)}/time_means.npy', time_means)

    print("means: ", global_means)
    print("stds: ", global_stds)



if __name__ == '__main__':
    train_path = '/home/khw/mount/_docandria1/data/FCN_ERA5_data_v0/train/'
    out_path = '/home/khw/mount/_docandria1/data/FCN_ERA5_data_v0/train_tif/'
    years = [2018]
    get_stats_tif(out_path)
    # export_h5_to_tif(train_path, years, out_path)
