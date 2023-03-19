import os
import numpy as np
import h5py

def get_stats(train_path, years):

    global_means = np.zeros((1, 21, 1, 1))
    global_stds = np.zeros((1, 21, 1, 1))
    time_means = np.zeros((1, 21, 721, 1440))

    for ii, year in enumerate(years):
        with h5py.File(os.path.join(train_path, f'{year}.h5'), 'r') as f:
            rnd_idx = np.random.randint(0, 1460 - 500)
            global_means += np.mean(f['fields'][rnd_idx:rnd_idx + 500], keepdims=True, axis=(0, 2, 3))
            global_stds += np.var(f['fields'][rnd_idx:rnd_idx + 500], keepdims=True, axis=(0, 2, 3))

    global_means = global_means / len(years)
    global_stds = np.sqrt(global_stds / len(years))
    time_means = time_means / len(years)

    np.save(f'{os.path.basename(train_path)}/global_means.npy', global_means)
    np.save(f'{os.path.basename(train_path)}/global_stds.npy', global_stds)
    np.save(f'{os.path.basename(train_path)}/time_means.npy', time_means)

    print("means: ", global_means)
    print("stds: ", global_stds)

if __name__ == '__main__':
    train_path = '/home/khw/mount/_docandria1/data/FCN_ERA5_data_v0/train/'
    years = [1979]
    get_stats(train_path, years)

