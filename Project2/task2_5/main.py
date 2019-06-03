import numpy as np
from Project2.task2_5.custom_KDTree import custom_KDTree
import os
import time
from sklearn.neighbors import KDTree


def read_dat_file(filename = '../data/data2-train.dat'):
    dt = np.dtype([("X", np.float), ("W", np.float), ("C", np.float)])
    data = np.loadtxt(filename, dtype=dt, comments="#", delimiter=None)
    data = np.array([[d[0], d[1], d[2]] for d in data])

    return data


def runtime_eval():
    data_root = ""
    train_file = os.path.join(data_root, '../data/data2-train.dat')
    test_file = os.path.join(data_root, "../data/data2-test.dat")
    train_data = read_dat_file(train_file)
    test_data = read_dat_file(test_file)
    leaf_sizes = [2, 100, 5000, 10000]
    for leaf_size in leaf_sizes:
        kd_tree = KDTree(train_data, leaf_size=leaf_size)
        start_time = time.time()
        dist, ind = kd_tree.query(test_data, k=2)
        end_time = time.time()
        print(
            "KD Tree with {} leaves, Time taken to get nearest neighbours: {} second(s)".format(
                leaf_size, end_time - start_time
            )
        )


if __name__ == "__main__":
    runtime_eval()

    data = read_dat_file()

    kd_tree = custom_KDTree(max_depth=3)

    kd_tree.create_tree(data, splitting_dim_mode=0, split_point_mode=0)
    kd_tree.draw('mode_0_0.png')

    kd_tree.create_tree(data, splitting_dim_mode=1, split_point_mode=0)
    kd_tree.draw('mode_1_0.png')

    kd_tree.create_tree(data, splitting_dim_mode=0, split_point_mode=1)
    kd_tree.draw('mode_0_1.png')

    kd_tree.create_tree(data, splitting_dim_mode=1, split_point_mode=1)
    kd_tree.draw('mode_1_1.png')

