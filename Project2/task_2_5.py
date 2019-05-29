import numpy as np
from sklearn.neighbors import KDTree
import os
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


def read_dat_file(filename):
    dt = np.dtype([("X", np.float), ("W", np.float), ("C", np.float)])
    data = np.loadtxt(filename, dtype=dt, comments="#", delimiter=None)
    data = np.array([[d[0], d[1], d[2]] for d in data])
    return data


def main():
    data_root = "./data"
    train_file = os.path.join(data_root, "data2-train.dat")
    test_file = os.path.join(data_root, "data2-test.dat")
    train_data = read_dat_file(train_file)
    test_data = read_dat_file(test_file)
    # leaf_sizes = [2, 10, 30, 50, 100]
    leaf_sizes = [2]
    for leaf_size in leaf_sizes:
        kd_tree = KDTree(train_data, leaf_size=leaf_size)
        dist, ind = kd_tree.query(train_data[:1], k=2)
        # print(dist, ind)

    print(train_data.shape)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(train_data[:, 0], train_data[:, 1], train_data[:, 2], "red")
    plt.show()


if __name__ == "__main__":
    main()
