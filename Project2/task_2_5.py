import numpy as np
from sklearn.neighbors import KDTree
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import os
import time

class custom_KDTree(object):
    def __init__(self):
        """
        Initialize the KD tree and parameters
        """
        pass

    def create_tree(self, data, leaf_size, metric="Euclidean"):
        """
        Create the tree
        """
        pass

    def get_KNearest_Neigbour(self, data, k=2):
        """
        Get K nearest neighbours for input data
        """
        pass


def read_dat_file(filename):
    """
    Helper function to read data files
    """
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
    leaf_sizes = [2, 100, 5000, 10000]
    for leaf_size in leaf_sizes:
        kd_tree = KDTree(train_data, leaf_size=leaf_size)
        start_time = time.time()
        dist, ind = kd_tree.query(test_data, k=2)
        end_time = time.time()
        # hours = int((end_time - start_time) / 3600)
        # minutes = int((end_time - start_time) / 60) - (hours * 60)
        # seconds = int((end_time - start_time) % 60)
        print(
            "KD Tree with {} leaves, Time taken to get nearest neighbours: {} second(s)".format(
                leaf_size, end_time - start_time
            )
        )

    # fig = plt.figure()
    # ax = plt.axes(projection="3d")
    # ax.scatter3D(train_data[:, 0], train_data[:, 1], train_data[:, 2], "red")
    # plt.show()


if __name__ == "__main__":
    main()
