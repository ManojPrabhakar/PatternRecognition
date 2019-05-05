import numpy as np
import matplotlib.pyplot as plt
import utils


def get_second_coordinates(x, p):
    temp = np.power(1 - np.power(np.abs(x), p), 1/float(p))
    return np.hstack((temp, - temp))


def plotting_norms():
    fig, axes = plt.subplots(1, 3)

    for ax in axes:
        ax.set_aspect('equal')

    x= np.linspace(-1, 1, 10000)
    x_copied = np.hstack((x, x))

    y_for_l1_norm = get_second_coordinates(x, 1)
    y_for_l2_norm = get_second_coordinates(x, 2)

    y_for_l1_2_norm = get_second_coordinates(x, 0.5)

    axes[0].set_title('L1-norm')
    axes[1].set_title('L2-norm')
    axes[2].set_title('L1/2-norm')

    axes[0].scatter(x_copied, y_for_l1_norm, s = 0.5)
    axes[1].scatter(x_copied, y_for_l2_norm, s = 0.5)
    axes[2].scatter(x_copied, y_for_l1_2_norm, s = 0.5)

    utils.show_or_save_plot('norms.pdf')


if __name__ == "__main__":
    plotting_norms()

"""
The L1/2 is not a norm because it does not satisfy the triangle inequality.  
"""