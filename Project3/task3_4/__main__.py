import argparse
import numpy as np
import matplotlib.pyplot as plt
from neuron import Neuron

parser = argparse.ArgumentParser(description='Non-monotonous neurons.')
parser.add_argument('--data',  default= 'data/' ,help='File data path')
args = parser.parse_args()


def plot_data(x, y, neuron, figname):
    color = [0.5 if label == -1 else -0.5 for label in y]
    fig, ax = plt.subplots()
    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    cmap = plt.get_cmap('RdYlBu')

    print(cmap(0), cmap(0.5), cmap(1))
    # generate 2 2d grids for the x & y bounds
    ygrid, xgrid = np.mgrid[slice(x[1, :].min() - 0.05, x[1, :].max() + 0.05, 0.01),
                    slice(x[0, :].min() - 0.02, x[0, :].max() + 0.02, 0.01)]

    yravel = ygrid.ravel()
    xravel = xgrid.ravel()
    temp = np.vstack((xravel, yravel)).T

    colors = np.ones(ygrid.shape)


    for i, row in enumerate(temp):
        indx = i % xgrid.shape[1]
        indy = int(i / xgrid.shape[1])

        out = neuron.activation_function(row)
        if out < 0:
            colors[indx][indy] = 0.4
        elif out >= 0:
            colors[indx][indy] = 1

        colors[0][0] = 0.9
        colors[90][90] = 0.8
        colors[200][0] = 0.3
        colors[0][200] = 0.2
        colors[200][200] = 0.1

    ax.pcolormesh(xgrid, ygrid, colors, cmap=cmap,  alpha=0.7)
    plt.scatter(x[0, :], x[1, :], c=color, s=60)
    plt.savefig(figname)
    plt.show()


if __name__ == '__main__':
    neuron = Neuron()
    data_path = args.data
    x = np.genfromtxt(data_path + 'xor-X.csv', delimiter=', ', dtype=np.float)
    y = np.genfromtxt(data_path + 'xor-y.csv', delimiter=', ', dtype=np.float)
    plot_data(x, y, neuron, 'before_training.png')
    data = list(zip(x.T, y))
    loss = neuron.train(data)
    predicted = neuron.test(data)
    print(neuron.loss_vectorized(predicted, y))
    plot_data(x, y, neuron, 'after_training.png')


