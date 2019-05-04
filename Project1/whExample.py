import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import norm

def plotData2D(X, filename=None):
    # create a figure and its axes
    fig = plt.figure()
    axs = fig.add_subplot(111)

    # see what happens, if you uncomment the next line
    # axs.set_aspect('equal')
    
    # plot the data 
    axs.plot(X[0,:], X[1,:], 'ro', label='data')

    # set x and y limits of the plotting area
    xmin = X[0,:].min()
    xmax = X[0,:].max()
    axs.set_xlim(xmin-10, xmax+10)
    axs.set_ylim(-2, X[1,:].max()+10)

    # set properties of the legend of the plot
    leg = axs.legend(loc='upper left', shadow=True, fancybox=True, numpoints=1)
    leg.get_frame().set_alpha(0.5)

    # either show figure on screen or write it to disk
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename, facecolor='w', edgecolor='w',
                    papertype=None, format='pdf', transparent=False,
                    bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
def fit_norm(mean, std_dev):
    data = np.linspace(int(mean-std_dev*3), int(mean+std_dev*3))
    pdf = sp.stats.norm.pdf(data, mean, std_dev)
    plt.plot(data, pdf, 'k', label='fitted normal', linewidth=2, color='orange')
    plt.plot(ws, np.zeros(ws.shape), 'ro', label='weights', color='blue')
    plt.title("Fit Normal Distribution to 1-D data")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    #######################################################################
    # 1st alternative for reading multi-typed data from a text file
    #######################################################################
    # define type of data to be read and read data from file
    dt = np.dtype([('w', np.float), ('h', np.float), ('g', np.str_, 1)])
    data = np.loadtxt('whData.dat', dtype=dt, comments='#', delimiter=None)

    # read height, weight and gender information into 1D arrays
    ws = np.array([d[0] for d in data])
    hs = np.array([d[1] for d in data])
    gs = np.array([d[2] for d in data])

    # task 1.2 (fitting normal to 1D data)
    mean, std_dev = sp.stats.norm.fit(ws)
    print('Before replacing outliers, Mean: {}, Standard Deviation:{}'.format(mean, std_dev))
    fit_norm(mean, std_dev)
    mean = np.mean(ws[ws>0])
    ws[ws==-1] = mean
    mean, std_dev = sp.stats.norm.fit(ws)
    print('After replacing outliers, Mean: {}, Standard Deviation:{}'.format(mean, std_dev))
    fit_norm(mean, std_dev)


    ##########################################################################
    # 2nd alternative for reading multi-typed data from a text file
    ##########################################################################
    # read data as 2D array of data type 'object'
    data = np.loadtxt('whData.dat',dtype=np.object,comments='#',delimiter=None)

    # read height and weight data into 2D array (i.e. into a matrix)
    X = data[:,0:2].astype(np.float)

    # read gender data into 1D array (i.e. into a vector)
    y = data[:,2]

    # Task 1.1 -- Removing the outliers
    X = X[X[:,0] > 0, :]
    
    # let's transpose the data matrix 
    X = X.T

    # now, plot weight vs. height using the function defined above
    plotData2D(X, '1.1/plotWH.pdf')

    # next, let's plot height vs. weight 
    # first, copy information rows of X into 1D arrays
    w = np.copy(X[0,:])
    h = np.copy(X[1,:])
    
    # second, create new data matrix Z by stacking h and w
    Z = np.vstack((h,w))

    # third, plot this new representation of the data
    plotData2D(Z, '1.1/plotHW.pdf')