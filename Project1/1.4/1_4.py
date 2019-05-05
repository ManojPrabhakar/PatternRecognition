# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 20:41:15 2019

@author: sgang
"""

#import pylab
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, linalg, random, sqrt, inf

def plotUnitCircle(p,round_of):
    for x in np.arange(-1, 1, 0.01):
        for y in np.arange(-1, 1, 0.01):
            xdist = abs(0-x)
            ydist = abs(0-y)
            xp = (xdist)**(p)
            yp = (ydist)**(p)
            lp = (xp+yp)**(1./p)
            if(round(lp,round_of)==1):
                plt.plot(x,y,'bo')
    plt.axis([-2, 2, -2, 2])
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.show()
    plt.savefig("plot.png")
 
p=float(input("Enter p value for unit circle :"))
round_of= int(input("Enter round of value for unit circle :"))
plotUnitCircle(p,round_of)