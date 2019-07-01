# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:27:18 2019

@author: Nikhil
"""

from sklearn.cluster import KMeans
from numpy import array
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance
from numpy.random import seed
from numpy.random import randint
from numpy import nanmean 
import timeit

def find_hartingan(X, n_clusters):
    labels = randint(0, n_clusters, len(X))
    centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])
    count=0
    converged=False
    while not converged:
        converged=True
        for index,tuple in enumerate(X):
            cluster_x_label = labels[index]
            X_minus_x = np.delete(X,(index),axis=0)
            labels_minus_x = np.delete(labels,(index),axis=0)
            cluster_minus_x= X[np.where(labels_minus_x==cluster_x_label)]
            cluster_minus_x_mean = np.mean(cluster_minus_x,axis=0,dtype=np.float64)  
            minimum_dist=9999999
            x_cluster=cluster_x_label
            for cluster_no in range(n_clusters):
                objective_cluster_index = cluster_no
                objective_cluster_label = labels
                objective_cluster_label[index]=cluster_no
                new_centers = np.array([[X[objective_cluster_label == i].mean(0)
                                for i in range(n_clusters)]])
                cluster=np.array([X[objective_cluster_label == i] for i in range(n_clusters)])
                new_centers=np.reshape(new_centers,(n_clusters,2))
                distance=0
                for centers_index in range(n_clusters):
                    for points in cluster[centers_index]:
                        distance=distance + dist(new_centers[centers_index],points)
                if(distance<=minimum_dist):
                    minimum_dist=distance
                    x_cluster=cluster_no
            if(x_cluster!=cluster_x_label):
                converged=False
                labels[index]=x_cluster
                centers = np.array([X[labels == i].mean(0)
                                  for i in range(n_clusters)])
        if(count%20==0):
            plt.scatter(data[:, 0], data[:, 1], c=labels,
            s=50, cmap='viridis')
            plt.show()    #Plot per iteration
        if(count==100):   #issue with convergence and hence forced break
            break
        count=count+1    
    return centers, labels
        
        

def find_loyd(X, n_clusters, rseed=2):
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    
    while True:
        labels = pairwise_distances_argmin(X, centers)
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])
        if np.all(centers == new_centers):
            break
        centers = new_centers
    
    return centers, labels


def find_macqueen(X,n_clusters,rseed=2):
    labels=np.zeros(len(X))
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    while(1):
        for index,tuple in enumerate(X):
            new_centers = centers
            cluster_number,winner_centroid = calc_pairwise_distance(tuple,new_centers)
            labels[index]=cluster_number
            n_count=(labels==cluster_number).sum()
            new_centers[cluster_number]=new_centers[cluster_number] + ((tuple -  new_centers[cluster_number])/n_count)
        if np.all(new_centers==centers):
            break
        centers=new_centers
    return centers, labels
       
def calc_pairwise_distance(value,centers):
    min_center= centers[0]
    min_distance= 99999999
    cluster_no=-1
    for index,center in enumerate(centers):
        dist = distance.euclidean(value,center)
        if(dist<=min_distance):
            min_distance= dist
            min_center= center
            cluster_no= index
    return cluster_no,min_center

def dist(x,y):   
    return np.sqrt(np.sum((x-y)**2))

data = np.loadtxt('data-clustering-1.csv', delimiter=',')
data = np.transpose(data)
average_time=0
for repeat in range(10):
   start = timeit.default_timer()
   #centers, labels = find_loyd(data, 3)
   centers,labels = find_macqueen(data,3)
   #centers,labels=find_hartingan(data,3)   #not a complete solution because of issue with convergence
   plt.scatter(data[:, 0], data[:, 1], c=labels,
            s=50, cmap='viridis')
   plt.show()
   stop = timeit.default_timer()
   average_time= average_time + (stop - start)
   
print('Time: ', average_time/10)
