import scipy.spatial as sp
import matplotlib.pyplot as plt
import numpy as np
import time
import math

def main():
	train_data=data = np.genfromtxt('data2-train.dat',delimiter=None)
	test_data= np.genfromtxt('data2-test.dat',delimiter=None)
	kNearestNeighbour(1,train_data,test_data)
	kNearestNeighbour(3,train_data,test_data)
	kNearestNeighbour(5,train_data,test_data)
def convert_to_column(mat,i):
  return [float(row[i]) for row in mat]        

def kNearestNeighbour(k,trainData, testData):
	final = []
	print("The value of k = ",k)
	train_x = np.asarray(convert_to_column(trainData,0))
	train_y = np.asarray(convert_to_column(trainData,1))
	test_x = np.asarray(convert_to_column(testData,0))
	test_y = np.asarray(convert_to_column(testData,1))

	labels_train = np.asarray(convert_to_column(trainData,2))
	labels_test = np.asarray(convert_to_column(testData,2))

	
	accuracy = 0
	total_sum = 0
	start = time.time()
	for i in range(test_x.shape[0]):
		xi = test_x[i]
		yi = test_y[i]
		temp_list = []

		for j in range(train_x.shape[0]):
			xj = train_x[j]
			yj = train_y[j]
			label = labels_train[j]
			dist = sp.distance.euclidean([xi,yi],[xj,yj])
			temp_list.append([dist,label])
		temp_list = np.asarray(temp_list)
		temp_list = temp_list[np.argsort(temp_list[:,0])][0:k]
		result = 1.0 if sum(temp_list[:,1]) > 0 else -1.0
		if labels_test[i] == result :
			accuracy += 1.0
		total_sum += 1.0
	end = time.time()
	time_elapsed = end-start
	accuracy_rate=(accuracy / total_sum * 100.0)
	print "Time elapsed: %.4f seconds"%time_elapsed
	print "Accuracy rate:  %.3f%%" % accuracy_rate
	

main()
