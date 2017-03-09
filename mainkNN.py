import matplotlib.pyplot as plt
import numpy as np
import math


# import and preprocess data
iris_test = np.genfromtxt('IrisTestML.dt')
iris_train = np.genfromtxt('IrisTrainML.dt')

#labels
iris_test_labels= iris_test[:,2]
iris_train_labels= iris_train[:,2]
#data without labels
iris_test= iris_test[:,0:2]
iris_train= iris_train[:,0:2]

# script that contains all functions
from kNN import *

# task 1.1 - Nearest neighbour

print "\n ## 1.1 ##"
print "\nRunning kNN classifier on train and test for k=1,3,5 (raw data)...\n"

### training data
_,k1=knn(iris_train,iris_train,iris_train_labels,1)
_,k3=knn(iris_train,iris_train,iris_train_labels,3)
_,k5=knn(iris_train,iris_train,iris_train_labels,5)
print "Training data error:"
print "k=1", count_error(iris_train_labels, k1),"%"
print "k=3", count_error(iris_train_labels, k3),"%"
print "k=5", count_error(iris_train_labels, k5),"%"

### test data

_,k1=knn(iris_train,iris_test,iris_train_labels,1)
_,k3=knn(iris_train,iris_test,iris_train_labels,3)
_,k5=knn(iris_train,iris_test,iris_train_labels,5)
print "Validation data error:"
print "k=1", count_error(iris_test_labels, k1),"%"
print "k=3", count_error(iris_test_labels, k3),"%"
print "k=5", count_error(iris_test_labels, k5),"%"

# task 1.2  - Hyperparameter selection using cross-validation

k_max=25
print "\n ## 1.2 ##"
print "Performing K-fold cross validation to select k out of odd numbers between 1 and %d:" % k_max

# random permutation of dataset by indices
np.random.seed(12)
rand_perm=np.random.permutation(len(iris_train_labels))

kbestcv, _ =cv(iris_train, iris_train_labels, k_max, rand_perm) #this will also produce a plot if you uncomment plt.show in kNN.py
print "\n Best k: ", kbestcv, "\n"

# Rerunning kNN with kbest
print "Train error with k best (%d):" % kbestcv
_,k=knn(iris_train,iris_train,iris_train_labels,kbestcv)
print count_error(iris_train_labels, k),"%"

print "Test error with k best (%d):" % kbestcv
_,k=knn(iris_train,iris_test,iris_train_labels,kbestcv)
print count_error(iris_test_labels, k),"%"


#  task 1.3  -  Data normalization
print "\n ## 1.3 ##"
print "Performing K-fold cross validation on Normalised data to select k out of odd numbers between 1 and %d:\n" % k_max
# normalise the data
iris_train_norm, iris_test_norm = cent_and_norm(iris_train, iris_test)

#run cross-validation
kbestcv, _ =cv(iris_train_norm, iris_train_labels, k_max, rand_perm) #this will also produce a plot
print "\n Best k after normalisation: ", kbestcv, "\n"

# Rerunning kNN with kbest
print "Train error with k best (%d):" % kbestcv
_,k=knn(iris_train_norm,iris_train_norm,iris_train_labels,kbestcv)
print count_error(iris_train_labels, k),"%"

print "Test error with k best (%d):" % kbestcv
_,k=knn(iris_train_norm,iris_test_norm,iris_train_labels,kbestcv)
print count_error(iris_test_labels, k),"%"

