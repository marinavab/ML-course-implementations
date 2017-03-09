from __future__ import division
import numpy as np

import matplotlib.pyplot as plt
import random
import math

### helper functions

def euclideanDistance(instance1, instance2):
    """Calculates Eucledian distance between datapoints"""
    distance = 0
    for x in range(len(instance1)):
        onesum=pow((instance1[x] - instance2[x]), 2)
        distance += onesum
    return math.sqrt(distance)


def rescale(data_matrix, means, stdevs):
    """Rescales the input data so that each column  has mean 0 and standard deviation 1; leaves alone columns with no deviation"""

    def rescaled(i, j):
        if stdevs[j] > 0:
            return (data_matrix[i][j] - means[j]) / stdevs[j]
        else:
            return data_matrix[i][j]

    rows, cols = data_matrix.shape
    new_matrix=[[rescaled(i, j)
               for j in range(cols)]
               for i in range(rows)]
    return new_matrix


def cent_and_norm(train, test):
    """Center and normalise the data"""
    rows, cols = train.shape

    # get means and std of all features in train set
    means = np.mean(train,axis=0)
    stdevs = np.std(train,axis=0)

    #use train data means and stdvs to rescale both train and test
    train_cent_norm = rescale(train, means, stdevs)
    test_cent_norm = rescale(test, means, stdevs)

    print "Means and stds for train and test before normalisation:"
    print np.mean(train,axis=0)
    print np.mean(test,axis=0)
    print np.std(train,axis=0)
    print np.std(test,axis=0)
    print "Means and stds for train and test after normalisation:"
    print np.mean(train_cent_norm,axis=0)
    print np.mean(test_cent_norm,axis=0)
    print np.std(train_cent_norm,axis=0)
    print np.std(test_cent_norm,axis=0)

    return (train_cent_norm, test_cent_norm)

def count_error(test_labels, predicted):
    """Return error or accuracy percentage (comment out the other option)"""
    error = 0
    for i in range(len(test_labels)):
        if test_labels[i] == predicted[i]:
            pass
        else:
            error += 1
    #accuracy= ((1-error/len(test_labels))*100)      #report accuracy
    #return "%.2f" % accuracy
    errorp = (error / len(test_labels) * 100)     #report error
    return "%.2f" % errorp



### main functions

#knn algorithm
def knn(train, test, trainlabels, k):
    """MAIN function: Assigns labels to the test data by training on labels from the train data"""
    ##part 1 -creating a matrix
    matrix=[]
    for i in test:
        row=[]
        for j in train:
            dist = euclideanDistance(i, j)
            row.append(dist)
        matrix.append(row)

    ##part 2 -determining the labels
    withlabels=[]
    for row in matrix:
        zipped= zip(row, trainlabels)
        withlabels.append(zipped)

    sortedwithlabels=[]
    for i in withlabels:
        x = sorted(i, key=lambda x: x[0]) #tuple of one row of distances and label #sorted by distance (ascending)
        sortedwithlabels.append(x)

    predicted_type = []
    line_covered = 1
    for index, row in enumerate(sortedwithlabels):
        #print "row:", index+1
        k_nearest_labels = []
        for _, label in row[:k]:
            k_nearest_labels.append(label)
            l = len(k_nearest_labels)

        #print "length of comarison is", l
        from collections import Counter
        while l > 0:
            #print "testing line no.: " + str(line_covered)
            type_counts = Counter(k_nearest_labels)
            type, most_common_count = type_counts.most_common(1)[0]  # one most common type, no of these (out of k)
            num_label_types = len([count for count in type_counts.values() if count == most_common_count])

            if num_label_types == 1:
                predicted_type.append(type)
                #print "line %s done" % line_covered
                break
            else:
                k_nearest_labels = k_nearest_labels[:-1]  # try again without the farthest
                l -= 1
        line_covered += 1
        #print "######################################", "\n"

    matrix = np.array(matrix)
    return (matrix, predicted_type)


# cross validation

from sklearn.cross_validation import KFold
def cv(train, trainlabels,kmax, index):
    """Performs 5-fold cross validation in training data and returns the best (most accurate) k for kNN"""
    kvalues = range(1,kmax+1,2) #get odd k's
    train =np.array(train)
    kf = KFold(len(trainlabels), n_folds=5)     #split indices of train data in 5 folds

    trains=[]
    tests=[]
    #index=np.array(range(0,len(trainlabels)))
    for train_index, test_index in kf:      #for each fold
        res=(index[train_index], index[test_index])      #res will hold  indices for each (5) pair of train/test data
        trains.append(res[0])   #splitting train and test data indices  into two lists
        tests.append(res[1])

    #data in folds
    train1, test1 = train[trains[0],:], train[tests[0],:]   # split the data into 10 variables (5 train sets, 5 test sets)
    train2, test2 = train[trains[1],:], train[tests[1],:]
    train3, test3 = train[trains[2],:], train[tests[2],:]
    train4, test4 = train[trains[3],:], train[tests[3],:]
    train5, test5 = train[trains[4],:], train[tests[4],:]

    labelslist = np.array(trainlabels).tolist()

    train_label_list=[] ; test_label_list=[] ;   train_list=[];   test_list=[]

    #reordering labels according to permutation
    for fold in tests:
        for i in fold:
            test_list.append(labelslist[i])
        test_label_list.append(test_list)
        test_list=[]

    for fold in trains:
        for i in fold:
            train_list.append(labelslist[i])
        train_label_list.append(train_list)
        train_list=[]


    def knn_calc_accuracy(train,test,labels_train,labels_test):
        fold_predictedlabels=[]
        for k in kvalues:
            _,out=knn(train,test,labels_train,k)
            fold_predictedlabels.append(out)

        folderrors=[] #this will be a list of 5 lists (for 1,3,5,7,9 etc)
        for prediction in fold_predictedlabels:
            error=0
            for i in range(len(labels_test)):
                if labels_test[i] != prediction[i]:
                    error +=1
            errorm= error/len(labels_test)
            folderrors.append(errorm)
        return folderrors

    fold1= knn_calc_accuracy(train1,test1,train_label_list[0],test_label_list[0])
    fold2= knn_calc_accuracy(train2,test2,train_label_list[1],test_label_list[1])
    fold3= knn_calc_accuracy(train3,test3,train_label_list[2],test_label_list[2])
    fold4= knn_calc_accuracy(train4,test4,train_label_list[3],test_label_list[3])
    fold5= knn_calc_accuracy(train5,test5,train_label_list[4],test_label_list[4])

    fold_matrix=[]
    fold_matrix.extend([fold1]+[fold2]+[fold3]+[fold4]+[fold5])
    fold_matrix=np.array(fold_matrix)
    #print fold_matrix

    error_means=np.mean(fold_matrix,axis=0)
    print "Means for each k",error_means
    print "List of corresponding k-values:",kvalues
    kbest_index= np.array(error_means).argmin()
    kbest=kvalues[kbest_index]

    #plot error
    print "\nClose the plot to see the best K!\n"
    x_axis=range(0,13)
    plt.figure(facecolor="white")
    plt.scatter(x_axis,error_means)
    plt.title("Cross-validation error for different k")
    plt.xlabel("k")
    plt.ylabel("Mean error (averaged over the 5 folds)")
    plt.ylim([0,0.5])
    plt.xticks(x_axis, kvalues)
    plt.grid()
    #plt.show()

    return kbest, fold_matrix

