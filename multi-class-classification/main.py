import math
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import Counter
from sklearn.model_selection import KFold
np.random.seed(0)

"""
# kNN algorith implementation
# The goal of this problem is to predict the forest cover type (column 55) from cartographic variables.
# There are 7 different cover types (classes) and 54 input variables.
# Here: a small subset of the data, 3750 training and 1250 test patterns (i.e. rows).

To run:  python3 main.py data/covtype_test.csv data/covtype_train.csv 
"""

def parse_args():
    """Parse command line arguments"""
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('test_data', help="Test dataset")
    parser.add_argument('train_data', help="Train dataset")
    args = parser.parse_args()
    return args


def main():
    """ Main function:
     - data centering and normalisation
     - k-Nearest Neighbour (kNN) classification
        - try on test data (raw and normalised)
        - try on train data (raw and normalised)
     - cross validation on train data to select the best k
     """

    kvalues = [1, 3, 5, 7, 9]
    args = parse_args()

    # check inputs
    test_data = args.test_data
    train_data = args.train_data
    cover_test = np.genfromtxt(test_data, delimiter=',')
    cover_train = np.genfromtxt(train_data, delimiter=',')

    # labels
    cover_test_labels = cover_test[:, 54]
    cover_train_labels = cover_train[:, 54]
    # data without labels
    cover_test = cover_test[:, 0:54]
    cover_train = cover_train[:, 0:54]

    # center and normalise data
    cover_train_norm, cover_test_norm = cent_and_norm(cover_train, cover_test)

    # create distance matrices
    # dist = calc_distance_matrix(cover_test, cover_train)
    # dist_norm = calc_distance_matrix(cover_test_norm, cover_train_norm)

    # create distance matrices FAST (this is more optimal than the option above)
    dist = distance_matrix_fast(cover_test, cover_train)
    dist_norm = distance_matrix_fast(cover_test_norm, cover_train_norm)

    # run kNN on the test data
    print("---- Running kNN of test data ----")
    for my_k in kvalues:
        out_raw = knn(dist, cover_train_labels, k=my_k)
        print("Raw data. k={}, accuracy: {}".format(my_k, estimate_accuracy(cover_test_labels, out_raw)))
        out_norm = knn(dist_norm, cover_train_labels, k=my_k)
        print("Norm data. k={}, accuracy: {}".format(my_k, estimate_accuracy(cover_test_labels, out_norm)))
    exit()
    dist_train = distance_matrix_fast(cover_train, cover_train)
    dist_norm_train = distance_matrix_fast(cover_train_norm, cover_train_norm)

    # on the train data
    print("---- Running kNN of train data ----")
    for my_k in kvalues:
        out_raw = knn(dist_train, cover_train_labels, k=my_k)
        print("Raw data. k={}, accuracy: {}".format(my_k, estimate_accuracy(cover_train_labels, out_raw)))
        out_norm = knn(dist_norm_train, cover_train_labels, k=my_k)
        print("Norm data. k={}, accuracy: {}".format(my_k, estimate_accuracy(cover_train_labels, out_norm)))

    # Cross validation
    print("---- Running 5-fold Cross Validation kNN to identify the best k ----")
    rand_perm = np.random.permutation(len(cover_train_labels))
    k_best = cross_validation_best_k(cover_train, cover_train_labels, rand_perm, kvalues)
    print("Best k: {}".format(k_best))


def rescale(data_matrix, means, stdevs):
    """ Rescales the input data so that each column has mean 0 and standard deviation 1;
     leaves alone columns with no deviation"""

    def rescaled(i, j):
        if stdevs[j] > 0:
            return (data_matrix[i][j] - means[j]) / stdevs[j]
        else:
            return data_matrix[i][j]

    rows, cols = data_matrix.shape
    new_matrix = [[rescaled(i, j)
                   for j in range(cols)]
                  for i in range(rows)]
    return new_matrix


def cent_and_norm(train, test):
    """Center and normalise the data"""
    rows, cols = train.shape
    means = [np.mean([row[j] for row in train]) for j in range(cols)]
    stdevs = [np.std([row[j] for row in train]) for j in range(cols)]

    train_cent_norm = rescale(train, means, stdevs)
    test_cent_norm = rescale(test, means, stdevs)

    return np.array(train_cent_norm), np.array(test_cent_norm)


def euclidean_distance(instance1, instance2):
    """Calculate Euclidean distance"""
    distance = 0
    for x in range(len(instance1)):
        onesum = pow((instance1[x] - instance2[x]), 2)
        distance += onesum
    return math.sqrt(distance)


def calc_distance_matrix(test, train):
    """Creating a distance matrix:
        (compare each test case to all train cases)"""
    distance_matrix = []
    for i in test:
        row = []
        for j in train:
            dist = euclidean_distance(i, j)
            row.append(dist)
        distance_matrix.append(row)
    print("Created distance matrix")
    return np.array(distance_matrix)


def distance_matrix_fast(test, train):
    """ Create distance matrix fast, without separate distance function"""

    # subtract each test vector from the entire training matrix.
    # this uses numpy addition broadcasting: as long as the dimensions match up,
    # numpy knows to do a row-wise subtraction if the element on the right is 1-D
    # After this subtraction, simply element-wise square and sum along the column
    # dimension to get a single row of the distance matrix for test vector i

    dists = np.zeros((len(test), len(train)))
    for i in range(len(test)):
        dists[i, :] = np.sum((train - test[i, :]) ** 2, axis=1)

    return np.sqrt(dists)


def knn(distance_matrix, trainlabels, k):
    """k-Nearest Neighbour algorithm implementation:
    Assigns labels to the test data by training on labels from the train data"""

    # determining the labels
    data_w_labels = []

    for row in distance_matrix:
        # add labels to each row in matrix
        vector_labelled = zip(row, trainlabels)  # tuple of one row of distances and label
        # sort row by distance (ascending)
        vector_labelled_sorted = sorted(vector_labelled, key=lambda x: x[0])
        data_w_labels.append(vector_labelled_sorted)

    # kNN starts here
    predicted_types = []
    line_covered = 1
    for index, row in enumerate(data_w_labels):
        # print("row:", index+1)
        k_nearest_labels = []
        for _, label in row[:k]:
            k_nearest_labels.append(label)
            l = len(k_nearest_labels)

        # print ("length of comparison is", l)
        while l > 0:
            # print ("testing line no.: " + str(line_covered))
            type_counts = Counter(k_nearest_labels)
            # one most common types, no of these (out of k)
            type, most_common_count = type_counts.most_common(1)[0]
            num_types = len([count for count in type_counts.values() if count == most_common_count])

            if num_types == 1:
                predicted_types.append(type)
                # print ("line %s done" % line_covered)
                break
            else:
                k_nearest_labels = k_nearest_labels[:-1]  # try again without the farthest
                l -= 1
        line_covered += 1
        # print ("######################################", "\n")

    return predicted_types


def estimate_accuracy(test_labels, predicted):
    """Return accuracy percentage"""
    error = 0
    for i in range(len(test_labels)):
        if test_labels[i] == predicted[i]:
            pass
        else:
            error += 1
    accuracy = ((1 - error / len(test_labels)) * 100)  # report accuracy
    return accuracy


def cross_validation_best_k(train, trainlabels, rand_perm, kvalues):
    """Performs 5-fold cross validation in training data and returns the best (most accurate) k for kNN"""
    # split indices of train data in 5 folds
    kf = KFold(n_splits=5)

    trains = []
    tests = []
    # for each fold randomise the indices
    for train_index, test_index in kf.split(trainlabels):
        # res tuple will hold randomised indices for each (5) pair of train/test data
        res = (rand_perm[train_index], rand_perm[test_index])
        # splitting train and test data indices  into two lists
        trains.append(res[0])
        tests.append(res[1])

    # data in folds
    # using permutated indices to split the data into 10 variables
    # (5 train sets, 5 test sets)
    train1, test1 = train[trains[0], :], train[tests[0], :]
    train2, test2 = train[trains[1], :], train[tests[1], :]
    train3, test3 = train[trains[2], :], train[tests[2], :]
    train4, test4 = train[trains[3], :], train[tests[3], :]
    train5, test5 = train[trains[4], :], train[tests[4], :]

    # for both train and test data,
    # create a subset of true labels for indexes of cases in each fold
    labels_list = np.array(trainlabels).tolist()
    train_label_list = [[labels_list[i] for i in fold] for fold in trains]
    test_label_list = [[labels_list[i] for i in fold] for fold in tests]

    # run kNN on each train+test split, and estimate accuracy for each k
    fold1 = knn_calc_accuracy(kvalues, train1, test1, train_label_list[0], test_label_list[0])
    fold2 = knn_calc_accuracy(kvalues, train2, test2, train_label_list[1], test_label_list[1])
    fold3 = knn_calc_accuracy(kvalues, train3, test3, train_label_list[2], test_label_list[2])
    fold4 = knn_calc_accuracy(kvalues, train4, test4, train_label_list[3], test_label_list[3])
    fold5 = knn_calc_accuracy(kvalues, train5, test5, train_label_list[4], test_label_list[4])

    # combine accuracies for different k from each split into a matrix
    fold_matrix = []
    fold_matrix.extend([fold1] + [fold2] + [fold3] + [fold4] + [fold5])
    fold_matrix = np.array(fold_matrix)

    # get mean accuracy for each k
    accuracy_means = np.mean(fold_matrix, axis=0)
    # get the best perfoming k
    kbest_index = np.array(accuracy_means).argmax()
    kbest = kvalues[kbest_index]

    return kbest


def knn_calc_accuracy(kvalues, train, test, labels_train, labels_test):
    """Run kNN for each k and estimate accuracy"""

    # run kNN
    fold_predictedlabels = []  # list of 5 lists of predictions (for 1,3,5,7,9)
    dist = distance_matrix_fast(test, train)
    for k in kvalues:
        out = knn(dist, labels_train, k)
        fold_predictedlabels.append(out)

    # estimate prediction error
    fold_errors = []
    for prediction in fold_predictedlabels:
        error = estimate_accuracy(labels_test, prediction)
        fold_errors.append(error)

    return fold_errors


if __name__ == '__main__':
    main()
