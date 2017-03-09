from __future__ import division
import matplotlib.pyplot as plt
import numpy as np



# import and preprocess data
iris_test = np.genfromtxt('IrisTestML.dt')
iris_train = np.genfromtxt('IrisTrainML.dt')

def binary_data_separate_labels(data):

    print "Original data:",data.shape
    data_binary=[]
    for i in xrange(0,data.shape[0]):
        if data[i,2]==0 or data[i,2]==1:
            data_binary.append(data[i,:])
        else:   # if class=2, ignore
            pass
    y=np.array(data_binary)
    x=np.reshape(np.ravel(y), (len(y),3))
    print "Binary data:",x.shape

    feature_data=x[:,0:2]
    labels=x[:,2]
    return feature_data,labels

# preparing data
train_data, train_labels=binary_data_separate_labels(iris_train)
test_data, test_labels = binary_data_separate_labels(iris_test)

# turning 0 and 1 to -1 and +1
train_labels = (train_labels-0.5)*2
test_labels = (test_labels-0.5)*2


#### data normalisation!
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
    return (np.array(train_cent_norm), np.array(test_cent_norm))
train_data, test_data = cent_and_norm(train_data,test_data)



### funcrions used in main

def logistic(input):
    out = np.exp(input)/(1+np.exp(input))
    #out=1 / (1 + np.exp(-input)) #OR
    return out

def logistic_insample(X, y, w):
    N, features = X.shape
    Ein = 0 #sum in the loop
    for n in range(N):
        Ein = Ein + (1/N)*np.log(1/logistic(y[n]*np.dot(w,X[n,:]))) #feeding it vectors
    return Ein

def logistic_gradient(X, y, w):
    N, _ = X.shape
    g = 0*w # sum in the loop
    for n in range(N):
        g = g + ((-1/N)*y[n]*X[n,:])*logistic(-y[n]*np.dot(w,X[n,:]))
    return g

def log_reg(Xorig, y, max_iter, threshold):

    num_pts, features = Xorig.shape
    onevec = np.ones((num_pts,1))
    X = np.concatenate((onevec, Xorig), axis = 1)

    # Initialize learning rate for gradient descent
    learningrate = 0.1

    # Initialize weights at time step 0
    w = 0.1*np.random.randn(features + 1)

    # Compute value of logistic log likelihood
    previous_likelihood = logistic_insample(X,y,w)

    difference = threshold + 1

    # Keep track of function values
    E_in =[previous_likelihood]

    num_iter = 0

    while (difference > threshold) and (num_iter < max_iter):
        g = np.array(logistic_gradient(X,y,w))
        w = w - learningrate * g
        temp = logistic_insample(X,y,w)
        difference = np.abs(temp - previous_likelihood)
        previous_likelihood = temp
        E_in.append(previous_likelihood)
        num_iter += 1


    print "Stopped after %d iterations" % num_iter
    return w, E_in


w, E = log_reg(train_data, train_labels, max_iter=100000, threshold=1e-5)  #if put threshold at 0, will go through all iterations
#print (E)
print(w)



def log_pred(Xorig, w):
    #X is a dxN data matrix of input variables
    num_pts,features =Xorig.shape
    onevec=np.ones((num_pts,1))
    X=np.concatenate((onevec,Xorig),axis=1)
    N, _ =X.shape
    P=np.zeros(N)
    for n in range(N):
        arg=np.exp(np.dot(w,X[n,:]))
        prob_i=arg/(1+arg)
        P[n]=prob_i
        #print P[n]

    Pthresh=np.round(P)   #0/1 class labels because go from probalility
    #print Pthresh
    pred_classes=(Pthresh-0.5)*2 #convert to +1/-1
    return P, pred_classes



############################################### calls below:

#prediction on train
P_train, pred_classes_train = log_pred(train_data, w)
errors_train = np.sum(np.abs(pred_classes_train - train_labels)/2)
N_train = len(train_labels)
error_rate_train  = errors_train/N_train
print "train data predictions:",(error_rate_train , errors_train )

#prediction on test
P_test, pred_classes_test = log_pred(test_data, w)
errors_test = np.sum(np.abs(pred_classes_test - test_labels)/2)
N_test = len(test_labels)
error_rate_test = errors_test/N_test
print "test data predictions:",(error_rate_test, errors_test)

# extracting weights for plotting the boundary line
w0 = w[0]; w1 = w[1]; w2 = w[2]
X = np.array([-3,4])
f= ((-w0-w1*X)/w2)


col_labels_train =[]
for i in pred_classes_train:
    if i==-1:
        col_labels_train .append('deeppink')
    elif i ==1:
        col_labels_train .append('dodgerblue')

col_labels_test=[]
for i in pred_classes_test:
    if i==-1:
        col_labels_test.append('deeppink')
    elif i ==1:
        col_labels_test.append('dodgerblue')


plt.figure(facecolor="white")
for i in range(len(col_labels_train)):
    plt.scatter(train_data[i,0],train_data[i,1],marker='+', s =50, color=col_labels_train [i])

for i in range(len(col_labels_test)):
    plt.scatter(test_data[i,0],test_data[i,1],  marker='o', s= 50, facecolors='none', edgecolors=col_labels_test[i])

plt.plot(X, f, 'k-')

plt.title("Logistic regression on Iris train (+) and test (o) data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.xlim([-3,4])
plt.ylim([-4,4])
plt.show()





######## separate plots



# col_labels_train =[]
# for i in pred_classes_train :
#     if i==-1:
#         col_labels_train .append('deeppink')
#     elif i ==1:
#         col_labels_train .append('dodgerblue')
#
# plt.figure(facecolor="white")
# for i in range(len(col_labels_train )):
#     plt.scatter(train_data[i,0],train_data[i,1],  color=col_labels_train [i], marker='v')
#
# plt.title("Logistic regression on Iris train data")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()







# col_labels_test=[]
# for i in pred_classes_test:
#     if i==-1:
#         col_labels_test.append('deeppink')
#     elif i ==1:
#         col_labels_test.append('dodgerblue')
#
# plt.figure(facecolor="white")
# for i in range(len(col_labels_test)):
#     plt.scatter(test_data[i,0],test_data[i,1],  color=col_labels_test[i], marker='v')
# plt.title("Logistic regression on Iris test data")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()