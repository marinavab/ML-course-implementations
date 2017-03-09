import numpy as np
import matplotlib.pylab as plt


input=np.loadtxt('ML2016TrafficSignsTrain.csv', delimiter=',')
#
# print input.shape #(1275L, 1569L)
labels = input[0:1275,1568]
#data = input[0:1275,0:1568]

# assign colors to labels for plotting
col_labels=[]
names=[]
for i in labels:
    if i>18 and i<31 or i==11:
        col_labels.append('deeppink')
    elif i ==12:
        col_labels.append('dodgerblue')
    elif i ==13:
        col_labels.append('slateblue')
    elif i ==14:
        col_labels.append('cyan')
    else:
        col_labels.append('pink')

cols=['deeppink','dodgerblue','slateblue','cyan','pink']
classes=['Up triangle', 'Diamond', 'Down triangle', 'Octagon', 'Round']


def apply_pca(rawdata):
    """PCA on the dataset that projects every data point onto the first 2 principal components"""

    #rawdata=np.transpose(rawdata)
    #center data
    items, features = rawdata.shape
    mean_vector = np.mean(rawdata, axis=0)
    rep_data=np.tile(mean_vector, (items,1))
    c_data=rawdata-rep_data
    #OR
    #c_data=rawdata-mean_vector

    #compute the covariance of the dataset
    #covariance_matrix= np.cov(c_data) #same as:
    covariance_matrix = (c_data).T.dot((c_data)) / (features-1)
    evals, evecs = np.linalg.eig(covariance_matrix)

    # the column evecs[:,i] is the ith unit eigenvector
    # the vector evals contains the eigenvalues, ie the projected variances of the eigenvectors

    evals_r = np.real(evals)
    evecs_r = np.real(evecs)
    # returns real parts of evals and evecs
    # evals and evecs of a covariance matrix are always real,
    #  but in python they are represented as complex

    #sort evals
    idx = np.argsort(evals_r)
    # in descending order
    idx = np.flipud(idx)
    evals_sorted =evals_r[idx]
    evecs_sorted = evecs_r[:,idx]


    # show that the output is sorted by decreasing eigenvalues
    #and make a list of (eigenvalue, eigenvector) tuples
    eigen_pairs = []
    for i in range(len(evals)):
        eigen_pairs.append((np.abs(evals[i]), evecs[:,i]))
    #for i in eigen_pairs:
        #print(i[0])

    # extracting the first 2 PC
    projected_matrix = np.hstack((eigen_pairs[0][1].reshape(features, 1),
                                  eigen_pairs[1][1].reshape(features, 1)))

    Y = np.dot(rawdata, projected_matrix)

    return evals_sorted,evecs_sorted,Y









########################### everything below is in the main function! below is just a backup
















#evals,evecs,Y = apply_pca(data)

# #variance vs PC
# # plt.plot(evals)
# # plt.title('Plot of projected variance on each principal component')
# # plt.xlabel('PCs in descending order')
# # plt.ylabel('Projected variance')
# # plt.show()
#
# #hom may PCs do i need to explain 90% variance?
#
# #cumulative varince in %
# #c_var = np.cumsum(evals/np.sum(evals))  #cumsum takes this one plus all previous ones
#
# over_90=[i for i,v in enumerate(c_var) if v > 0.9]
# ninety=over_90[0]+1
# print ninety,"PCs are needed to explain 90% variance"
#
# # plt.plot(c_var)
# # plt.ylabel('Cumulative variance')
# # plt.xlabel('Number of PCs')
# # plt.title('Cumulative variance vs PC')
# # plt.show()
# #explains varince by all pcs

#
# from kmeans import *
# initialIndices=np.array([0,1,2,3])
# k=4
# ###initialIndices=np.array(random.sample(range(0, len(data)), k)) #if want random
#
# new_centers,_ =kmeans(data,initialIndices,k)
#
# #adding centeres as new datapoints
# data2 = np.vstack([data, new_centers])
# print data2.shape
#
# _,_,Y_centers=apply_pca(data2)
#
# col_labels.extend(['yellow','yellow','yellow','yellow'])
#
# #####plotting
# PC1_centers=Y_centers[:,0]
# PC2_centers=Y_centers[:,1]
# print Y_centers.shape
# print len(col_labels)
#
#
# for i in range(len(col_labels)):
#     if i < 1275:
#         plt.scatter(PC1_centers[i],PC2_centers[i],color=col_labels[i])
#     else:
#         plt.scatter(PC1_centers[i],PC2_centers[i],color=col_labels[i], marker='s', s=50)
#
# import matplotlib.patches as mpatches
# recs = []
# for i in range(0,len(classes)):
#     recs.append(mpatches.Rectangle((0,0),1,1,fc=cols[i]))
# plt.legend(recs,classes,loc=2)
# plt.title('PCA Traffic Signs dataset')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.show()
#
