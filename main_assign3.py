import numpy as np
import matplotlib.pylab as plt


input=np.loadtxt('ML2016TrafficSignsTrain.csv', delimiter=',')

#print input.shape #(1275L, 1569L)
labels = input[0:1275,1568]
data = input[0:1275,0:1568]

### question 3.2

from pca import * # pca script contains main function for PCA

#get evals,evecs,and projection matrix
evals,evecs,Y = apply_pca(data)


### PCA PLOT

# separate two PCs into separate variables for convenience
# PC1=Y[:,0]
# PC2=Y[:,1]
# for i in range(len(col_labels)):
#     plt.scatter(PC1[i],PC2[i],color=col_labels[i])
#
# import matplotlib.patches as mpatches
# recs = []
# for i in range(0,len(classes)):
#     recs.append(mpatches.Rectangle((0,0),1,1,fc=cols[i]))
# plt.legend(recs,classes,loc=3)
# plt.title('PCA Traffic Signs dataset')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.show()


### eigenspectrum analysis

#variance vs PC

# plt.plot(evals)
# plt.title('Plot of projected variance on each principal component')
# plt.xlabel('PCs in descending order')
# plt.ylabel('Projected variance')
# plt.show()

#hom many PCs do i need to explain 90% variance?

# c_var = np.cumsum(evals/np.sum(evals))  #cumsum takes this one plus all previous ones
# over_90=[i for i,v in enumerate(c_var) if v > 0.9]
# ninety=over_90[0]+1
# print ninety,"PCs are needed to explain 90% variance"

#cumulative varince in %
# plt.plot(c_var)
# plt.ylabel('Cumulative variance')
# plt.xlabel('Number of PCs')
# plt.title('Cumulative variance vs PC')
# plt.show()



exit()



# question 3.3

from kmeans import *   # this script contains everything for kmeans algorithm running

k=4
initialIndices=np.array([0,1,2,3]) # first 4 data point will be initial clusters
###initialIndices=np.array(random.sample(range(0, len(data)), k)) #if want random


# run kmeans
new_centers,_ =kmeans(data,initialIndices,k)

#adding centeres as new datapoints to the original dataset
data2 = np.vstack([data, new_centers])
#print data2.shape #[1279,1568]


#running PCA on the new dataset
_,_,Y_centers=apply_pca(data2)

#adding color labels for the cluster centers
col_labels.extend(['yellow','yellow','yellow','yellow'])

#####plotting
PC1_centers=Y_centers[:,0]
PC2_centers=Y_centers[:,1]
print Y_centers.shape
print len(col_labels)

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