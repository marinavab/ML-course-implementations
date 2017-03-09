import numpy as np
import matplotlib.pylab as plt
import math
import random

# input=np.loadtxt('ML2016TrafficSignsTrain.csv', delimiter=',')
#
# print input.shape #(1275L, 1569L)
# labels = input[0:1275,1568]
# data = input[0:1275,0:1568]


#initialIndices=np.array([0,1,2,3])
#k=4

def euclideanDistance(instance1, instance2):
    distance = 0
    for x in range(len(instance1)):
        onesum=pow((instance1[x] - instance2[x]), 2)
        distance += onesum
    return math.sqrt(distance)


def clustering(data, centers):
    """Performing clustering of data points into k clusters"""
    cluster_dict= dict(enumerate(centers))      #creating a dictionary with number of k as keys
    distance_matrix=[]
    for i in data:
        datapoint = []
        for center in centers:
            for x in center:                        #this step is needed to get though double brackets
                dist = euclideanDistance(i, x)
                datapoint.append(dist)                        #contains distances i - x for each cluster center x
        distance_matrix.append(datapoint)       #contains distances to to the current center for each row

    indicesFori= np.array(distance_matrix).argmin(axis=1)
    for row, clusterNo in zip(data,indicesFori):
        try:
            cluster_dict[clusterNo].append(row)             #add the current row to the cluster dictionary (by its key)
        except KeyError:
            cluster_dict[clusterNo] = [row]                 #add the current row to the cluster dictionary (by its key)
    return cluster_dict, indicesFori


def relocate_mean(cluster):
    """Estimating the mean of a cluster"""
    cluster=np.array(cluster)
    new_center=[]
    new_center.append(np.mean(cluster, axis=0))
    new_center=np.array(new_center).tolist()
    return new_center[0]

def find_new_center(cluster_dict):
    """Calculating new cluster centers"""
    k=len(cluster_dict.keys())
    newcenters=[]
    for cluster_number,value in cluster_dict.items():
        newcenter = relocate_mean(value)
        newcenters.append(newcenter)
    new_centroids=[[] for i in range(k)]
    for i in range(k):
        new_centroids[i].append(newcenters[i])
    return new_centroids

def converged(final,previous,iterations):
    """Checking for convergence i.e. have the cluster centers moved"""
    list1= [val for sublist in final for val in sublist]        #flattening

    if len(list1)==1 and iterations>0:                  #this catches weird bug when k=1
        list2=[previous[0][0]]
    else:                                    # as normal
        list2= [val for sublist in previous for val in sublist]  #flattening as normal

    bool_vector=[]
    for i,j in zip(list1, list2):
        bool_vector.append(np.allclose(i,j))    #checking for similarity
        break
    if False in bool_vector:        #if there are no diffeneces (i.e. Fasle), then the function has converged
        return False
    else:
        return True


def kmeans(data, initialIndices,k):
    """The main function. Performs clustering and recolates the means until the function has converged and the means no longer move"""
    data = np.array(data).tolist()

    #original randomly picked centroids; this variable will be used for new changing centroids
    test_centers = [[] for i in range(k)]  #creating double bracket [[list]] list; this is needed for simplifying subsequent steps
    for i in range(k):
        test_centers[i].append(data[initialIndices[i]])

    #dummy centers for  the first iteration only
    initialIndices2=np.array(random.sample(range(0, len(data)), k))
    current_centers = [[] for i in range(k)]      #creating double bracket [[list]] list; this is needed for simplifying subsequent steps
    for i in range(k):
        current_centers[i].append(data[initialIndices2[i]])


    iterations =0
    while not converged(test_centers,current_centers,iterations):   # checking for convergence
        current_centers=None
        current_centers=test_centers

        cluster_dict, assigned_clusters = clustering(data, test_centers)  # assign all data points to clusters, and return the new new cluster dictionary
        test_centers = find_new_center(cluster_dict)     # relocate the centers

        iterations+=1
        print "Iteration: ", iterations

        if iterations > 100:
            break    #stop after 100 iterations

    # if the function has converged (the centers dont change any more) OR the number of iteration is 100:


    assigned_clusters=np.array(assigned_clusters)   #the assigned clusters will be the clusters from the last iteration
    final_cluster_centers = np.array(test_centers).tolist()
    flattened = [val for sublist in final_cluster_centers for val in sublist]   #remove double brackets
    final_cluster_centers=np.array(flattened)   #final centroids
    print final_cluster_centers.shape


    return final_cluster_centers, assigned_clusters



# new_centers,assigned_clusters =kmeans(data,initialIndices,k)
# print "New centers:"
# print new_centers
# # print assigned_clusters
