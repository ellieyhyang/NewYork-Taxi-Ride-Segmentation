"""
Description: This file contains a pySpark program that builds a k-means clustering on
             the training data using the best k identified in the last phase. It will 
             then build a GMM (Gaussian Misture Models) using the EM algorithm that uses
             restuls from the k-means as initial parameters. 

             The script will produce and store results for both clustering models for
             furthur analysis. 
"""

from __future__ import print_function
import sys
import psutil
from pyspark import SparkContext
import numpy as np
from operator import add
from pyspark.ml.linalg import Vectors
from pyspark.sql import SQLContext
from pyspark.sql.types import FloatType
from pyspark.ml.evaluation import ClusteringEvaluator
from scipy.stats import multivariate_normal as multi_nrml


# ========================== User-Defined Functions  ===========================

# Assign label to a trip; each trip as a pair of labels (Dx, Px)
'''
D2, D1, D0: top 25%, middle 50% and bottom 25% of 'driver fare'/mile
P2, P1, P0: top 25%, middle 50% and bottom 25% of 'passenger fare'/mile
'''
def assignLabel(d, p, quantile_d, quantile_p):
    if d >= quantile_d[0]:
        d_label = 'D2'
    elif d >= quantile_d[2]:
        d_label = 'D1'
    else:
        d_label = 'D0'
    if p >= quantile_p[0]:
        p_label = 'P2'
    elif p >= quantile_p[2]:
        p_label = 'P1'
    else:
        p_label = 'P0'
    return (d_label, p_label)


# Return a code in format (x, x, x) for each label
def labelCode(label):
    digit = int(label[-1])
    if digit not in [0, 1, 2]:
        code =  None
    else:
        if digit == 2:
            code = [1, 0, 0]
        elif digit == 1:
            code = [0, 1, 0]
        else:
            code = [0, 0, 1]
    return np.array(code)


# Returns the weighted covariance matrix by taken x(featureMatrix), mu and weight
def covMat(x, mu, w):
    covars = []
    for i in range(len(mu)):
        diff = x[i] - mu[i]
        diffT = diff.reshape(-1,1)
        covar = diff * diffT * w[i]
        covars.append(covar)  
    return np.array(covars)


# Calculate the probability that vector x belongs to k distributions
# Returns a vector of size n
def predict_prob(x, dist):
    probs = []
    for gaussian in dist:
        prob = gaussian.pdf(x)
        probs.append(prob)
    if probs == [0]*k:
        probs = np.repeat(1/k, k)
    return np.array(probs)


# ============================= End of Functions  =============================


if __name__ == "__main__":

    # System Arguments: 
    # arg[1]: training data
    # arg[2]: path to store kmeans centroids
    # arg[3]: path to store kmeans cluster summary
    # arg[4]: path to store gmm compoment MUs
    # arg[5]: path to store gmm compoment Sigmas
    # arg[6]: path to store gmm cluster summary

    if len(sys.argv) != 7:
        print("Usage: wordcount <file> <output> ", file=sys.stderr)
        exit(-1)

    sc = SparkContext.getOrCreate()
    sqlContext = SQLContext(sc)

    # ============================= Data Preprocessing  =============================

    # Load training data
    # line = sc.textFile('dataSets/taxiTraining80000v2')
    line = sc.textFile(sys.argv[1])

    # Map each trip record to a list of floats
    '''
    [0] weekday 
    [1] time(hour) 
    [2] pu_long 
    [3] pu_lat 
    [4] dp_long
    [5] dp_lat
    [6] tripTime(min)
    [7] fare
    [8] surcharge
    [9] tip
    [10] toll
    [11] distance(mile)
    '''
    taxilines = line\
                .map(lambda x: x[1:-1])\
                .map(lambda x: x.split(', '))\
                .map(lambda x: [float(i) for i in x])


    # Add index to training data
    # Create an RDD that has (tripFareVector, featureVector) pairs
    trainData = taxilines.map(lambda x: (np.array(x[7:]), np.array(x[:7])))

    # Add index for each trip entry -> (idx, tripFareVector, featureVector)
    indexedTrainData = trainData\
                    .zipWithIndex()\
                    .map(lambda x: (x[1], x[0][0], x[0][1]))

    # Create an RDD that has (idx, featureMatrix) pairs -- used in GMM model iteration
    idxFeatureMatrix = indexedTrainData.map(lambda x: (x[0], x[2]))
    idxFeatureMatrix.cache()

    # --------------- Labeling ---------------

    # -> each trip has a pair of label (Dx, Px)
    '''
    D2, D1, D0: top 25%, middle 50% and bottom 25% of 'driver fare'/mile
    P2, P1, P0: top 25%, middle 50% and bottom 25% of 'passenger fare'/mile
    '''

    # calculate (fare + tip)/mile and (fare + surcharge + toll)/ mile
    # And calculate the 20%, 50% and 75% percentile for each figure
    '''
    [0] fare
    [1] surcharge
    [2] tip
    [3] toll
    [4] distance(mile)
    '''
    # 1st map --> (idx, (fare, surcharge, tip, toll, distance))
    # 2nd map --> (idx, fare+surcharge+tip, fare+surcharge+toll, distance)
    # 3rd map -> (idx, (fare + tip + surcharge)/mile, (fare + surcharge + toll)/mile)
    idxFarePerMile = indexedTrainData\
            .map(lambda x: (x[0], x[1]))\
            .map(lambda x: (x[0], np.sum(x[1][[0,1,2]]), np.sum(x[1][[0,1,3]]), x[1][-1]))\
            .map(lambda x: (x[0], x[1]/x[3], x[2]/x[3]))

    # Analyze fare/miles and calculate the 20%, 50% and 75% percentile for each figure
    farePerMile = idxFarePerMile.map(lambda x: (float(x[1]), float(x[2])))
    farePerMileDF = sqlContext.createDataFrame(farePerMile, ["(fare+tip+schrg)/mile", "(fare+schrg+toll)/mile"])
    # Calculate quantiles
    quantile_d = []
    quantile_p = []

    for i in ["(fare+tip+schrg)/mile", "(fare+schrg+toll)/mile"]:
        print(i)
        for q in (0.75, 0.5, 0.25):
            qtlVal = farePerMileDF.approxQuantile(i, [q], 0)[0]
            if i == "(fare+tip+schrg)/mile":
                quantile_d.append(round(qtlVal,2))
            else:
                quantile_p.append(round(qtlVal,2))
            print ("  {:.0f}% quantile: {:.4f}".format(q*100, qtlVal))
        print()

    # assign label to each trip by calling function 'assignLabel'
    idxAndLabel = idxFarePerMile\
                .map(lambda x: (x[0], assignLabel(x[1], x[2], quantile_d, quantile_p)))


    # collect as Map and broad ast this to all worker nodes. 
    idxAndLabelAsMap = idxAndLabel.collectAsMap()
    sc.broadcast(idxAndLabelAsMap)


    # --------------- Feature Scaling (K-Means only) ---------------

    # Store the number of trips in the training set
    numTrips = indexedTrainData.count()

    # Create an RDD that only has the features Vectors
    featureMatrix = indexedTrainData.map(lambda x: (x[2]))

    # compute mean for all features
    meanFeatures = featureMatrix.reduce(add)/numTrips

    # compute variance for all features
    varFeatures = featureMatrix\
                    .map(lambda x: (x[0]-meanFeatures)**2)\
                    .reduce(add)/numTrips

    # compute standard deviation for all features
    stdFeatures = np.sqrt(varFeatures)

    # scale features using standardization method (z-score)
    # gives an RDD that has(idx, scaledFeatureMatrix) pairs
    scaledFeatures = indexedTrainData\
                    .map(lambda x: (x[0], (x[2]-meanFeatures)/stdFeatures))

    # RDD used in K-means modle iteration
    scaledFeatures.cache()

    # ========================== End of Data Preprocessing  ==========================

    # ======================== K-Means Clustering from Scratch =======================

    '''Parameters Initialization'''
    
    # k=5 selected using the Elbow Method
    k = 5

    # set the initial clusters to an 'out-of-range' cluster ID
    prevC = np.repeat(k, numTrips)

    # initialize centroids by picking k random points, store as array
    ctrd = np.array(scaledFeatures.map(lambda x: x[1]).takeSample(False, k))

    # Number of re-assignment -> set to numTrips
    numReass = numTrips
    num_iterations = 0


    ''' Main Iterative Part of Kmeans Clustering'''

    # Iterate until there is no more cluster re-assignment
    while numReass != 0:
    
        # increment numIterations
        num_iterations += 1
     
        # assign cluster to each datapoint based on euclidean distance
        # 1st map -> (idx, featureArray, distMatrix)
        # 2nd map -> (idx, featureArray, ClusterID)
        dataCluster = scaledFeatures\
                  .map(lambda x: (x[0], x[1], np.sqrt(((x[1] - ctrd)**2).sum(axis=1))))\
                  .map(lambda x: (x[0], x[1], np.argmin(x[2])))

        # update centroids using data from the new clusters
        # 1st map -> (ClusterID, (featureArray, 1))
        # then sum by Key gives (ClusterID, (sumFeatureArray, count))
        # 2nd map -> (ClusterID, meanFeatureArray)
        # Finally, sort by clusterID such that the index represents cluster Id
        #          and save as a numpy array
        ctrd = dataCluster\
                  .map(lambda x: (x[2], (x[1], 1)))\
                  .reduceByKey(lambda x, y: (x[0]+y[0] ,x[1]+y[1]))\
                  .map(lambda x: (x[0], x[1][0]/x[1][1]))\
                  .sortByKey().map(lambda x: x[1]).collect()
        ctrd = np.array(ctrd)

        # compute the number of re-assignments
        currC = np.array(dataCluster.map(lambda x: x[2]).collect())
        numReass = np.sum(prevC!=currC)
        # update clusters
        prevC = currC
        
    # Calculate sum of distances 
    # First calculate distance from each data to its assigned cluster
    # Reduce by summation to get the intertia (sum of distances)
    sumDistances = dataCluster\
                    .map(lambda x: np.linalg.norm(x[1]-ctrd[x[2]]))\
                    .reduce(add)

    ''' Evaluation '''

    # Evaluate clustering by computing Silhouette score
    featureMatrixAndPred = dataCluster\
                        .map(lambda x: (Vectors.dense(x[1]), float(x[2])))
    featureMatrixAndPredDF = sqlContext.createDataFrame(featureMatrixAndPred, ['features', 'prediction'])

    # initialize evaluator
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(featureMatrixAndPredDF)

    print("Number of Iters:", num_iterations)
    print("\nWithin Set Sum of Squared Errors:", sumDistances)
    print("Silhouette with squared euclidean distance:", silhouette)


    # ========================== End of K-Means Clustering ==========================

    # ========================== KMeans Results Collection ==========================

    # collcet(idx, cluster) pairs RDD as map (a dict in python)
    idxAndClusterAsMap = dataCluster.map(lambda x: (x[0], x[2])).collectAsMap()

    # broad cast this to all worker nodes. 
    sc.broadcast(idxAndClusterAsMap)

    # Create an RDD that has (clusterID, Label, tripFareInfo, featureVector) sets
    clusterLabelTripFareFeatures = indexedTrainData\
                .map(lambda x: (idxAndClusterAsMap.get(x[0]), idxAndLabelAsMap.get(x[0]), x[1], x[2]))

    # Calculate cluster centroids (unscaled)
    # Create an RDD that has (cluster, featureVector, 1) set
    # Then calculate the features mean for each cluster
    ctrdUnscaled = clusterLabelTripFareFeatures\
                        .map(lambda x: (x[0], (x[3], 1)))\
                        .reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1]))\
                        .map(lambda x: (x[0], x[1][0]/x[1][1]))\
                        .sortByKey().map(lambda x: x[1]).collect()

    # Get a summary of cluster that contains average info for the label and tripFare per cluster
    # First, create an RDD that has (clusterID, P_code, tripFareInfo, 1) sets
    # Then, reduce by cluster to get sum
    # Finally, divid P_code, P_code and tripFareInfo by the count to get an average summary for each cluster
    clusterSummary = clusterLabelTripFareFeatures\
                        .map(lambda x: (x[0], ((labelCode(x[1][0]), labelCode(x[1][1]), x[2], 1))))\
                        .reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1], x[2]+y[2], x[3]+y[3]))\
                        .map(lambda x: (x[0], x[1][0], x[1][1], x[1][2], x[1][3]))\
                        .map(lambda x: (x[0], x[4], x[1]/x[4], x[2]/x[4], x[3]/x[4]))\
                        .map(lambda x: (x[0], x[1:]))\
                        .sortByKey()

    # store centroids and cluster summary separately in a single file on the cluster.
    ctrdToASingleFile = sc.parallelize(ctrdUnscaled).coalesce(1)
    ctrdToASingleFile.saveAsTextFile(sys.argv[2])

    # summryToASingleFile = sc.parallelize(clusterSummary).coalesce(1)
    clusterSmryToASingleFile = clusterSummary.coalesce(1)
    clusterSmryToASingleFile.saveAsTextFile(sys.argv[3])
    # dataToASingleFile.saveAsTextFile('kmean_clusterSummary')

    # ====================== End of KMeans Results Collection ======================


    # ==================== GMM Clustering Using EM From Scratch ====================

    '''Using resutls from kmeans as the initial Gaussian Components'''

    # Initital
    mu = np.array(ctrdUnscaled)

    # Initial phi and sigma
    # 1st map -> create an RDD that has (clusterID, unscaledFeatureMatrix, 1) set
    # 2nd map, calculate (Xi-Xmu) for each feature matrix
    # 3rd map, calculate (Xi-Xmu) * (Xi-Xmu).T for ach feature matrix
    # Finally, sum up by cluster and divide by the cluster size to get the covariance matrix for each
    sigmaAndphi = clusterLabelTripFareFeatures\
             .map(lambda x: (x[0], x[3], 1))\
             .map(lambda x: (x[0], x[1]-ctrdUnscaled[x[0]], x[2]))\
             .map(lambda x: (x[0], (x[1].reshape(-1,1) * x[1], x[2])))\
             .reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1]))\
             .map(lambda x: (x[0], (x[1][0]/x[1][1], x[1][1]/numTrips)))\
             .sortByKey().map(lambda x: x[1])
    sigma = sigmaAndphi.map(lambda x: x[0]).collect()
    phi = np.array(sigmaAndphi.map(lambda x: x[1]).collect())

    # Initial weights
    # weights = np.full(shape=(numTrips, k), fill_value=1/k)
    weights = np.full(shape=(numTrips, k), fill_value=phi)

    ''' Main Iterative Part of Kmeans Clustering'''

    phi_chg = float('inf')

    while phi_chg > 0.0001:

        ''' *** E-Step *** '''
        # E-Step: update weights and phi holding mu and sigma constant

        # list to store k gaussian distributions
        gaussian_dist = []

        for i in range(k):
            dist = multi_nrml(mean=mu[i], cov=sigma[i], allow_singular=True)
            gaussian_dist.append(dist)
            # likelihood[:,i] = gaussian_dist.pdf(x)

        ### Update weights
        # First compute (Prob_ij * Phi_j) for each trip
        # Then, compute W_ij by dividing the sum of (Prob_ij * Phi_j) for each component j
        # Final RDD has (idx, weightMatrix) pair
        weights = idxFeatureMatrix\
                      .map(lambda x: (x[0], (predict_prob(x[1], gaussian_dist)*phi)))\
                      .map(lambda x: (x[0], x[1]/np.sum(x[1])))

        ### Update phi
        new_phi = weights.map(lambda x: x[1]).reduce(add)/numTrips

        # record the change in phi
        phi_chg = np.sum(abs(new_phi - phi))
        print("Delta Phi:", phi_chg)

        # update phi
        phi = new_phi

        ''' *** M Step *** '''
        # M-Step: update mu and sigma holding phi and weights constant

        # Calculate weight for each component
        componentWeight = weights.map(lambda x: x[1]).reduce(add)

        ### Update mu
        # Calculate new mu for each components
        # First, mulitply each featureMatrix by its respective weights in each component k
        # Then reduce by summation to get the features total for each component k 
        mu = idxFeatureMatrix\
                        .join(weights)\
                        .map(lambda x: np.outer(x[1][1],x[1][0]))\
                        .reduce(add)

        mu = [mu[j]/componentWeight[j] for j in range(k)]

        ### Update sigma
        # Use function 'covMat' to calcualte weighted covariance matrix for each trip
        # Then add them up to get the updated covariance matrix for each component 
        sigma = idxFeatureMatrix\
                    .join(weights)\
                    .map(lambda x: covMat(x[1][0], mu, x[1][1]))\
                    .reduce(add)

        sigma = [sigma[j]/componentWeight[j] for j in range(k)]


    ''' Model Evaluation '''

    # Evaluate clustering by computing Silhouette score

    # add cluster id to each idx based on weights and collect as map
    idxAndPredGMMAsMap = weights.map(lambda x: (x[0], np.argmax(x[1]))).collectAsMap()
    print("phi:", phi)

    # ====================== End of GMM Clustering Using EM  =======================

    # =========================== GMM Results Collection ===========================

    # store centroids (MUs) info in a single file on the cluster.
    compMUToASingleFile = sc.parallelize(mu).coalesce(1)
    compMUToASingleFile.saveAsTextFile(sys.argv[4])

    # store sigma (covariances) info in a single file on the cluster.
    compSigmaToASingleFile = sc.parallelize(sigma).coalesce(1)
    compSigmaToASingleFile.saveAsTextFile(sys.argv[5])

    # first creat an RDD that has (GaussianID, Label, tripFareInfo) sets
    # then, calculate size, label distribution and mean of fares for each components
    componentSummary = indexedTrainData\
                        .map(lambda x: (idxAndPredGMMAsMap.get(x[0]), idxAndLabelAsMap.get(x[0]), x[1]))\
                        .map(lambda x: (x[0], ((labelCode(x[1][0]), labelCode(x[1][1]), x[2], 1))))\
                        .reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1], x[2]+y[2], x[3]+y[3]))\
                        .map(lambda x: (x[0], x[1][0], x[1][1], x[1][2], x[1][3]))\
                        .map(lambda x: (x[0], x[4], x[1]/x[4], x[2]/x[4], x[3]/x[4]))\
                        .map(lambda x: (x[0], x[1:]))\
                        .sortByKey()

    # store component info in a single file on the cluster.
    compSmryToASingleFile = componentSummary.coalesce(1)
    compSmryToASingleFile.saveAsTextFile(sys.argv[6])

    # ======================= End of GMM Results Collection ========================

    sc.stop()
