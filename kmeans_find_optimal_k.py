"""
Description: This file runs a set of k-means clustering using different number of k
             and generate the sum of within-cluster errros and sihouettes scores for each. 
             These results will be used to select the best k for the next phase. 
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

# ========================== User-Defined Functions  ===========================


# Assign label to a trip; each trip as a pair of labels (Dx, Px)
'''
D2, D1, D0: top 25%, middle 50% and bottom 25% of 'driver fare'/mile
P2, P1, P0: top 25%, middle 50% and bottom 25% of 'passenger fare'/mile
'''
def assignLabel(d, p):
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


# ============================= End of Functions  =============================

if __name__ == "__main__":

    # System Arguments: 
    # [1]: Training Data
    # [2]: File path to save the scores for different k

    if len(sys.argv) != 3:
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

    
    # --------------- Feature Scaling (K-Means) ---------------

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

    print("mean:", meanFeatures)
    print("\nstd", stdFeatures)
    scaledFeatures.cache()

    # ========================== End of Data Preprocessing  ==========================

    # ========================== K-Means From Scratch  ==========================

    '''Parameters Initialization'''

    scores = []

    for k in range(2, 21):

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

        print("K", k, " WSSSE:", sumDistances, " Silhouette: ", silhouette)
        scores.append([k, sumDistances, silhouette])

    # store centroids info in a single file on the cluster.
    dataToASingleFile = sc.parallelize(scores).coalesce(1)
    dataToASingleFile.saveAsTextFile(sys.argv[2])

    # ========================== End of K-Means  ==========================
