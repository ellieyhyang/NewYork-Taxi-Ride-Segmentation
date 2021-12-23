"""
Description: This script contains a pySpark script that cleans and preprocess the 
			 lines from the Taxi Data to be used in the k-means and GMM clustering
			 model. This file will randomly pick 200k lines as sample dataset. 
"""

import sys
import psutil
from pyspark import SparkContext
import numpy as np
import datetime
import calendar


# ========================== User-Defined Functions  ===========================

# Check if a value is of float type
def isfloat(value):
    try:
        float(value)
        return True
    except:
        return False

    
# https://www.netstate.com/states/geography/ny_geography.htm
# Longitude: 71° 47' 25" W to 79° 45' 54" W
# Latitude: 40° 29' 40" N to 45° 0' 42" N

# Filter out the incorrect rows (ex: wrongly formatted or have missing information)
def correctRows(p):
    # ensure each rows has 17 values
    if len(p) == 17:
        # ensuring each row has a medallion, hack_license and a pickup_datetime
        if p[0] != '' or p[1] != '' or p[2] != '':
            # for task 3, validate the following: 
            # [4]trip_time_in_secs: must be non-zero float-type value
            # [5]trip_distance: must be non-zero float-type value
            # [6][7][8][9] coordinates must be non-zero float-type value
            # [11]fare_amount: must be between 1 and 600, float-type value
            # [12]surcharge: must be between 1 and 600, float-type value
            # [14]tip_amount: must be non-negative float-type value
            # [15]tolls_amount: must be non-negative float-type value
            if isfloat(p[4]) and isfloat(p[5]) and isfloat(p[11]) and isfloat(p[14]):                                              
                if float(p[4]) != 0 and float(p[5]) !=0 and float(p[11]) >=1 and float(p[11]) <=600 and float(p[12]) >=0 and float(p[14]) >=0 and float(p[15]) >=0:
                    if -float(p[6]) >= 71 and -float(p[6]) <= 80 and -float(p[8]) >= 71 and -float(p[8]) <= 80:
                        if float(p[7]) >= 40 and float(p[7]) <= 45 and float(p[9]) >= 40 and float(p[9]) <= 45:
                            return p


def get_weekday(date):
    year, month, day = (int(x) for x in date.split('-'))    
    weekday = datetime.date(year, month, day)
    # return calendar.day_name[weekday.weekday()]
    return weekday.weekday()+1

# ============================= End of Functions  =============================

# System Arguments: 
# [1]: Taxi Data set
# [2]: File path to store the sample dataset

sc = SparkContext.getOrCreate()

# Read the file name from the program arguments
# line = sc.textFile("taxi-data-sorted-small.csv.bz2")
line = sc.textFile(sys.argv[1])

# Map each trip record to a list of characters
taxilines = line.map(lambda x: x.split(','))

# Cleanup the data
taxilinesCorrected = taxilines.filter(correctRows)

# Below variables are  needed for preprocessing from the original dataset:
# [2]: pickup_datetime
# [4]: trip_time_in_secs
# *[5]: trip_distance (in mile)
# [6]: pickup_longitude
# [7]: pickup_latitude
# [8]: dropoff_longitude
# [9]: dropoff_latitude
# [11]: fare_amount (in dollars)
# [12]: surcharge (in dollars)
# [14]: tip_amount (in dollars)
# [15]: tolls_amount (in dollars)
dataRDD = taxilinesCorrected\
              .map(lambda x: (x[2].split(' '), float(x[4])/60, float(x[5]),
                              float(x[6]), float(x[7]), float(x[8]), float(x[9]),
                              float(x[11]), float(x[12]), float(x[14]), float(x[15])))


# 1st Map -> RDD that containts (weekday, hour, distance(mile), pu_long, pu_lat, dp_long, dp_lat, 
#                                tripTime(min), fare, surcharge, tip, toll)
# 2nd Map -> RDD (weekday, hour, pu_long, pu_lat, dp_long, dp_lat, tripTime(min),
#                 fare, surchage, tip, toll, distance(mile))
# featureArray = [weekday, hour, pu_long, pu_lat, dp_long, dp_lat, tripTime(min)]
dataRDD = dataRDD\
              .map(lambda x: (get_weekday(x[0][0]), int(x[0][1][:2]), x[2], 
                              x[3], x[4], x[5], x[6], x[1], x[7], x[8], x[9], x[10]))\
              .map(lambda x: (x[0], x[1], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[2]))

# Take 200000 sample from the dataset as the training
sampleData_training = dataRDD.takeSample(False, 200000)

# Store to cluster as text file
trainDataToASingleFile = sc.parallelize(sampleData_training).coalesce(1)
# trainDataToASingleFile.saveAsTextFile("taxiTraining200k")
trainDataToASingleFile.saveAsTextFile(sys.argv[2])

sc.stop()
