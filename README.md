# NewYork-Taxi-Ride-Segmentation


## Description

In this project, I have implemented clustering using k-means and GMM using Expectation Maximization algorithm from scratch on the New York Taxi Trip data. Before fitting the model, I have identified the segment of interest using features that won't be used in fitting the model. The goal of the experiment is to see whether the clustering can detect those pre-defined segments. 


## About the dataset

The original data set consists of New York City Taxi trip reports in the Year 2013. The dataset was released under the FOIL (The Freedom of Information Law) and made public by Chris Whong (https://chriswhong.com/open-data/foil_nyc_taxi/).


# Obtaining the Dataset

Small Dataset: 
gs://github-data-yang/taxi-data-sorted-small.csv.bz2

Large Dataset:
gs://github-data-yang/taxi-data-sorted-large.csv.bz2


# Python Scripts

This project consists of 3 .py files. 

1. clean_and_generate_sample_data: include data cleaning and feature engineering process to get the sampled data ready for used in the clustering model.  
    
    - Number of arguments needed: 2
    - Number of ouput produced: 1

2. kmeans_find_optimal_k: include a k-Means clustering algorithm that runs 19 k-means models using k from 2,3,4..20 and calculate within cluster sum of       squared errors and silhouette coefficients for each. 
    
    - Number of arguments needed: 2
    - Number of ouput produced: 1


3. kmeans_gmm: Implement a k-means clustering using k=5 and a Gaussian Mixture Models using clusters results from the k-means as the initial parameters.

    - Number of arguments needed: 6
    - Number of ouput produced: 5


# Instructions to run

Make sure that you have Download and configured Apache Spark on your machine. 

Each of the python scrips will take arguments when executing:

 1. clean_and_generate_sample_data:
 
     - sys.argv[1]: link/path to the original dataset
     - sys.argv[2]: folder path to store the sampled dataset (O1)
     
 2. kmeans_find_optimal_k
     
     - sys.argv[1]: link/path to the training set
     - sys.argv[2]: folder path to store the scores for the 19 k-means clustering (O2)
 
 
 3. kmeans_gmm
     
     -  sys.argv[1]: link/path to the training set
     -  sys.argv[2]: folder path to store k-means centroids (with k=5) (O3)
     -  sys.argv[3]: folder path to store k-means results summary (O4)
     -  sys.argv[4]: folder path to store final mu (centroids) from GMM (with k=5) (O5)
     -  sys.argv[5]: folder path to store final sigma from GMM (with k=5) (O6)
     -  sys.argv[6]: folder path to store GMM results summary (O7)
     

## Running locally on an IDE (e.g Pycharm) - Not recommended
Download the dataset to your local disk. Clone the script and paste it to your IDE, substitute all argument fields (sys.argv[?]) with corresponding path.

## Running on a Cloud Service (e.g Google Cloud or Amzon AWS)
Upload the script to your Cloud Drive. When submitting a job, supply the internal path of the script, pass in the corresponding interal URLs for the required arguments. 

