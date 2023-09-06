# Random Projections; KDE
Random Projections & KDE
Tuesday, March 28, 2023
11:22 AM

# Main Goals
Identify outliers based on the cortical thickness data (our design matrix X) in the ABIDE dataset
Perform supervised machine learning method (e.g. LASSO) with and without the outliers (to hopefully show the ability to obtain a better model fit when outliers are removed).

# Previous steps:
 
We divided the data into the previously-defined 32 parcels in the brain referring to different regions of the brain.
 
Created a dataframe of 1050 rows (this is the number of people), and then within each parcel, I created a set of random projections. 
With 50 random projections per parcel
Next steps:
 
Make sure to save the random number seed to start so that results can be repeated.  
Decide how many columns to select per parcel. Call this number M, and we propose to start with M=5.
We would like to perform KDE(Kernel Density Estimation) on the parcellated data.
Pick M columns from each parcel (in the reduced matrix) and perform KDE on each of these 5x32 selected lists.
 
 Pick Mx32 columns at random from the whole dataset and perform KDE on this dataset.
 
Compare the above two methods.
Compare again with another choice for M, such as M = 5, or M=20 or M=50.
 
All of the above methods must be performed inside a cross-validation approach.  
 
Before starting,  the dataset must be divided randomly into training and test set, but retaining the case/control balance such that the  Test set contains 1/3 of cases and 1/3 of controls ( this is a stratified test set)
 
In order to do this, 
Choose a random number seed.
Divide cases randomly into training (2/3) and test(1/3)
Divide controls randomly into training(2/3) and test(1/3)
Save the IDs assigned to training and test. 
Choose another random number seed and save it.  Working only with the training data:
Randomly divide the data for cross-validation (say five-fold)
For each fold of the CV analysis (i.e. each 80%)
Randomly select Mx32 columns as described above for each of the two methods.
Fit models
Use the cross-validation to pick the best tuning parameters for each column selection method, using the left-out fold.
When the model and tuning parameters are fixed, evaluate performance on the test set.
 
Finally, predict on the test data
 
We perform KDE first and then do the same with other methods like KNN, Lasso and Ridge.
 
Save the predictions on made on each of the methods to create a summary of the predictions (set the same seed).

Updated process - 6th to 13th April 2023

Choose a random number seed.
Divide cases randomly into training (2/3) and test(1/3) - this is done
Divide controls randomly into training(2/3) and test(1/3) - this is done
Save the IDs assigned to training and test. 

Whole training dataset contains: 704 rows and 149960 columns (704X149960)
Second step has two process: 
We divided the 149960 columns to 32 parcels
Within the 32 parcels we selected 50 columns from each parcel by random projections
So we have 704 X 32 X 50 i.e. 704 X 1600 variables 
To compare, we created random projections of 704 X 1600 variables from the whole dataset of 704 X 149960 variables NOT USING PARCELS.
Crossvalidation – We create a 5-fold crossvalidation by leaving out 20% of observations(people) and within this crossvalidation:
Strategy-1: Randomly select 5 variables from 50 variables in each parcel (from 2A)
This will give us 160 variables for each fold. (Checking: this algorithm will pick probably different variables within each fold, right?)
Perform KDE on these 160 variables for each fold.
Strategy-2: Randomly select 160 variables from 2B within each fold
Perform KDE on these 160 variables
Outlier detection + filter: 
Take what comes out of KDE, figure out which points are T standard deviations from the mean.
You do this for a whole series of values for T.
Remove the outliers
Predict y using the trimmed dataset using SMLE and LASSO


# Discussion April 21 2023

For (j in 1:5) {
    in the 80% of the samples after removing the 20% of the samples in fold j
We fit KDE twice, after sampling columns with either strategy1 or strategy 2. We need to find a nice software package for multivariate KDE.
try the mvkde function below using ‘data’ = the 80% leaving out fold j (rows) and the columns selected by strategy 1 or strategy 2 AND ‘x’ = the 20% of the left out fold and the same number of columns as above.
Define a tuning parameter which is going to be something like lambda. For example it might be (lambda = 1.5), or (lambda = 3).  Define a series of these lambdas. (1.5, 3, 5, 10). Do this using crossvalidation.
Then find the outliers in the fold that was left out using the lambdas defined on the KDE results.
This should lead to a list of points to be defined as outliers for each value of lambda.
}
After completing the loop, each sample should be associated with a series of “calls”, where each call is a decision as to whether this point is an outlier for a particular lambda. 
=====================================


Wu, Ximing (2019), "Robust Likelihood Cross Validation for Kernel Density Estimation," Journal of Business and Economic Statistics, 37(4): 761-770.

##radially symmetric kernel (Gussian kernel)
RadSym <- function(u)
 exp(-rowSums(u^2)/2) / (2*pi)^(ncol(u)/2)
##multivariate extension of Scott's bandwidth rule
Scott <- function(data)
 t(chol(cov(data))) * nrow(data) ^ (-1/(ncol(data)+4))
##compute KDE at x given data
mvkde <- function(x, data, bandwidth=Scott, kernel=RadSym) {
#bandwidth may be a function or matrix
 if(is.function(bandwidth))
   bandwidth <- bandwidth(data)
 u <- t(solve(bandwidth, t(data) - x))
 mean(kernel(u))
}


Sample without replacement

Do svd and a scatter plot of the first two 'loadings' which have dimension of 160 and colour by parcel

Do the same for 160 variables chosen at random still color by parcel but it won’t show the same columns as above in each parcel.

Visualize the outliers in the real data. svd of the raw data and color the outliers and look at the ages of the outliers relative to the distribution of other people in the dataset.

Density plot instead of a boxplot. Do a rug command on the plot to show the data points.

# 29/06/2023

1. Colored heatmaps to find outliers:
 Based the projections we graphed for each of 32 parcels
Take all the people in the rows
Random projections in the columns
Sort the people by KDE so the outliers are either at the top or the bottom
Plot the actual data by color
Do this just on parcel 10 first and then maybe compare it with another parcel?

Notes:
Do image function instead of heatmap function in R. This way you can sort the people by KDE values and color the data.
Choose the same columns that was chosen to do KDE to plot the heatmap.

2. Randomness associated with selection of the features. Use TSNE instead of random projections to select features.

3. Simulations: https://www.sciencedirect.com/science/article/pii/S1361841512000564?via%3Dihub

Based on Amadou’s paper and datagen.R code to create outliers
We want to keep the structure of the data but work with smaller scaled data.
< 160 variables
We want to retain the idea of parcels -> Maybe do 5 parcels instead of 32 as in the original
Keep the idea of a bad data point; What if the data is bad only in a certain parcel and not the others?
Put the outliers in the data that are not easy to find. How to simulate more subtle problems
4. Brainstorm about what the KDE is actually finding.


# Next steps:

Make a pipeline of the analysis you’ve done so far. The random projections, parcellations and no parcellations analysis.
Make a function of all the analysis:
This function should contain all the parameters like a series of lambda, a series of changing seed and changing the number of outliers.
Number of random projections desired would also be a good parameter to pass into a function. 
This would give us a set of results for each/ all of the changes made. Save the results for all of them for both Lasso and SMLE techniques.
 The end result could be a table of values for each of the parameters changed.
We can perform some kind of graph analysis with this ( like a boxplot for example).

