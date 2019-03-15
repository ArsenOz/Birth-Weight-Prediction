# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 21:23:57 2019

@author: Arsen_Oz

Working Directory:
C:/Users/...../Predicting Birth Weights

Purpose:
    Comparison of different models types in order to find best model to predict
    birth weights.The steps that we will follow are the next:
        
    A) Testing Different Models
        1) sklearn OLS Regression
        2) sklearn KNN
        3) sklearn SVRegressor
        4) sklearn SGDRegressor
    
    B) Model Accucary Comparison
        1) sklearn OLS Regression
        2) sklearn KNN
        3) sklearn SVRegressor
        4) sklearn SGDRegressor
    
    C) Selecting Final Model
    
"""
# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean
from sklearn.preprocessing import StandardScaler # for scaling variables
from sklearn.model_selection import train_test_split # train/test split
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
from sklearn.svm import SVR # support vector machines for regression
from sklearn.linear_model import SGDRegressor # SGDRegressor
from sklearn.linear_model import LinearRegression # Linear Regression

# Importing Explored Dataset
bw = pd.read_excel('birthweight_featured.xlsx')

###############################################################################
# A) TESTING DIFFERENT MODELS
###############################################################################
"""
We will try 4 different models(OLS Regression, KNearestNeighbors, Support
Vector Regressor and Stochastic Greadient Descent Regressor) in order to find 
the best accuracy score for our dataset.
"""
#################
# 1) sklearn OLS Regression
#################

# Data and Target Split
bw_lr_data = bw.drop(['bwght'],
                     axis = 1)

bw_lr_target = pd.DataFrame(bw['bwght'])

# Test and Training Split
X_train, X_test, y_train, y_test = train_test_split(
            bw_lr_data,
            bw_lr_target,
            random_state = 508,
            test_size = 0.1)

# Preparing the Model
bw_lr= LinearRegression()

# Fitting Results
bw_lr.fit(X_train, y_train)

# Predicting Train and Test Dataset
lr_y_pred = bw_lr.predict(X_train)

lr_y_hat_pred = bw_lr.predict(X_test)

# Comparing Training and Test Scores
lr_y_score = bw_lr.score(X_train, y_train)

lr_y_hat_score = bw_lr.score(X_test, y_test)

print(lr_y_score)
print(lr_y_hat_score)


#################
# 2) sklearn KNN
#################

# Data and Target Split
bw_knn_data = bw.drop(['bwght'],
                      axis = 1)

bw_knn_target = pd.DataFrame(bw['bwght'])

# Test and Training Split
X_train, X_test, y_train, y_test = train_test_split(
            bw_knn_data,
            bw_knn_target,
            random_state = 508,
            test_size = 0.1)

# Creating lists for recording score for different n values
training_accuracy = []
test_accuracy = []

# Setting neighbors to be tested between 1 and 50
neighbors_settings = range(1, 51)

# For better results, we need to scale the variables
var_scaler = StandardScaler()

X_train = var_scaler.fit_transform(X_train)
X_test = var_scaler.fit_transform(X_test)
y_train = var_scaler.fit_transform(y_train)
y_test = var_scaler.fit_transform(y_test)

for n_neighbors in neighbors_settings:
    # Building the model
    bw_knn = KNeighborsRegressor(n_neighbors = n_neighbors)
    bw_knn.fit(X_train, y_train)
    
    # Recording the training set accuracy
    training_accuracy.append(bw_knn.score(X_train, y_train))
    
    # Recording the generalization accuracy
    test_accuracy.append(bw_knn.score(X_test, y_test))


# Plotting the visualization
fig, ax = plt.subplots(figsize=(12,9))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

# Best score when n = 9
test_accuracy.index(max(test_accuracy))

############
############

# Preparing the optimal KNN Model
bw_knn= KNeighborsRegressor(n_neighbors = 9)

# Fitting Results
bw_knn.fit(X_train, y_train)

# Predicting Train and Test Dataset
knn_y_pred = bw_knn.predict(X_train)

knn_y_hat_pred = bw_knn.predict(X_test)

# Comparing Training and Test Scores
knn_y_score = bw_knn.score(X_train, y_train)

knn_y_hat_score = bw_knn.score(X_test, y_test)

print(knn_y_score)
print(knn_y_hat_score)


#################
# 3) sklearn SVR (Support Vector Machines)
#################

bw_svr_data = bw.drop(['bwght'],
                      axis = 1)

bw_svr_target = pd.DataFrame(bw['bwght'])

# Test and Training Split
X_train, X_test, y_train, y_test = train_test_split(
            bw_svr_data,
            bw_svr_target,
            random_state = 508,
            test_size = 0.1)

# For better results, we need to scale the variables
var_scaler = StandardScaler()

X_train = var_scaler.fit_transform(X_train)
X_test = var_scaler.fit_transform(X_test)
y_train = var_scaler.fit_transform(y_train)
y_test = var_scaler.fit_transform(y_test)

# Preparing the Model
bw_svr= SVR(kernel = 'linear')

# Fitting Results
bw_svr.fit(X_train, y_train)

# Predicting Train and Test Dataset
svr_y_pred = bw_svr.predict(X_train)

svr_y_hat_pred = bw_svr.predict(X_test)

# Comparing Training and Test Scores
svr_y_score = bw_svr.score(X_train, y_train)

svr_y_hat_score = bw_svr.score(X_test, y_test)

print(svr_y_score)
print(svr_y_hat_score)


#################
# 4) sklearn SGDRegressor
#################

bw_sgdr_data = bw.drop(['bwght'],
                       axis = 1)

bw_sgdr_target = pd.DataFrame(bw['bwght'])

# Test and Training Split
X_train, X_test, y_train, y_test = train_test_split(
            bw_sgdr_data,
            bw_sgdr_target,
            random_state = 508,
            test_size = 0.1)

# For better results, we need to scale the variables
var_scaler = StandardScaler()

X_train = var_scaler.fit_transform(X_train)
X_test = var_scaler.fit_transform(X_test)
y_train = var_scaler.fit_transform(y_train)
y_test = var_scaler.fit_transform(y_test)

# Preparing the Model
bw_sgdr= SGDRegressor(loss = 'epsilon_insensitive',
                      max_iter = 1000)

# Fitting Results
bw_sgdr.fit(X_train, y_train)

# Predicting Train and Test Dataset
sgdr_y_pred = bw_sgdr.predict(X_train)

sgdr_y_hat_pred = bw_sgdr.predict(X_test)

# Comparing Training and Test Scores
sgdr_y_score = bw_sgdr.score(X_train, y_train)

sgdr_y_hat_score = bw_sgdr.score(X_test, y_test)

print(sgdr_y_score)
print(sgdr_y_hat_score)

"""
We can see that OLS Regression scores slightly better in both training and test
dataset compare to SVR and SGDRegressor. However, we are testing our models
with specific random state, therefore we need to make sure that OLS Regression
works better on average if we have any other random states.
"""
###############################################################################
# B) MODEL COMPARISONS
###############################################################################
"""
In order to make sure OLS is working better different random stated cases, we
created a loop that will run each model with the same different random states
for 1000 times and return the average accuracy for test and training datasets.
"""
#################
# 1) sklearn OLS Regression Loop
#################

# Data and Target Split
bw_lr_data = bw.drop(['bwght'],
                     axis = 1)

bw_lr_target = pd.DataFrame(bw['bwght'])

# Creating lists for 
lr_training_accuracy = []
lr_test_accuracy = []

# Trying 1000 random test and training dataset
for x in range(1,1001):
    
    # Test and Training Split
    X_train, X_test, y_train, y_test = train_test_split(
            bw_lr_data,
            bw_lr_target,
            test_size = 0.1,
            random_state = x)
    
    bw_lr= LinearRegression()
    
    bw_lr.fit(X_train, y_train)
    
    # Recording the training set accuracy
    lr_training_accuracy.append(bw_lr.score(X_train, y_train))
    
    # Recording the test set accuracy
    lr_test_accuracy.append(bw_lr.score(X_test, y_test))

lr_training_accuracy_mean = mean(lr_training_accuracy)

lr_test_accuracy_mean = mean(lr_test_accuracy)


#################
# 2) sklearn KNN Loop
#################

# Data and Target Split
bw_knn_data = bw.drop(['bwght'],
                      axis = 1)

bw_knn_target = pd.DataFrame(bw['bwght'])

# Creating lists for 
knn_training_accuracy = []
knn_test_accuracy = []

# Trying 1000 random test and training dataset
for x in range(1,1001):
    
    # Test and Training Split
    X_train, X_test, y_train, y_test = train_test_split(
            bw_knn_data,
            bw_knn_target,
            test_size = 0.1,
            random_state = x)
    
    # For better results, we need to scale the variables
    var_scaler = StandardScaler()

    X_train = var_scaler.fit_transform(X_train)
    X_test = var_scaler.fit_transform(X_test)
    y_train = var_scaler.fit_transform(y_train)
    y_test = var_scaler.fit_transform(y_test)
    
    # Creating lists for recording score for different n values
    training_accuracy = []
    test_accuracy = []
    
    # For loop to find best n_neighbors
    for n_neighbors in neighbors_settings:
        # Building the model
        bw_knn = KNeighborsRegressor(n_neighbors = n_neighbors)
        bw_knn.fit(X_train, y_train)
    
        # Recording the training set accuracy
        training_accuracy.append(bw_knn.score(X_train, y_train))
    
        # Recording the generalization accuracy
        test_accuracy.append(bw_knn.score(X_test, y_test))
        
    
    # Identifying ideal n value for the maximum score
    n = test_accuracy.index(max(test_accuracy)) + 1
    
    bw_knn= KNeighborsRegressor(n_neighbors = n)
    
    bw_knn.fit(X_train, y_train)
    
    # Recording the training set accuracy
    knn_training_accuracy.append(bw_knn.score(X_train, y_train))
    
    # Recording the test set accuracy
    knn_test_accuracy.append(bw_knn.score(X_test, y_test))

knn_training_accuracy_mean = mean(knn_training_accuracy)

knn_test_accuracy_mean = mean(knn_test_accuracy)


#################
# 3) sklearn SVR (Support Vector Machines) Loop
#################

# Data and Target Split
bw_svr_data = bw.drop(['bwght'],
                       axis = 1)

bw_svr_target = pd.DataFrame(bw['bwght'])

# Creating lists for 
svr_training_accuracy = []
svr_test_accuracy = []

# Trying 1000 random test and training dataset
for x in range(1,1001):
    
    # Test and Training Split
    X_train, X_test, y_train, y_test = train_test_split(
            bw_svr_data,
            bw_svr_target,
            test_size = 0.1,
            random_state = x)
    
    # For better results, we need to scale the variables
    var_scaler = StandardScaler()

    X_train = var_scaler.fit_transform(X_train)
    X_test = var_scaler.fit_transform(X_test)
    y_train = var_scaler.fit_transform(y_train)
    y_test = var_scaler.fit_transform(y_test)
    
    bw_svr= SVR(kernel = 'linear')
    
    bw_svr.fit(X_train, y_train)
    
    # Recording the training set accuracy
    svr_training_accuracy.append(bw_svr.score(X_train, y_train))
    
    # Recording the test set accuracy
    svr_test_accuracy.append(bw_svr.score(X_test, y_test))

svr_training_accuracy_mean = mean(svr_training_accuracy)

svr_test_accuracy_mean = mean(svr_test_accuracy)


#################
# 4) sklearn SGDRegressor Loop
#################

# Data and Target Split
bw_sgdr_data = bw.drop(['bwght'],
                       axis = 1)

bw_sgdr_target = pd.DataFrame(bw['bwght'])

# Creating lists for 
sgdr_training_accuracy = []
sgdr_test_accuracy = []

# Trying 1000 random test and training dataset
for x in range(1,1001):
    
    # Test and Training Split
    X_train, X_test, y_train, y_test = train_test_split(
            bw_sgdr_data,
            bw_sgdr_target,
            test_size = 0.1,
            random_state = x)
    
    # For better results, we need to scale the variables
    var_scaler = StandardScaler()

    X_train = var_scaler.fit_transform(X_train)
    X_test = var_scaler.fit_transform(X_test)
    y_train = var_scaler.fit_transform(y_train)
    y_test = var_scaler.fit_transform(y_test)
    
    bw_sgdr= SGDRegressor(loss = 'epsilon_insensitive',
                          max_iter = 1000)

    bw_sgdr.fit(X_train, y_train)
    
    # Recording the training set accuracy
    sgdr_training_accuracy.append(bw_sgdr.score(X_train, y_train))
    
    # Recording the test set accuracy
    sgdr_test_accuracy.append(bw_sgdr.score(X_test, y_test))

sgdr_training_accuracy_mean = mean(sgdr_training_accuracy)

sgdr_test_accuracy_mean = mean(sgdr_test_accuracy)


#################
## Comparing all 4 models average score
#################

# Creating dictionary with scores
data = {'Model': ['Linear',
                  'KNN',
                  'SVReg',
                  'SGDReg'],
        'Training':[lr_training_accuracy_mean,
                    knn_training_accuracy_mean,
                    svr_training_accuracy_mean,
                    sgdr_training_accuracy_mean],
        'Test': [lr_test_accuracy_mean,
                 knn_test_accuracy_mean,
                 svr_test_accuracy_mean,
                 sgdr_test_accuracy_mean]}


# Creating DataFrame for comparing scores with plot
model_accuracies = pd.DataFrame(data = data)

model_accuracies = pd.melt(model_accuracies,
                           id_vars="Model",
                           var_name="Dataset",
                           value_name="Accuracy")

# Factorplot for comparing model accuracies
sns.factorplot(x='Dataset',
               y='Accuracy',
               hue='Model',
               data=model_accuracies,
               kind='bar',
               palette='Set1')

plt.title('Model Accuracy Comparison')

plt.savefig('Model Accuracy Comparison.png')


###############################################################################
# B) SELECTING FINAL MODEL
###############################################################################

"""
If we look at the plot with the model accuracies, OLS is still working slightly
better with the training dataset. However, we can see that both SVR and SGDR 
passes OLS for the test dataset. Since we want greater scores for test dataset 
and also smaller gap between the training and test dataset accuracy, we will 
select either SVR or SGDR.

It is difficult to understand which one is performing better on test dataset 
from the plot. However, if we look at the numbers, we can see that SGDRegressor
is performing slightly better(almost identical values) compare to SVR model.

Therefore, we selected SGDRegressor model as our final model.
"""
#################
## FINAL MODEL (SGD Regressor)
#################

bw_sgdr_data = bw.drop(['bwght'],
                       axis = 1)

bw_sgdr_target = pd.DataFrame(bw['bwght'])

# Test and Training Split
X_train, X_test, y_train, y_test = train_test_split(
            bw_sgdr_data,
            bw_sgdr_target,
            random_state = 508,
            test_size = 0.1)

# For better results, we need to scale the variables
var_scaler = StandardScaler()

X_train = var_scaler.fit_transform(X_train)
X_test = var_scaler.fit_transform(X_test)
y_train = var_scaler.fit_transform(y_train)
y_test = var_scaler.fit_transform(y_test)

# Preparing the Model
bw_sgdr= SGDRegressor(loss = 'epsilon_insensitive',
                      max_iter = 1000)

# Fitting Results
bw_sgdr.fit(X_train, y_train)

# Predicting Train and Test Dataset
sgdr_y_pred = bw_sgdr.predict(X_train)

sgdr_y_hat_pred = bw_sgdr.predict(X_test)

# Comparing Training and Test Scores
sgdr_y_score = bw_sgdr.score(X_train, y_train)

sgdr_y_hat_score = bw_sgdr.score(X_test, y_test)

print(sgdr_y_score)
print(sgdr_y_hat_score)







