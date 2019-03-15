# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 22:59:07 2019

@author: Arsen_Oz

Working Directory:
C:/Users/...../Predicting Birth Weights

Purpose:
    Exploring Birth Weight dataset for further analysis. The steps that we will
    follow are:
        
    A) Exploratory Data Analysis
        1) Importing and First Look at Birth Weight Dataset
        2) Missing Value Detection, Flagging and Imputation
        3) Outlier Detection and Flagging
        4) Correlation Analysis
    
    B) Model Trials and Feature Engineering
        1) OLS Trials and Feature Engineering
        2) Exporting Final Datasets

"""
# Importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import iqr # IQR for Outlier Detection
import statsmodels.formula.api as smf # for standard OLS 
from sklearn.model_selection import train_test_split # train/test split
from sklearn.linear_model import LinearRegression # Linear Regression

###############################################################################
# A) EXPLORATORY DATA ANALYSIS
###############################################################################

################################
# 1) Importing and First Look at Birth Weight Dataset
################################

# Importing dataset
bw = pd.read_excel('birthweight.xlsx')

# Setting printing options
pd.set_option('display.max_columns', 25)

# First glimpse at dataset
bw.head()

bw.describe()

bw.info()

"""
Based on our research and dataset dictionary, we identified that One Minute
Test(omaps) and Five Minute Test(fmaps) are metrics that are collected after
the baby was born, which cannot be used for predicting the birth weight.
Therefore, we removed those metrics from our initial dataset and continued with
our analysis without those metrics.
"""

bw = bw.drop(['omaps',
              'fmaps'],
              axis = 1)

################################
# 2) Missing Value Detection, Flagging and Imputation
################################

#################
# Flagging Missing Variables
#################

for col in bw:
    if bw[col].isnull().any():
        bw['m_'+col] = bw[col].isnull().astype(int)


#################
# Exploring Columns with Missing Values
#################

# Dropping NA values
bw_dropped = bw.dropna()


# Mother Education
meduc_mean = bw_dropped['meduc'].mean()
meduc_median = bw_dropped['meduc'].median()

sns.distplot(bw_dropped['meduc'],
             color='blue')

# Mean Line
plt.axvline(meduc_mean,
            color='r',
            linestyle='--')

# Median Line
plt.axvline(meduc_median,
            color='g',
            linestyle='-')

plt.legend({'Mean':meduc_mean,
            'Median':meduc_median})

plt.savefig('Mother Education Distribution.png')

plt.show()

############
# Father Education
feduc_mean = bw_dropped['feduc'].mean()
feduc_median = bw_dropped['feduc'].median()

sns.distplot(bw_dropped['feduc'],
             color = 'blue')

# Mean Line
plt.axvline(feduc_mean,
            color='r',
            linestyle='--')

# Median Line
plt.axvline(feduc_median,
            color='g',
            linestyle='-')

plt.legend({'Mean':feduc_mean,
            'Median':feduc_median})

plt.savefig('Father Education Distribution.png')

plt.show()

############
# Number of Prenatal Visits
npvis_mean = bw_dropped['npvis'].mean()
npvis_median = bw_dropped['npvis'].median()

sns.distplot(bw_dropped['npvis'],
             color = 'blue')   
   
# Mean Line      
plt.axvline(npvis_mean,
            color='r',
            linestyle='--')

# Median Line
plt.axvline(npvis_median,
            color='g',
            linestyle='-')

plt.legend({'Mean':npvis_mean,
            'Median':npvis_median})

plt.savefig('Number of Prenatal Visits.png')

plt.show()

"""
Our observations showed that for all the missing values columns, the mean and
the median values are close to each other(almost same). Since all three columns
have discrete values, we decided to use median for missing value imputation.
"""

#################
# Missing Value Imputation
#################

fill = 0

# Filling all missing values with median

fill = bw['meduc'].median()

bw['meduc'] = bw['meduc'].fillna(fill)


fill = bw['npvis'].median()

bw['npvis'] = bw['npvis'].fillna(fill)


fill = bw['feduc'].median()

bw['feduc'] = bw['feduc'].fillna(fill)


################################
# 3) Outlier Detection and Flagging
################################
"""
For outlier detection, we will use histograms, IQR metric and research
foundings to identify high and low limits for the variables in this case.
Firstly, we start with plotting histograms.
"""

#################
# Distribution Plots (Histograms)
#################

# Mother and Father Information

f, ax = plt.subplots(figsize=(10, 8))
plt.subplot(2, 2, 1)

sns.distplot(bw['mage'],
             bins = 10,
             color = 'r',
             kde = False)

plt.xlabel('Mother Age')

############
plt.subplot(2, 2, 2)
sns.distplot(bw['meduc'],
             bins = 7,
             color = 'r',
             kde = False)

plt.xlabel('Mother Education')

############
plt.subplot(2, 2, 3)
sns.distplot(bw['fage'],
             bins = 10,
             color = 'b',
             kde = False)

plt.xlabel('Father Age')

############
plt.subplot(2, 2, 4)

sns.distplot(bw['feduc'],
             bins = 7,
             color = 'b',
             kde = False)

plt.xlabel('Father Education')

plt.tight_layout()
plt.savefig('Birth Weight Data Histograms 1 of 4.png')

plt.show()

########################
########################

# Prenatal Visits and Habits

f, ax = plt.subplots(figsize=(10, 8))
plt.subplot(2, 2, 1)
sns.distplot(bw['monpre'],
             bins = 5,
             color = 'g',
             kde = False)

plt.xlabel('Months Pregnant')

############
plt.subplot(2, 2, 2)
sns.distplot(bw['npvis'],
             bins = 10,
             color = 'g',
             kde = False)

plt.xlabel('Number of Prenatal Visits')

############
plt.subplot(2, 2, 3)

sns.distplot(bw['cigs'],
             bins = 7,
             color = 'b',
             kde = False)

plt.xlabel('Cigarettes')

############
plt.subplot(2, 2, 4)

sns.distplot(bw['drink'],
             bins = 5,
             color = 'b',
             kde = False)

plt.xlabel('Drinks')

plt.tight_layout()
plt.savefig('Birth Weight Data Histograms 2 of 4.png')

plt.show()

########################
########################

# Father and Mother Race

f, ax = plt.subplots(figsize=(10, 8))
plt.subplot(2, 3, 1)
sns.distplot(bw['mwhte'],
             bins = 5,
             color = 'r',
             kde = False)

plt.xlabel('White Mother')

############
plt.subplot(2, 3, 2)
sns.distplot(bw['mblck'],
             bins = 5,
             color = 'r',
             kde = False)

plt.xlabel('Black Mother')

############
plt.subplot(2, 3, 3)
sns.distplot(bw['moth'],
             bins = 5,
             color = 'r',
             kde = False)

plt.xlabel('Other Mother')

############
plt.subplot(2, 3, 4)

sns.distplot(bw['fwhte'],
             bins = 5,
             color = 'b',
             kde = False)

plt.xlabel('White Father')

############
plt.subplot(2, 3, 5)

sns.distplot(bw['fblck'],
             bins = 5,
             color = 'b',
             kde = False)

plt.xlabel('Black Father')

############
plt.subplot(2, 3, 6)

sns.distplot(bw['foth'],
             bins = 5,
             color = 'b',
             kde = False)

plt.xlabel('Other Father')


plt.tight_layout()
plt.savefig('Birth Weight Data Histograms 3 of 4.png')

plt.show()

########################
########################

# Baby Gender and Birth Weight

f, ax = plt.subplots(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.distplot(bw['male'],
             bins = 5,
             color = 'g',
             kde = False)

plt.xlabel('Baby is Male')

############
plt.subplot(1, 2, 2)
sns.distplot(bw['bwght'],
             bins = 20,
             color = 'g',
             kde = False)

plt.xlabel('Birth Weight')

plt.tight_layout()
plt.savefig('Birth Weight Data Histograms 4 of 4.png')

plt.show()


#################
# Outlier Identification
#################

# Checking different quantiles for outlier detection
bw_quantiles = bw.loc[:, :].quantile([0.05,
                                      0.20,
                                      0.40,
                                      0.60,
                                      0.80,
                                      0.95,
                                      1.00])

print(bw_quantiles)

#################
## Using IQR to Identify Outliers 
#################
"""
In order to identify outliers with IQR, we will use boxplots to visaully
observe the outliers that are 1.5*IQR away from Q1(0.25) and Q3(0.75) limits. 
"""

#################
# Boxplots
#################

# Mother and Father Information

f, ax = plt.subplots(figsize=(10, 8))
plt.subplot(2, 2, 1)

bw.boxplot(['mage'],
           vert = False,
           patch_artist = False,
           meanline = True,
           showmeans = True)

plt.xlabel('Mother Age')

############
plt.subplot(2, 2, 2)

bw.boxplot(['meduc'],
           vert = False,
           patch_artist = False,
           meanline = True,
           showmeans = True)


plt.xlabel('Mother Education')

############
plt.subplot(2, 2, 3)

bw.boxplot(['fage'],
           vert = False,
           patch_artist = False,
           meanline = True,
           showmeans = True)

plt.xlabel('Father Age')

############
plt.subplot(2, 2, 4)

bw.boxplot(['feduc'],
           vert = False,
           patch_artist = False,
           meanline = True,
           showmeans = True)

plt.xlabel('Father Education')

plt.tight_layout()
plt.savefig('Birth Weight Boxplots 1 of 3.png')

plt.show()

########################
########################

# Prenatal Visits and Habits

f, ax = plt.subplots(figsize=(10, 8))
plt.subplot(2, 2, 1)

bw.boxplot(['monpre'],
           vert = False,
           patch_artist = False,
           meanline = True,
           showmeans = True)

plt.xlabel('Months Pregnant')

############
plt.subplot(2, 2, 2)

bw.boxplot(['npvis'],
           vert = False,
           patch_artist = False,
           meanline = True,
           showmeans = True)


plt.xlabel('Number of Prenatal Visits')

############
plt.subplot(2, 2, 3)

bw.boxplot(['cigs'],
           vert = False,
           patch_artist = False,
           meanline = True,
           showmeans = True)

plt.xlabel('Cigarettes')

############
plt.subplot(2, 2, 4)

bw.boxplot(['drink'],
           vert = False,
           patch_artist = False,
           meanline = True,
           showmeans = True)

plt.xlabel('Drinks')

plt.tight_layout()
plt.savefig('Birth Weight Boxplots 2 of 3.png')

plt.show()

########################
########################

# Birth Weight

bw.boxplot(['bwght'],
           vert = False,
           patch_artist = False,
           meanline = True,
           showmeans = True)

plt.xlabel('Birth Weight')

plt.tight_layout()
plt.savefig('Birth Weight Boxplots 3 of 3.png')

plt.show()


"""
Based on our boxplots, we identified 8 different outlier limits for 7 different
variables that we have in the birth weight dataset.
"""
#################
# IQR Limits Calculation
#################

mage_hi_IQR = bw['mage'].quantile(0.75) + iqr(bw['mage'])*1.5

fage_hi_IQR = bw['fage'].quantile(0.75) + iqr(bw['fage'])*1.5

feduc_lo_IQR = bw['feduc'].quantile(0.25) - iqr(bw['feduc'])*1.5

monpre_hi_IQR = bw['monpre'].quantile(0.75) + iqr(bw['monpre'])*1.5

npvis_hi_IQR = bw['npvis'].quantile(0.75) + iqr(bw['npvis'])*1.5

npvis_lo_IQR = bw['npvis'].quantile(0.25) - iqr(bw['npvis'])*1.5

drink_hi_IQR = bw['drink'].quantile(0.75) + iqr(bw['drink'])*1.5

bwght_lo_IQR = bw['bwght'].quantile(0.25) - iqr(bw['bwght'])*1.5


#################
# Outlier Limits
#################
"""
Based on our histograms, research and IQR analysis, we created 3 different sets
of outlier limits based on each one of them. After that, we tried different
combinations of those outliers limits and run OLS Regression for each of the
combined set of outliers in order to get best r-squared values from our model.

Finally, we decided to have 11 different outlier limits for 9 different
variables in our dataset by using the combination of three different outlier
sets.
"""

# Based on IQR calculation (Boxplot) 
mage_hi = 65

# Based on Research and Histogram
meduc_lo = 11

# Based on IQR calculation (Boxplot)
fage_hi = 56

# Based on Research and Histogram
feduc_lo = 10

# Based on Research and Histogram
monpre_hi = 5

# Based on Research
npvis_lo = 11

# Based on IQR calculation (Boxplot)
npvis_hi = 15

# Based on Research
cigs_hi = 6

# Based on Research 
drink_hi = 7

# Based on Research and Histogram
bwght_lo = 2250

# Based on Research 
bwght_hi = 3250


#################
# Outlier Flagging
#################

# Mother Age
bw['out_mage'] = 0

for val in enumerate(bw.loc[ : , 'mage']):
    
    if val[1] >= mage_hi:
        bw.loc[val[0], 'out_mage'] = +1


# Mother Education
bw['out_meduc'] = 0

for val in enumerate(bw.loc[ : , 'meduc']):
    
    if val[1] <= meduc_lo:
        bw.loc[val[0], 'out_meduc'] = -1


# Father Age
bw['out_fage'] = 0

for val in enumerate(bw.loc[ : , 'fage']):
    
    if val[1] >= fage_hi:
        bw.loc[val[0], 'out_fage'] = +1
    

# Father Education
bw['out_feduc'] = 0

for val in enumerate(bw.loc[ : , 'feduc']):
    
    if val[1] <= feduc_lo:
        bw.loc[val[0], 'out_feduc'] = -1


# Months Pregnant
bw['out_monpre'] = 0

for val in enumerate(bw.loc[ : , 'monpre']):
    
    if val[1] >= monpre_hi:
        bw.loc[val[0], 'out_monpre'] = +1


# Number of Prenatal Visits
bw['out_npvis'] = 0

for val in enumerate(bw.loc[ : , 'npvis']):
    
    if val[1] <= npvis_lo:
        bw.loc[val[0], 'out_npvis'] = -1
        
for val in enumerate(bw.loc[ : , 'npvis']):
    
    if val[1] >= npvis_hi:
        bw.loc[val[0], 'out_npvis'] = +1
        

# Cigarettes
bw['out_cigs'] = 0

for val in enumerate(bw.loc[ : , 'cigs']):
    
    if val[1] >= cigs_hi:
        bw.loc[val[0], 'out_cigs'] = +1


## Drinks
bw['out_drink'] = 0

for val in enumerate(bw.loc[ : , 'drink']):
    
    if val[1] >= drink_hi:
        bw.loc[val[0], 'out_drink'] = +1


## Birth Weight
bw['out_bwght'] = 0

for val in enumerate(bw.loc[ : , 'bwght']):
    
    if val[1] <= bwght_lo:
        bw.loc[val[0], 'out_bwght'] = -1
        
for val in enumerate(bw.loc[ : , 'bwght']):        
        
    if val[1] >= bwght_hi:
        bw.loc[val[0], 'out_bwght'] = +1


################################
# 4) Correlation Analysis
################################

#################
# Correlation Matrix        
#################

# Creating Correlation Matrix
cor_mat = bw.corr()

# Creating Heatmap
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(cor_mat, cmap = cmap, square = True)
plt.savefig('Heatmap for BirthWeight.png')
plt.show()


#################
# Regression Plots
#################
"""
We created regression plots with variables that have significant correlation
with each other in order to understand the variables behavior more clearly.
"""

# Age, Education and Birth Weight

f, ax = plt.subplots(figsize=(10, 8))
plt.subplot(2, 2, 1)
sns.regplot(x = bw['mage'],
            y = bw['fage'],
            x_jitter = 0.2,
            color = 'xkcd:light red')

plt.xlabel('Mother Age')
plt.ylabel('Father Age')

############
plt.subplot(2, 2, 2)
sns.regplot(x = bw['meduc'],
            y = bw['feduc'],
            x_jitter = 0.2,
            color = 'xkcd:medium blue')

plt.xlabel('Mother Education')
plt.ylabel('Father Education')

############
plt.subplot(2, 2, 3)
sns.regplot(x = bw['mage'],
            y = bw['bwght'],
            x_jitter = 0.2,
            color = 'xkcd:medium green')

plt.xlabel('Mother Age')
plt.ylabel('Birth Weight')

############
plt.subplot(2, 2, 4)
sns.regplot(x = bw['fage'],
            y = bw['bwght'],
            x_jitter = 0.2,
            color = 'xkcd:dark yellow')

plt.xlabel('Father Age')
plt.ylabel('Birth Weight')

plt.tight_layout()
plt.savefig('Birth Weight Regression Plots 1 of 2.png')

plt.show()

########################
########################
# Habits and Birth Weight

f, ax = plt.subplots(figsize=(12, 4))
plt.subplot(1, 3, 1)
sns.regplot(x = bw['cigs'],
            y = bw['drink'],
            x_jitter = 0.2,
            color = 'xkcd:light red')

plt.xlabel('Cigarettes')
plt.ylabel('Drinks')

############
plt.subplot(1, 3, 2)
sns.regplot(x = bw['cigs'],
            y = bw['bwght'],
            x_jitter = 0.2,
            color = 'xkcd:medium blue')

plt.xlabel('Cigarettes')
plt.ylabel('Birth Weight')

############
plt.subplot(1, 3, 3)
sns.regplot(x = bw['drink'],
            y = bw['bwght'],
            x_jitter = 0.2,
            color = 'xkcd:medium green')

plt.xlabel('Drinks')
plt.ylabel('Birth Weight')

plt.tight_layout()
plt.savefig('Birth Weight Regression Plots 2 of 2.png')

plt.show()


#################
## All Variables vs. Birthweight Regression Plots
#################
"""
In order to clearly see the relationship of each variable with the birhtweight,
we created regression plots.
"""

# Mother and Father Information

f, ax = plt.subplots(figsize=(10, 8))
plt.subplot(2, 2, 1)
sns.regplot(x = bw['mage'],
            y = bw['bwght'],
            x_jitter = 0.2,
            color = 'xkcd:light red')

plt.xlabel('Mother Age')
plt.ylabel('Birth Weight')

############
plt.subplot(2, 2, 2)
sns.regplot(x = bw['meduc'],
            y = bw['bwght'],
            x_jitter = 0.2,
            color = 'xkcd:light red')

plt.xlabel('Mother Education')
plt.ylabel('Birth Weight')

############
plt.subplot(2, 2, 3)
sns.regplot(x = bw['fage'],
            y = bw['bwght'],
            x_jitter = 0.2,
            color = 'xkcd:medium blue')

plt.xlabel('Father Age')
plt.ylabel('Birth Weight')

############
plt.subplot(2, 2, 4)
sns.regplot(x = bw['feduc'],
            y = bw['bwght'],
            x_jitter = 0.2,
            color = 'xkcd:medium blue')

plt.xlabel('Father Education')
plt.ylabel('Birth Weight')

plt.tight_layout()
plt.savefig('Regression Plots 1 of 2.png')

plt.show()

########################
########################

# Pregnancy and Habits

f, ax = plt.subplots(figsize=(10, 8))
plt.subplot(2, 2, 1)
sns.regplot(x = bw['monpre'],
            y = bw['bwght'],
            x_jitter = 0.2,
            color = 'xkcd:dark yellow')

plt.xlabel('Months Pregnant')
plt.ylabel('Birth Weight')

############
plt.subplot(2, 2, 2)
sns.regplot(x = bw['npvis'],
            y = bw['bwght'],
            x_jitter = 0.2,
            color = 'xkcd:dark yellow')

plt.xlabel('Number of Prenatal Visits')
plt.ylabel('Birth Weight')

############
plt.subplot(2, 2, 3)
sns.regplot(x = bw['cigs'],
            y = bw['bwght'],
            x_jitter = 0.2,
            color = 'xkcd:medium green')

plt.xlabel('Cigarettes')
plt.ylabel('Birth Weight')

############
plt.subplot(2, 2, 4)
sns.regplot(x = bw['drink'],
            y = bw['bwght'],
            x_jitter = 0.2,
            color = 'xkcd:medium green')

plt.xlabel('Drinks')
plt.ylabel('Birth Weight')

plt.tight_layout()
plt.savefig('Plots 2 of 2.png')

plt.show()


###############################################################################
# B) MODEL TRIALS AND FEATURE ENGINEERING
###############################################################################

################################
# 1) OLS Trials and Feature Engineering
################################

#################
# Initial sklearn OLS Trial
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
y_pred = bw_lr.predict(X_train)

y_hat_pred = bw_lr.predict(X_test)

# Comparing Training and Test Scores
y_score = bw_lr.score(X_train, y_train)

y_hat_score = bw_lr.score(X_test, y_test)

print(y_score)
print(y_hat_score)


"""
We have good r-squared values for both training and test datasets. However, we
should check the variables and their p values in order to see if there is any 
room for improvement.
"""

#################
# Initial statsmodels OLS Trial
#################

# Creating OLS Regression for explored dataset
lm_bw = smf.ols(formula = """bwght ~ bw['mage'] +
                                     bw['meduc'] +
                                     bw['monpre'] +
                                     bw['npvis'] +
                                     bw['fage'] +
                                     bw['feduc'] +
                                     bw['cigs'] +
                                     bw['drink'] +
                                     bw['male'] +
                                     bw['mwhte'] +
                                     bw['mblck'] +
                                     bw['moth'] +
                                     bw['fwhte'] +
                                     bw['fblck'] +
                                     bw['foth'] + 
                                     bw['m_meduc'] +
                                     bw['m_npvis'] +
                                     bw['m_feduc'] +
                                     bw['out_mage'] +
                                     bw['out_meduc'] +
                                     bw['out_fage'] +
                                     bw['out_feduc'] +
                                     bw['out_monpre'] +
                                     bw['out_npvis'] +
                                     bw['out_cigs'] +
                                     bw['out_drink'] +
                                     bw['out_bwght']
                                     """,
                                     data = bw)

# Fitting results
results_bw = lm_bw.fit()

# Printing Summary Statistics
print(results_bw.summary())

print(f"""
Summary Statistics:
R-Squared:          {results_bw.rsquared.round(3)}
Adjusted R-Squared: {results_bw.rsquared_adj.round(3)}
""")

# Checking variables that have p value more than 0.05
lm_bw_high_p = results_bw.pvalues.round(3) >= 0.05

results_bw.pvalues[lm_bw_high_p]

"""
We can see that the model has 16 variables which have higher p values than 0.05.
Therefore, we decided to use feature engineering to remove the number of
variables that have high p values, without significantly reducing our r-squared
values.
"""
#################
# Feature Engineering
#################
"""
Based on our research and trials, we decided that we can create 4 different new
variables by combining other variables, and we also decided to use 4 of our 
variables' flagged columns to improve our model. Therefore, our conclusions:
    - We will check outliers for cigarettes and drinks together.
    - We will check outliers for father and mother age together.
    - We will check father and mother education together.
    - We will check outliers for father and mother education together.
    
    - We will use missing and outlier flags for number of prenatal visits.
    - We will use combination of father and mother outlier age metric, instead
    of father age.
    - We will use outlier flag for months pregnant variable.
    - We decided not to use male variable, because we couldn't find any
    relationship between the gender of the baby and the birth weight in our
    dataset.
"""

bw_1 = bw[:]

bw_1['out_habit'] = round((bw_1['out_cigs'] + bw_1['out_drink']+0.1)/2,0)

bw_1['out_age'] = bw_1['out_mage'] + bw_1['out_fage']

bw_1['out_educ'] = round((bw_1['out_meduc'] + bw_1['out_feduc']+0.1)/2,0)

bw_1['educ'] = bw_1['meduc'] + bw_1['feduc']

bw_1 = bw_1.drop(['out_cigs',
                  'out_drink',
                  'out_mage',
                  'out_fage',
                  'out_meduc',
                  'out_feduc',
                  'npvis',
                  'fage',
                  'monpre',
                  'male',
                  'meduc',
                  'feduc'],
                  axis = 1)


################################
################################

#################
# Final sklearn OLS Trial
#################

# Data and Target Split
bw_1_lr_data = bw_1.drop(['bwght'],
                          axis = 1)
bw_1_lr_target = pd.DataFrame(bw_1['bwght'])

# Test and Training Split
X_train, X_test, y_train, y_test = train_test_split(
            bw_1_lr_data,
            bw_1_lr_target,
            random_state = 508,
            test_size = 0.1)

# Preparing the Model
bw_1_lr= LinearRegression()

# Fitting Results
bw_1_lr.fit(X_train, y_train)

# Predicting Train and Test Dataset
y_pred = bw_1_lr.predict(X_train)

y_hat_pred = bw_1_lr.predict(X_test)

# Comparing Training and Test Scores
y_score = bw_1_lr.score(X_train, y_train)

y_hat_score = bw_1_lr.score(X_test, y_test)

print(y_score)
print(y_hat_score)


#################
# Final statsmodels OLS Trial
#################

# Creating OLS Regression for explored dataset
lm_bw_1 = smf.ols(formula = """bwght ~ bw_1['mage'] +
                                       bw_1['cigs'] +
                                       bw_1['drink'] +
                                       bw_1['mwhte'] +
                                       bw_1['mblck'] +
                                       bw_1['moth'] +
                                       bw_1['fwhte'] +
                                       bw_1['fblck'] +
                                       bw_1['foth'] +
                                       bw_1['m_meduc'] +
                                       bw_1['m_npvis'] +
                                       bw_1['m_feduc'] +
                                       bw_1['out_monpre'] +
                                       bw_1['out_npvis'] +
                                       bw_1['out_bwght'] +
                                       bw_1['out_habit'] +
                                       bw_1['out_age'] + 
                                       bw_1['out_educ'] +
                                       bw_1['educ']
                                       """,
                                       data = bw_1)

# Fitting results
results_bw_1 = lm_bw_1.fit()

# Printing Summary Statistics
print(results_bw_1.summary())

print(f"""
Summary Statistics:
R-Squared:          {results_bw_1.rsquared.round(3)}
Adjusted R-Squared: {results_bw_1.rsquared_adj.round(3)}
""")

# Checking variables that have p value more than 0.05
lm_bw_1_high_p = results_bw_1.pvalues.round(3) >= 0.05

results_bw_1.pvalues[lm_bw_1_high_p]

"""
By using feature engineering and creating/dropping some columns, we increased
our models' accuracy on predicting test dataset. Our next step will be exporting
both the explored and the feature engineered dataset. Using our featured dataset to
compare different models, we will find the best model for our problem.
"""
################################
# 2) Exporting Final Datasets
################################

bw.to_excel('birthweight_explored.xlsx')
bw_1.to_excel('birthweight_featured.xlsx')

