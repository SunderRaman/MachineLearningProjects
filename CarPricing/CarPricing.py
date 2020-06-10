#!/usr/bin/env python
# coding: utf-8

# # Car Price Prediction

# ## Data Understanding

# In[3]:


# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import os


# In[6]:


# Set the working directory & Read the data
os.chdir("C:\\Sunder\\DataScience\\MachineLearning\\Projects\\CarPricing")
cars = pd.read_csv("CarPrice.csv")


# In[7]:


cars.head()


# In[8]:


print(cars.info())


# In[10]:


cars.isna().sum()
# No null values or NA Values


# In[18]:


unique_values = cars.nunique()
print(unique_values)


# ### Understand some of the variables data through plots and counts

# In[23]:


# symboling
cars['symboling'].astype('category').value_counts()


# In[24]:


# Aspiration
cars['aspiration'].astype('category').value_counts()


# In[25]:


# Drive Wheel
cars['drivewheel'].astype('category').value_counts()


# In[26]:


# Wheel base
sns.distplot(cars['wheelbase'])
plt.show()


# In[27]:


# Stroke
sns.distplot(cars['stroke'])
plt.show()


# In[28]:


# Target variable (Price)
sns.distplot(cars['price'])
plt.show()


# ### Data Exploration

# In[30]:


# Identify all the numeric variables as this would be key during linear regression
cars_numeric = cars.select_dtypes(include=['float64', 'int64'])
cars_numeric.head()


# In[31]:


# dropping symboling and car_ID 
cars_numeric = cars_numeric.drop(['symboling', 'car_ID'], axis=1)
cars_numeric.head()


# In[32]:


# Compute correlation annd plot
cor = cars_numeric.corr()
# figure size
plt.figure(figsize=(16,8))

# heatmap
sns.heatmap(cor, cmap="YlGnBu", annot=True)
plt.show()


# ### Some Inferences
# Correlation of price with independent variables:
# - Price is highly (positively) correlated with wheelbase, carlength, carwidth, curbweight, enginesize, horsepower 
#   and (negatively) correlated with citympg and highwaympg
# 
# Some of the independent variables are correlated among themselves (wheelbase to carlength, carwidth, curbweight)      

# ## Data Cleaning

# In[33]:


# Converting symboling to categorical variable
cars['symboling'] = cars['symboling'].astype('object')
cars.info()


#  "CarName" column is a combination of company name and model name. We need to extract the company name 

# In[35]:


cars['CarName'][:10]


# In[40]:


# Extract the first token based on space
carnames = cars['CarName'].apply(lambda x: x.split(" ")[0])
carnames[:20]


# In[57]:


# There are a few names which are separated by "-". Using Regular expressions
import re

exp = re.compile(r'\w+-?\w+')
carnames = cars['CarName'].apply(lambda x: re.findall(exp, x)[0])
cars['car_company'] = carnames
# look at all values 
cars['car_company'].astype('category').value_counts().sort_index(ascending=True)


# Some company names like toyota/Toyouta , nissan/Nissan are misspelled or multispelled in different cases. We need to merge them

# In[53]:


# Replacing misspelled carnames 
# volkswagen
cars.loc[(cars['car_company'] == "vw") | 
         (cars['car_company'] == "vokswagen")
         , 'car_company'] = 'volkswagen'

# porsche
cars.loc[cars['car_company'] == "porcshce", 'car_company'] = 'porsche'

# toyota
cars.loc[cars['car_company'] == "toyouta", 'car_company'] = 'toyota'

# nissan
cars.loc[cars['car_company'] == "Nissan", 'car_company'] = 'nissan'

# mazda
cars.loc[cars['car_company'] == "maxda", 'car_company'] = 'mazda'


# In[56]:


cars['car_company'].astype('category').value_counts().sort_index()


# In[58]:


# drop carname variable
cars = cars.drop('CarName', axis=1)


# ## Data Preparation for Model Bulding

# In[59]:


X = cars.loc[:, ['symboling', 'fueltype', 'aspiration', 'doornumber',
       'carbody', 'drivewheel', 'enginelocation', 'wheelbase', 'carlength',
       'carwidth', 'carheight', 'curbweight', 'enginetype', 'cylindernumber',
       'enginesize', 'fuelsystem', 'boreratio', 'stroke', 'compressionratio',
       'horsepower', 'peakrpm', 'citympg', 'highwaympg',
       'car_company']]

y = cars['price']


# In[61]:


cars_categorical = X.select_dtypes(include=['object'])
cars_categorical


# In[62]:


# Convert the categorical values to columns
cars_dummies = pd.get_dummies(cars_categorical, drop_first=True)
cars_dummies.head()


# In[63]:


# Drop the categorical columns and merge the dummies columns
X = X.drop(list(cars_categorical.columns), axis=1)
X = pd.concat([X, cars_dummies], axis=1)


# In[64]:


# scaling the features
from sklearn.preprocessing import scale

# storing column names in cols, since column names are (annoyingly) lost after 
# scaling (the df is converted to a numpy array)
cols = X.columns
X = pd.DataFrame(scale(X))
X.columns = cols
X.columns


# In[66]:


# split into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.7,
                                                    test_size = 0.3, random_state=100)


# In[67]:


# Building the model initially with all the features

# instantiate
lm = LinearRegression()

# fit
lm.fit(X_train, y_train)


# In[68]:


# Predict the dependent variable for the test values 
y_pred = lm.predict(X_test)

# metrics
from sklearn.metrics import r2_score

print(r2_score(y_true=y_test, y_pred=y_pred))


# 1. RFE:- Select 15 features using RFE (Recursive Feature Elimination) 

# In[69]:


# RFE with 15 features
from sklearn.feature_selection import RFE

# RFE with 15 features
lm = LinearRegression()
rfe_15 = RFE(lm, 15)

# fit with 15 features
rfe_15.fit(X_train, y_train)

# Printing the boolean results
print(rfe_15.support_)           
print(rfe_15.ranking_)  


# In[70]:


# making predictions using rfe model (15 features)
y_pred = rfe_15.predict(X_test)

# r-squared
print(r2_score(y_test, y_pred))


# In[71]:


# import statsmodels
import statsmodels.api as sm  

# subset the features selected by rfe_15
col_15 = X_train.columns[rfe_15.support_]

# subsetting training data for 15 selected columns
X_train_rfe_15 = X_train[col_15]

# add a constant to the model
X_train_rfe_15 = sm.add_constant(X_train_rfe_15)
X_train_rfe_15.head()


# In[72]:


# fit the model with these 15 variables
lm_15 = sm.OLS(y_train, X_train_rfe_15).fit()   
print(lm_15.summary())


# In[73]:


# making predictions using rfe_15 sm model
X_test_rfe_15 = X_test[col_15]


# # Adding a constant variable 
X_test_rfe_15 = sm.add_constant(X_test_rfe_15, has_constant='add')
X_test_rfe_15.info()


# # Making predictions
y_pred = lm_15.predict(X_test_rfe_15)


# In[74]:


# r-squared
r2_score(y_test, y_pred)


# ## Choosing the optimal number of features

# In[78]:


# Find the  optimal number of features. We shall be trying from 4 to 20 features
adjusted_r2 = []
r2 = []
test_r2 = []

for n_features in range(4, 20):

    # RFE with n features
    lm = LinearRegression()

    # specify number of features
    rfe_n = RFE(lm, n_features)

    # fit with n features
    rfe_n.fit(X_train, y_train)

    # subset the features selected by rfe_n
    col_n = X_train.columns[rfe_n.support_]

    # subsetting training data for n selected columns
    X_train_rfe_n = X_train[col_n]

    # add a constant to the model
    X_train_rfe_n = sm.add_constant(X_train_rfe_n)


    # fitting the model with n variables
    lm_n = sm.OLS(y_train, X_train_rfe_n).fit()
    adjusted_r2.append(lm_n.rsquared_adj)
    r2.append(lm_n.rsquared)
    
    # making predictions using rfe_15 sm model
    X_test_rfe_n = X_test[col_n]

    # # Adding a constant variable 
    X_test_rfe_n = sm.add_constant(X_test_rfe_n, has_constant='add')

    # # Making predictions
    y_pred = lm_n.predict(X_test_rfe_n)
    
    test_r2.append(r2_score(y_test, y_pred))


# In[80]:


# plotting adjusted_r2,r2,test_r2 against n_features
n_features_list = list(range(4,20))
plt.figure(figsize=(10, 8))
plt.plot(n_features_list, adjusted_r2, label="adjusted_r2")
plt.plot(n_features_list, r2, label="train_r2")
plt.plot(n_features_list, test_r2, label="test_r2")
plt.legend(loc='upper left')
plt.show()


# ### Final Model

# In[88]:


#We shall be going ahead with 12 features for the final model.
# RFE with n features
lm = LinearRegression()

# specify number of features
rfe_12 = RFE(lm, 12)

# fit with n features
rfe_12.fit(X_train, y_train)

# subset the features selected by rfe_n
col_12 = X_train.columns[rfe_12.support_]

# subsetting training data for n selected columns
X_train_rfe_12 = X_train[col_12]

# add a constant to the model
X_train_rfe_12 = sm.add_constant(X_train_rfe_12)


# fitting the model with n variables
lm_12 = sm.OLS(y_train, X_train_rfe_12).fit()

# making predictions using rfe_15 sm model
X_test_rfe_12 = X_test[col_12]

# # Adding a constant variable 
X_test_rfe_12 = sm.add_constant(X_test_rfe_12, has_constant='add')

# # Making predictions
y_pred_12 = lm_12.predict(X_test_rfe_12)

print(r2_score(y_test, y_pred_12))


# In[89]:


col_12


# In[92]:


# MultiCollinearity (12 columns)
cors = X.loc[:, list(col_12)].corr()
plt.figure(figsize=(16,8))
sns.heatmap(cors, annot=True)
plt.show()


#  Use VIF (Variance Inflation Factor) to remove the multicollinear columns based on these 12 columns and 
#  check, if it further improves the r2_score

# In[95]:


# Function to calculate VIF (Variance Inflation Factor)
from statsmodels.stats.outliers_influence import variance_inflation_factor    

def calculate_vif_(X, thresh=5.0):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]


# In[97]:


vif_based_columns = calculate_vif_(X.loc[:, list(col_12)])


# In[98]:


vif_based_columns


# In[99]:


# MultiCollinearity (9 Columns)
cors = X.loc[:, list(vif_based_columns)].corr()
plt.figure(figsize=(16,8))
sns.heatmap(cors, annot=True)
plt.show()


# In[108]:


vif_based_columns.columns


# ## Updated model (with 9 columns)

# In[111]:


# fitting the model with 9 variables
#col_9 = X_train[vif_based_columns.columns]

X_train_rfe_9 = X_train[vif_based_columns.columns]

# Add constant variable
X_train_rfe_9 = sm.add_constant(X_train_rfe_9)
                               
lm_9 = sm.OLS(y_train, X_train_rfe_9).fit()

# making predictions using rfe_15 sm model
X_test_rfe_9 = X_test[vif_based_columns.columns]

# # Adding a constant variable 
X_test_rfe_9 = sm.add_constant(X_test_rfe_9, has_constant='add')

# # Making predictions
y_pred_9 = lm_9.predict(X_test_rfe_9)

print(r2_score(y_test, y_pred_9))


# Removal of the multicollinear columns did not improve the <b>r2_score</b>. hence, we can stick with the <b>12 columns </b> 
# and the r2_score is <b> 0.9160335989800035 </b>

# In[ ]:




