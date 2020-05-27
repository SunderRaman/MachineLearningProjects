#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
from dateutil import parser
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sn
import os
import time
import matplotlib
import random


# In[3]:


# Set the working directory & Read the data
os.chdir("C:\\Sunder\\DataScience\\MachineLearning\\ChurnRateMinimzation")
dataset = pd.read_csv("churn_data.csv")


# In[4]:


dataset.shape


# In[7]:


# Data Cleansing
dataset.isna().any()
dataset.isna().sum()
# Two column having large NA and hence dropping them from analysis
dataset = dataset.drop(columns = ['credit_score', 'rewards_earned'])


# In[8]:


# Data Viewing using Histograms
dataset2 = dataset.drop(columns = ['user', 'churn'])
fig = plt.figure(figsize=(15, 12))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(6, 5, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i - 1])

    vals = np.size(dataset2.iloc[:, i - 1].unique())
    
    plt.hist(dataset2.iloc[:, i - 1], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# In[10]:


# Viewing the distribution of data for various columns using piechart
dataset2 = dataset[['housing', 'is_referred', 'app_downloaded',
                    'web_user', 'app_web_user', 'ios_user',
                    'android_user', 'registered_phones', 'payment_type',
                    'waiting_4_loan', 'cancelled_loan',
                    'received_loan', 'rejected_loan', 'zodiac_sign',
                    'left_for_two_month_plus', 'left_for_one_month', 'is_referred']]
fig = plt.figure(figsize=(15, 12))
plt.suptitle('Pie Chart Distributions', fontsize=20)
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(6, 3, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i - 1])
   
    values = dataset2.iloc[:, i - 1].value_counts(normalize = True).values
    index = dataset2.iloc[:, i - 1].value_counts(normalize = True).index
    plt.pie(values, labels = index, autopct='%1.2f%%')
    plt.axis('equal')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])


# In[20]:


# Five columns has very low variations and hence analyzing them further on how they effect the churn variable
dataset[dataset2.waiting_4_loan == 1].churn.value_counts() # when the value is 1, the churn variable variation is 3:1
dataset[dataset2.cancelled_loan == 1].churn.value_counts() # when the value is 1, the churn variable variation is 1:1
dataset[dataset2.received_loan == 1].churn.value_counts() # when the value is 1, the churn variable variation is 3:2
dataset[dataset2.rejected_loan == 1].churn.value_counts() # w1hen the value is 1, the churn variable variation is 4:1
dataset[dataset2.left_for_one_month == 1].churn.value_counts() # when the value is 1, the churn variable variation is 13:11
# Since there is no lop sided variation in churn variable due to these low variations in these variables, retaining them


# In[21]:


# ## Correlation with Response Variable (for Numerical variables)
dataset2.drop(columns = ['housing', 'payment_type',
                         'registered_phones', 'zodiac_sign']
    ).corrwith(dataset.churn).plot.bar(figsize=(20,10),
              title = 'Correlation with Response variable',
              fontsize = 15, rot = 45,
              grid = True)


# In[22]:


# Correlation Matrix between Independent Variables
# Compute the correlation matrix
corr = dataset.drop(columns = ['user', 'churn']).corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))

# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[23]:


# Removing Correlated Fields
dataset = dataset.drop(columns = ['ios_user'])


# In[24]:


# Creating the new .csv file
dataset.to_csv('updated_churn_data.csv', index = False)


# In[26]:


## Data Preparation for model building
dataset = pd.read_csv('updated_churn_data.csv')
user_identifier = dataset['user']
dataset = dataset.drop(columns = ['user'])


# In[36]:


## One hot encoding 
dataset.housing.value_counts()
dataset.groupby('housing')['churn'].nunique().reset_index()
dataset = pd.get_dummies(dataset)


# In[40]:


dataset.columns


# In[39]:


# Dropping the new columns with _na (after hot encoding)
dataset.drop(list(dataset.filter(regex = '_na')), axis = 1, inplace = True)


# In[41]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset.drop(columns = 'churn'), dataset['churn'],
                                                    test_size = 0.2,
                                                    random_state = 0)


# In[45]:


# Balancing the Training Set (The ratio is nearly 3:2)
y_train.value_counts()
pos_index = y_train[y_train.values == 1].index
neg_index = y_train[y_train.values == 0].index

if len(pos_index) > len(neg_index):
    higher = pos_index
    lower = neg_index
else:
    higher = neg_index
    lower = pos_index

random.seed(0)
higher = np.random.choice(higher, size=len(lower))
lower = np.asarray(lower)
new_indexes = np.concatenate((lower, higher))

X_train = X_train.loc[new_indexes,]
y_train = y_train[new_indexes]


# In[47]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2


# # Model Building

# In[53]:


# Fitting Model to the Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0,solver='lbfgs')
classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = classifier.predict(X_test)


# In[54]:


# Evaluating Results
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred) # tp / (tp + fp)
recall_score(y_test, y_pred) # tp / (tp + fn)
f1_score(y_test, y_pred)


# In[55]:


# Displaying the confusion matrix
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))


# In[56]:


## K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("SVM Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))


# In[57]:


# Displaying the coefficients
pd.concat([pd.DataFrame(X_train.columns, columns = ["features"]),
           pd.DataFrame(np.transpose(classifier.coef_), columns = ["coef"])
           ],axis = 1)


# ## Feature Selection

# In[58]:


## Use Recursive Feature Elimination from scikitLearn and selecting top 20 features
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# In[61]:


# Model to Test
classifier = LogisticRegression(solver = 'lbfgs')
# Select Best 20 Features
rfe = RFE(classifier, 20)
rfe = rfe.fit(X_train, y_train)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)
X_train.columns[rfe.support_]


# In[62]:


# Check for correlation among these features and display

# Compute the correlation matrix
corr = X_train[X_train.columns[rfe.support_]].corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))

# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})    

# Results:- Deposits has strong correlation with withdrawl, purchases, but its value is only 0.3. 
# Hence, considering all columns for further model building


# In[71]:


# Model building based on the 20 features
# Fitting Model to the Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver = 'lbfgs')
classifier.fit(X_train[X_train.columns[rfe.support_]], y_train)

# Predicting Test Set
y_pred = classifier.predict(X_test[X_train.columns[rfe.support_]])

# Evaluating Results
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred) # tp / (tp + fp)
recall_score(y_test, y_pred) # tp / (tp + fn)
f1_score(y_test, y_pred)


# In[68]:


# Displaying the Confusion matrix with these 20 features
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))


# In[72]:


# Applying k fold cross validation with 10 folds
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier,
                             X = X_train[X_train.columns[rfe.support_]],
                             y = y_train, cv = 10)
print("SVM Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))


# In[74]:


# Formatting Final Results and comparing the churn values (Actual Vs Predicted)
final_results = pd.concat([y_test, user_identifier], axis = 1).dropna()
final_results['predicted_churn'] = y_pred
final_results = final_results[['user', 'churn', 'predicted_churn']].reset_index(drop=True)
final_results


# In[ ]:




