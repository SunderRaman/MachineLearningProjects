#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Classification

# In[1]:


# I shall be using the data from sklearn for Breast Cancer Classification

# import libraries 
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import seaborn as sns # Statistical data visualization


# In[2]:


# Import Cancer data drom the Sklearn library
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()


# In[5]:


cancer


# In[6]:


# Go through the attributes of the dataset
print(cancer['DESCR'])


# In[7]:


# Shape of the object and preparing the dataframe
cancer['data'].shape


# In[11]:


df_cancer = pd.DataFrame(np.c_[cancer['data'],cancer['target']], columns = np.append(cancer['feature_names'],'target'))
df_cancer.head()


# In[12]:


df_cancer.shape


# # Visualizing the Data

# In[13]:


# Plotting a few variables (5 of them)
sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'] )


# In[14]:


# Checking the correlation between the variables 
# Strong correlation between the mean radius and mean perimeter, mean area and mean primeter
plt.figure(figsize=(20,10)) 
sns.heatmap(df_cancer.corr(), annot=True) 


#  # Model Training

# In[16]:


X = df_cancer.drop(['target'],axis=1)
y = df_cancer['target']
from sklearn.model_selection import train_test_split

# Split the data into training and testing (0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=5)


# In[18]:


# Implementing the SVC Model
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC(gamma = 'auto')
svc_model.fit(X_train, y_train)


# # Evaluating the Model

# In[19]:


y_predict = svc_model.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True)


# In[20]:


print(classification_report(y_test, y_predict))


# In[21]:


# This model does not predict for any malignant tumor (target: 0).  This model had not scaled the features. 
# We shall now trying with scaling the features
min_train = X_train.min()
train_range = (X_train - min_train).max()
X_train_sc = (X_train - min_train)/train_range


# In[23]:


# Scaling the Test Data as well
min_test = X_test.min()
test_range = (X_test - min_test).max()
X_test_sc = (X_test - min_test)/test_range


# In[25]:


# Fitting the model again based on scaled data
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC(gamma = 'auto')
svc_model.fit(X_train_sc, y_train)


# In[26]:


# Evaluating the metrics and plotting the heatmap for confusion matrix
y_predict = svc_model.predict(X_test_sc)
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True, fmt="d")


# In[27]:


print(classification_report(y_test,y_predict))


# # Cross Fold Validation

# In[29]:


# Implement Cross Fold Validation to check for further improving the model
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)
grid.fit(X_train_sc,y_train)


# In[30]:


grid.best_params_


# In[31]:


grid.best_estimator_


# In[36]:


# Predict the test data and plot the confusion matrix
grid_predict = grid.predict(X_test_sc)
cm = confusion_matrix(y_test,grid_predict)
sns.heatmap(cm, annot=True)


# In[34]:


print(classification_report(y_test,grid_predict))


# In[ ]:




