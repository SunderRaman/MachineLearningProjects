#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis

# In[14]:


import pandas as pd
import numpy as np
#import keras
import seaborn as sn
import os
import time
import matplotlib.pyplot as plt


# In[15]:


# Set the working directory & Read the data
os.chdir("C:\\Sunder\\DataScience\\MachineLearning\\Projects\\CreditCardFraudDetection")
data = pd.read_csv("creditcard.csv")


# In[16]:


data.shape


# In[17]:


data.head()


# # Data Preprocessing

# In[18]:


# Normalize the amount
from sklearn.preprocessing import StandardScaler
data['normalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Amount'],axis=1)


# In[19]:


# Drop the Time
data = data.drop(['Time'],axis=1)


# In[20]:


data.shape


# In[21]:


# Separate the dependent and independent variables
X = data.iloc[:, data.columns != 'Class'] # All Independent Variables
y = data.iloc[:, data.columns == 'Class'] # Dependent Variable


# In[22]:


X.shape


# In[23]:


y.shape


# In[24]:


# Split the Data into Training and Testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=0)
print(X_train.shape)
print(X_test.shape)


# # Model Building

# ## Random Forest (Model-1)

# In[43]:


# Random Forest (Model-1)
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100,verbose=2)
random_forest.fit(X_train,y_train.values.ravel())


# In[20]:


y_pred = random_forest.predict(X_test)
random_forest.score(X_test,y_test)


# In[44]:


# Evaluate the metrics
# Evaluating Results
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
print("Accuracy Score %0.3f " % accuracy_score(y_test, y_pred))
print("Precision Score %0.3f " % precision_score(y_test, y_pred)) # tp / (tp + fp)
print("Recall Score %0.3f " % recall_score(y_test, y_pred)) # tp / (tp + fn)
print("F1 Score %0.3f " % f1_score(y_test, y_pred))


# In[46]:


df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
ax = sn.heatmap(df_cm, annot=True, fmt='g',robust=True)
ax.set(title="Confusion Matrix (RandomForest) Heatmap",
      xlabel="Predicted",
      ylabel="Actual")
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))


# ## Decision Trees (Model-2)

# In[47]:


from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train,y_train.values.ravel())


# In[48]:


y_pred = decision_tree.predict(X_test)
decision_tree.score(X_test,y_test)


# In[49]:


# Evaluate the metrics & Plotting the confusion matrix
# Evaluating Results
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
print("Accuracy Score %0.3f " % accuracy_score(y_test, y_pred))
print("Precision Score %0.3f " % precision_score(y_test, y_pred)) # tp / (tp + fp)
print("Recall Score %0.3f " % recall_score(y_test, y_pred)) # tp / (tp + fn)
print("F1 Score %0.3f " % f1_score(y_test, y_pred))
# Plotting the confusion Matrix
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
ax = sn.heatmap(df_cm, annot=True, fmt='g',robust=True)
ax.set(title="Confusion Matrix (Decision Tree) Heatmap",
      xlabel="Predicted",
      ylabel="Actual")
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))


# In[41]:


y_test['Class'].value_counts()


# ## Deep Neural Network (Model-3)

# In[26]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# In[27]:


# Convert the Train and Test data into an np array
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[28]:


neural_model = Sequential([
    Dense(units=16, input_dim = 29,activation='relu'),
    Dense(units=24,activation='relu'),
    Dropout(0.5),
    Dense(20,activation='relu'),
    Dense(24,activation='relu'),
    Dense(1,activation='sigmoid'),
])


# In[29]:


neural_model.summary()


# In[30]:


# Training the Neural network model
neural_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
neural_model.fit(X_train,y_train,batch_size=15,epochs=5)


# In[31]:


score =neural_model.evaluate(X_test, y_test)


# In[32]:


print(score)


# In[33]:


# Predict the test data
y_pred =neural_model.predict(X_test)
y_test = pd.DataFrame(y_test)


# In[34]:


# Evaluate the metrics & Plotting the confusion matrix
# Evaluating Results
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred.round())
print("Accuracy Score %0.3f " % accuracy_score(y_test, y_pred.round()))
print("Precision Score %0.3f " % precision_score(y_test, y_pred.round())) # tp / (tp + fp)
print("Recall Score %0.3f " % recall_score(y_test, y_pred.round())) # tp / (tp + fn)
print("F1 Score %0.3f " % f1_score(y_test, y_pred.round()))
# Plotting the confusion Matrix
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
ax = sn.heatmap(df_cm, annot=True, fmt='g',robust=True)
ax.set(title="Confusion Matrix (Deep Neural Network) Heatmap",
      xlabel="Predicted",
      ylabel="Actual")
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred.round()))


# In[35]:


# Predict for the entire dataset
y_pred = neural_model.predict(X)
y_expected = pd.DataFrame(y)


# In[36]:


# Calculate the metrics
cm = confusion_matrix(y_expected, y_pred.round())
print("Accuracy Score %0.3f " % accuracy_score(y_expected, y_pred.round()))
print("Precision Score %0.3f " % precision_score(y_expected, y_pred.round())) # tp / (tp + fp)
print("Recall Score %0.3f " % recall_score(y_expected, y_pred.round())) # tp / (tp + fn)
print("F1 Score %0.3f " % f1_score(y_expected, y_pred.round()))
# Plotting the confusion Matrix
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
ax = sn.heatmap(df_cm, annot=True, fmt='g',robust=True)
ax.set(title="Confusion Matrix (Deep Neural Network) Heatmap",
      xlabel="Predicted",
      ylabel="Actual")
print("Test Data Accuracy: %0.4f" % accuracy_score(y_expected, y_pred.round()))


# # Undersampling

# In[76]:


# Since this is a imbalanced Dataset, we shall undersample the dominant class (0) for a total of the number of records 
# as the number of records for the dwarf class (1 - Fraud transactions)
fraud_indices = np.array(data[data.Class == 1].index)
number_records_fraud = len(fraud_indices)
print(number_records_fraud)


# In[78]:


normal_indices = data[data.Class == 0].index
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
random_normal_indices = np.array(random_normal_indices)
print(len(random_normal_indices))


# In[79]:


under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])
print(len(under_sample_indices))


# In[80]:


# Undersample Data Preparation and split into train and test data
under_sample_data = data.iloc[under_sample_indices,:]
X_undersample = under_sample_data.iloc[:,under_sample_data.columns != 'Class']
y_undersample = under_sample_data.iloc[:,under_sample_data.columns == 'Class']
X_train, X_test, y_train, y_test = train_test_split(X_undersample,y_undersample, test_size=0.3)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[83]:


# Model Summary and prediction
neural_model.summary()
neural_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
neural_model.fit(X_train,y_train,batch_size=15,epochs=5)


# In[85]:


# Predict the test data (undersampled)
y_pred = neural_model.predict(X_test)
y_expected = pd.DataFrame(y_test)


# In[86]:


# Calculate the metrics
cm = confusion_matrix(y_expected, y_pred.round())
print("Accuracy Score %0.3f " % accuracy_score(y_expected, y_pred.round()))
print("Precision Score %0.3f " % precision_score(y_expected, y_pred.round())) # tp / (tp + fp)
print("Recall Score %0.3f " % recall_score(y_expected, y_pred.round())) # tp / (tp + fn)
print("F1 Score %0.3f " % f1_score(y_expected, y_pred.round()))
# Plotting the confusion Matrix
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
ax = sn.heatmap(df_cm, annot=True, fmt='g',robust=True)
ax.set(title="Confusion Matrix (Deep Neural Network) Heatmap",
      xlabel="Predicted",
      ylabel="Actual")
print("Test Data Accuracy: %0.4f" % accuracy_score(y_expected, y_pred.round()))


# # SMOTE

# In[2]:


pip install imblearn

from imblearn import under_sampling 
from imblearn import over_sampling
from imblearn.over_sampling import SMOTE
# In[40]:


X_resample, y_resample = SMOTE().fit_sample(X,y.values.ravel())
X_resample = pd.DataFrame(X_resample)
y_resample = pd.DataFrame(y_resample)
X_resample.head()


# In[ ]:


from imblearn.over_sampling import SMOTE
from imblearn import under_sampling, over_sampling
from sklearn.model_selection import train_test_split


# In[41]:


# Split the data into training and test (0.3)
X_train, X_test, y_train, y_test = train_test_split(X_resample,y_resample,test_size=0.3)


# In[42]:


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[44]:


neural_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
neural_model.fit(X_train,y_train,batch_size=15,epochs=5)


# In[48]:


# Predict the test data (SMOTE)
y_pred = neural_model.predict(X_test)
y_expected = pd.DataFrame(y_test)


# In[49]:


print(y_pred.shape)


# In[52]:


# Calculate the metrics
cm = confusion_matrix(y_expected, y_pred.round())
print("Accuracy Score %0.3f " % accuracy_score(y_expected, y_pred.round()))
print("Precision Score %0.3f " % precision_score(y_expected, y_pred.round())) # tp / (tp + fp)
print("Recall Score %0.3f " % recall_score(y_expected, y_pred.round())) # tp / (tp + fn)
print("F1 Score %0.3f " % f1_score(y_expected, y_pred.round()))
# Plotting the confusion Matrix
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
ax = sn.heatmap(df_cm, annot=True, fmt='g',robust=True)
ax.set(title="Confusion Matrix (Deep Neural Network) Heatmap",
      xlabel="Predicted",
      ylabel="Actual")
print("Test Data Accuracy: %0.7f" % accuracy_score(y_expected, y_pred.round()))


# In[ ]:




