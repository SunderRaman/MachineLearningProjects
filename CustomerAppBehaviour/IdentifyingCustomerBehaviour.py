#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
from dateutil import parser
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sn
import os
import time


# In[3]:


import matplotlib
matplotlib.__version__


# In[4]:


# Set the working directory
os.chdir("C:\\Sunder\\DataScience\\MachineLearning\\CustomerAppBehaviour")
dataset = pd.read_csv("appdata10.csv")


# In[5]:


dataset.head(10)


# In[107]:


dataset.describe()


# In[6]:


dataset["hour"] = dataset.hour.str.slice(1,3).astype("int")


# In[7]:


# Plot histograms of Numberical columns to understand the data
# Removing non numerical columns
dataset2 = dataset.copy().drop(columns = ["user","screen_list","first_open","enrolled_date","enrolled"])
#gs1 = gridspec.GridSpec(3, 3)
plt.suptitle('Histograms of Numerical Columns', fontsize=20)

f, ax = plt.subplots(3, 3)
f.set_size_inches(18.5, 10.5, forward=True)
for i in range(1, dataset2.shape[1] + 1):
    ax1 = plt.subplot(3, 3, i)
    #f = plt.gca()
    ax1.set_title(dataset2.columns.values[i - 1])
    vals = np.size(dataset2.iloc[:, i - 1].unique())
    plt.hist(dataset2.iloc[:, i - 1], bins=vals, color='#3F5D7D')
plt.subplots_adjust(top=0.85)    
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# In[8]:


dataset.head()


# In[9]:


# Find the correlation of the various variables with the "Response" variable
dataset2.corrwith(dataset.enrolled).plot.bar(figsize=(20,10),
                  title = 'Correlation with Reposnse variable',
                  fontsize = 15, rot = 45,
                  grid = True)
# All the integer variables has correlation in the range of  -0.15 and 0.2


# In[10]:



# Compute the correlation matrix of the independent variables
get_ipython().run_line_magic('matplotlib', 'inline')
corr = dataset2.corr()

# Createa mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))
f.suptitle("Correlation Matrix", fontsize = 20)

# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sn.set(style="white", font_scale=2)
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[11]:



# Format the Date Columns
dataset.dtypes
dataset["first_open"] = [parser.parse(row_date) for row_date in dataset["first_open"]]
dataset["enrolled_date"] = [parser.parse(row_date) if isinstance(row_date, str) else row_date for row_date in dataset["enrolled_date"]]
#dataset.dtypes


# In[12]:


#  Converting the time differennce between enrolled date and first open
dataset["difference"] = (dataset.enrolled_date-dataset.first_open).astype('timedelta64[h]')
response_hist = plt.hist(dataset["difference"].dropna(), color='#3F5D7D')
plt.title('Time Difference between Enrollment and First Login (full Range)')
plt.show()

plt.hist(dataset["difference"].dropna(), color='#3F5D7D', range = [0, 100])
plt.title('Time Difference between Enrollment and First Login - 1st 100 hours')
plt.show()

dataset.loc[dataset.difference > 48, 'enrolled'] = 0
dataset = dataset.drop(columns=['enrolled_date', 'difference', 'first_open'])


# In[13]:


# Load the Top Viewed Screens
top_screens = pd.read_csv('top_screens.csv').top_screens.values

# Mapping these screens to the fields  qnd dropping other screens
dataset["screen_list"] = dataset.screen_list.astype(str) + ','

for sc in top_screens:
    dataset[sc] = dataset.screen_list.str.contains(sc).astype(int)
    dataset['screen_list'] = dataset.screen_list.str.replace(sc+",", "")

dataset['Other'] = dataset.screen_list.str.count(",")
dataset = dataset.drop(columns=['screen_list'])


# In[14]:


dataset.head()


# In[15]:


# Grouping Screens and calculating their counts
savings_screens = ["Saving1",
                    "Saving2",
                    "Saving2Amount",
                    "Saving4",
                    "Saving5",
                    "Saving6",
                    "Saving7",
                    "Saving8",
                    "Saving9",
                    "Saving10"]
dataset["SavingCount"] = dataset[savings_screens].sum(axis=1)
dataset = dataset.drop(columns=savings_screens)

cm_screens = ["Credit1",
               "Credit2",
               "Credit3",
               "Credit3Container",
               "Credit3Dashboard"]
dataset["CMCount"] = dataset[cm_screens].sum(axis=1)
dataset = dataset.drop(columns=cm_screens)

cc_screens = ["CC1",
                "CC1Category",
                "CC3"]
dataset["CCCount"] = dataset[cc_screens].sum(axis=1)
dataset = dataset.drop(columns=cc_screens)

loan_screens = ["Loan",
               "Loan2",
               "Loan3",
               "Loan4"]
dataset["LoansCount"] = dataset[loan_screens].sum(axis=1)
dataset = dataset.drop(columns=loan_screens)


# In[16]:


dataset.tail()


# In[17]:


# Save the updated dataset for model building
dataset.head()
dataset.describe()
dataset.columns
dataset.to_csv('new_updated_app_data.csv', index = False)


# In[18]:


#### Data Pre-Processing ####

# Splitting Independent and Response Variables
response = dataset["enrolled"]
dataset = dataset.drop(columns="enrolled")

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, response,
                                                    test_size = 0.2,
                                                    random_state = 0)


# In[19]:


# Removing User_ID
train_user_id = X_train['user']
X_train = X_train.drop(columns = ['user'])
test_user_id = X_test['user']
X_test = X_test.drop(columns = ['user'])


# In[20]:


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


# In[31]:


# Fitting Model to the Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, penalty = 'l2', solver='lbfgs')
classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = classifier.predict(X_test)

# Evaluating Results
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred) # tp / (tp + fp)
recall_score(y_test, y_pred) # tp / (tp + fn)
f1_score(y_test, y_pred)


# In[32]:


# Displaying the confusion Matrix
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
fig,ax = plt.subplots(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='d')
plt.xlabel("Actual")
plt.ylabel("Predicted")
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))


# In[43]:


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy of model based on k-fold cross validation: %0.3f (+/- %0.3f) " % (accuracies.mean(), accuracies.std() * 2))


# In[34]:


# Displaying the Coefficients
pd.concat([pd.DataFrame(dataset.drop(columns = 'user').columns, columns = ["features"]),
           pd.DataFrame(np.transpose(classifier.coef_), columns = ["coef"])
           ],axis = 1)


# In[38]:


#### Model Tuning ####

## Grid Search (Round 1)
from sklearn.model_selection import GridSearchCV

# Select Regularization Method
penalty = ['l2']

# Create regularization hyperparameter space
C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# Combine Parameters
parameters = dict(C=C, penalty=penalty)

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv = 10,
                           n_jobs = -1)
t0 = time.time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
rf_best_accuracy, rf_best_parameters


# In[40]:


# Another set of Hyper parameters for tuning the model
# Select Regularization Method
penalty = ['l2']

# Create regularization hyperparameter space
C = [0.1, 0.5, 0.9, 1, 2, 5]

# Combine Parameters
parameters = dict(C=C, penalty=penalty)

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv = 10,
                           n_jobs = -1)
t0 = time.time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
rf_best_accuracy, rf_best_parameters


# In[44]:


# Generating the model with the tuned hyperparameters
# Hyperparameters (C= 0.01 and penalty - 'l2')
classifier = LogisticRegression(random_state = 0, C= 0.01, penalty = 'l2', solver='lbfgs')
classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = classifier.predict(X_test)

# Evaluating Results

cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred) # tp / (tp + fp)
recall_score(y_test, y_pred) # tp / (tp + fn)
f1_score(y_test, y_pred)


# In[45]:


# Displaying the confusion Matrix
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
fig,ax = plt.subplots(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='d')
plt.xlabel("Actual")
plt.ylabel("Predicted")
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))


# In[49]:


# Generating the final results based on the tuned model
final_results = pd.concat([test_user_id, y_test],axis=1).dropna()
final_results['predicted_results'] = y_pred
final_results[['user','enrolled','predicted_results']].reset_index(drop=True)


# In[ ]:




