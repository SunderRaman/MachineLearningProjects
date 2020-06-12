#!/usr/bin/env python
# coding: utf-8

# # Loan Default Analysis
# This is an EDA project which comprises of 4 parts
#  1. Data Understanding
#  2. Data Cleansing
#  3. Data Analysis
#  4. Recommendations

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[51]:


# Set the working directory & Read the data
os.chdir("C:\\Sunder\\DataScience\\MachineLearning\\Projects\\LoanAnalysis")
loan = pd.read_csv("loan.csv", sep = ",", dtype = {"last_pymnt_d":object})
loan.info()


# # Data Understanding

# In[9]:


loan.head()


#  Data has 111 columns and from the names of the columns, some of the important columns are loan_amnt, term, int_rate, grade, sub_grade.
#     
#  The **target Variable** is **loan_status**. The idea is to  compare the average default rates across various independent variables 
#  and identify the ones that affect default rate the most.   

# # Data Cleansing

# In[12]:


# Identifying NULL values
loan.isnull().sum()
# Expressing above value as a percentage
round(loan.isnull().sum()/len(loan.index), 2)*100


# There are quite a few columns which has 100% missing values and hence these can be removed

# In[52]:


# Removing columns wiht 100% NULL values
missing_columns = loan.columns[100*(loan.isnull().sum()/len(loan.index)) > 90]
print(missing_columns)


# In[53]:


loan = loan.drop(missing_columns, axis=1)
print(loan.shape)


# There were 56 columns that had more than 90% missing data and hence removed them

# In[45]:


# Checking further for missing data
round(loan.isnull().sum()/len(loan.index), 2)*100


# In[54]:


# Taking a look at the data in the missing columns
loan.loc[:, ['desc', 'mths_since_last_delinq']].head()


# - **mths_since_last_deling** is the number months passed since the person last fell into the 90 DPD group. 
#     Since at the time of loan application, we will not have this data (This gets generated months after the loan 
#     has been approved), it cannot be used as a predictor of default at the time of loan approval.
# - **desc** contains description contains the comments the applicant had written while applying for the loan. It 
#     may contain features such as sentiment, positive/negative words etc, but this will not be used for further analysis    

# In[55]:


# Dropping the above two columns
loan = loan.drop(['desc', 'mths_since_last_delinq'], axis=1)


# In[18]:


# Check for  columns with missing data
round(loan.isnull().sum()/len(loan.index), 2)*100


# There are only 3 columns and that too, the percentage is very small. Hence, we can retain those columns

# In[21]:


#  Checking for rows with  columns having NULL values
loan.isnull().sum(axis=1)


# In[20]:


# checking whether some rows have more than 5 missing values
len(loan[loan.isnull().sum(axis=1) > 5].index)


# In[36]:


loan.info()


# In[56]:


# Replace int_rate to numeric
loan['int_rate'] = loan['int_rate'].apply(lambda x: pd.to_numeric(x.split("%")[0]))


# In[57]:


# Extract the numeric part from the variable employment length

# dropping the missing values from the column (otherwise the regex code below throws error)
loan = loan[~loan['emp_length'].isnull()]

# using regular expression to extract numeric values from the string
import re
loan['emp_length'] = loan['emp_length'].apply(lambda x: re.findall('\d+', str(x))[0])

# convert to numeric
loan['emp_length'] = loan['emp_length'].apply(lambda x: pd.to_numeric(x))


# In[62]:


loan.info()


# In[59]:


# also, lets extract the numeric part from the variable employment length

# first, let's drop the missing values from the column (otherwise the regex code below throws error)
loan = loan[~loan['term'].isnull()]

# using regular expression to extract numeric values from the string
import re
loan['term'] = loan['term'].apply(lambda x: re.findall('\d+', str(x))[0])

# convert to numeric
loan['term'] = loan['term'].apply(lambda x: pd.to_numeric(x))


# # Data Analysis

# The objective of the analysis is to identify those predictor variables that helps in determining 
# whether to sanction the loan. Broadly, there are 3 categories of variables
# - Variables related to applicant (Age, Occupation, Employment Details,etc)
# - Characteristics of the Loan (Loan Amount, Interest Rate, purpose,etc)
# - Applicant Behaviours (those which are generated after the loan is approved such as delinquent 2 years, revolving balance, next payment date etc)
# 
# For sanctioning of loan, the third category is not required. Hence, we shall be removing those variables for analysis

# In[64]:


# Removing behaviour varaibles
behaviour_var =  [
  "delinq_2yrs",
  "earliest_cr_line",
  "inq_last_6mths",
  "open_acc",
  "pub_rec",
  "revol_bal",
  "revol_util",
  "total_acc",
  "out_prncp",
  "out_prncp_inv",
  "total_pymnt",
  "total_pymnt_inv",
  "total_rec_prncp",
  "total_rec_int",
  "total_rec_late_fee",
  "recoveries",
  "collection_recovery_fee",
  "last_pymnt_d",
  "last_pymnt_amnt",
  "last_credit_pull_d",
  "application_type"]
df = loan.drop(behaviour_var, axis=1)
df.info()


# In[65]:


# Also, we will not be able to use the variables title, zip code, address, state etc.

df = df.drop(['title', 'url', 'zip_code', 'addr_state'], axis=1)


# Next, from the target variable - loan_status, We need to relabel the values to a binary form - 0 or 1, 
#  - 1 indicating that the person has defaulted 
#  - 0 otherwise.

# In[66]:


df['loan_status'] = df['loan_status'].astype('category')
df['loan_status'].value_counts()


# "Current" Loan status are not fully paid nor defaults, hence we can remove those rows for our analysis

# In[67]:


# Removing rows where loan_status is "Current"
df = df[df['loan_status'] != 'Current']


# In[68]:


# Mark Charged Off loan as "1" and Fully paid as "0"
df['loan_status'] = df['loan_status'].apply(lambda x: 0 if x=='Fully Paid' else 1)
# Convert to numeric type
df['loan_status'] = df['loan_status'].apply(lambda x: pd.to_numeric(x))
# Summarizing the loan_status data
df['loan_status'].value_counts()


# # Data Analysis

# ## Univariate Analysis

# In[71]:


# Mean of charged off loans
round(np.mean(df['loan_status']), 4)


# Average default rate is 14.38%

# Identify Default Rate across categories

# In[92]:


# Define a function to plot loan_status across categorical variables
def plot_cat(cat_var):
    sns.barplot(x=cat_var, y='loan_status',data=df)
    plt.show()


# In[93]:


# plotting default rates across grade of the loan
plot_cat("grade")


# As the grade of loan goes from A to G, the default rate increases. This is expected 

# In[94]:


# Plot the subgrade
plt.figure(figsize=(16, 6))
plot_cat('sub_grade')


# In[88]:


# Plot in ascending order of sub-grades
plt.figure(figsize=(16, 6))
sns.barplot(x='sub_grade', y='loan_status', data=df, order =["A1","A2","A3","A4","A5", 
                                                             "B1","B2","B3","B4","B5",
                                                             "C1","C2","C3","C4","C5",
                                                             "D1","D2","D3","D4","D5",
                                                             "E1","E2","E3","E4","E5",
                                                             "F1","F2","F3","F4","F5",
                                                             "G1","G2","G3","G4","G5"])


# In[95]:


# home ownership: Is it a discriminator?
plot_cat('home_ownership')


# Home ownership is not a great discriminator

# In[96]:


# Purpose of the loan
plt.figure(figsize=(16, 6))
plot_cat('purpose')


# From above, the order (top 3) of defaulting of loans based on category is
# - Small Business
# - Renewable Energy
# - House

# In[97]:


from datetime import datetime
df['issue_d'] = df['issue_d'].apply(lambda x: datetime.strptime(x, '%b-%y'))


# In[99]:


# extracting month and year from issue_date
df['month'] = df['issue_d'].apply(lambda x: x.month)
df['year'] = df['issue_d'].apply(lambda x: x.year)


# In[100]:


# Loans by year
df.groupby('year').year.count()


# In[103]:


df.groupby('year').sum()['loan_amnt']


# The number of **loans**  and **loan amount** has steadily increased across years

# In[104]:


# number of loans across months
df.groupby('month').month.count()


# In[105]:


# amount of loans across months
df.groupby('month').sum()['loan_amnt']


# Most loans are sanctioned in the month of December and the amount is also larger in the month of December

# In[106]:


# loan amount plot: 
sns.distplot(df['loan_amnt'])
plt.show()


# The median loan amount is around 10,000

# We can identify the defaulting of loan amount based on the amount of loan sactioned.
# - First we bin the amounts into four categories
# - Plot the loan amount

# In[107]:


# binning loan amount
def loan_amount(n):
    if n < 5000:
        return 'low'
    elif n >=5000 and n < 15000:
        return 'medium'
    elif n >= 15000 and n < 25000:
        return 'high'
    else:
        return 'very high'
        
df['loan_amnt_bin'] = df['loan_amnt'].apply(lambda x: loan_amount(x))


# In[109]:


plot_cat('loan_amnt_bin')


# In[110]:


df['loan_amnt_bin'].value_counts()


# "bin" the interest rates nd plot them to see it's impact on the loan_status

# In[111]:


# binning interest_rate
def int_rate(n):
    if n <= 10:
        return 'low'
    elif n > 10 and n <=15:
        return 'medium'
    else:
        return 'high'
    
    
df['int_rate_bin'] = df['int_rate'].apply(lambda x: int_rate(x))


# In[112]:


plot_cat('int_rate_bin')


# Higher the interest rate, higher is the defaulting

# In[ ]:


- Bin the dti (Debt to Income Ratio)
- Plot the binned dti


# In[113]:


# debt to income ratio
def dti(n):
    if n <= 10:
        return 'low'
    elif n > 10 and n <=20:
        return 'medium'
    else:
        return 'high'
    

df['dti_bin'] = df['dti'].apply(lambda x: dti(x))


# In[114]:


plot_cat('dti_bin')


# Higher the debt to income ratio, higher is the defaulting

# Binning based on installment amount and plot the samae

# In[115]:


# installment
def installment(n):
    if n <= 200:
        return 'low'
    elif n > 200 and n <=400:
        return 'medium'
    elif n > 400 and n <=600:
        return 'high'
    else:
        return 'very high'
    
df['installment_bin'] = df['installment'].apply(lambda x: installment(x))


# In[116]:


plot_cat('installment_bin')


# Higher the installment, higher is the default

# Binning the annual income and plot the same

# In[118]:


# annual income
def annual_income(n):
    if n <= 50000:
        return 'low'
    elif n > 50000 and n <=100000:
        return 'medium'
    elif n > 100000 and n <=150000:
        return 'high'
    else:
        return 'very high'

df['annual_inc_bin'] = df['annual_inc'].apply(lambda x: annual_income(x))


# In[119]:


plot_cat('annual_inc_bin')


# Binning the emp_length and plot it

# In[121]:


# first, let's drop the missing value observations in emp length
df = df[~df['emp_length'].isnull()]

# binning the variable
def emp_length(n):
    if n <= 1:
        return 'fresher'
    elif n > 1 and n <=3:
        return 'junior'
    elif n > 3 and n <=7:
        return 'senior'
    else:
        return 'expert'

df['emp_length_bin'] = df['emp_length'].apply(lambda x: emp_length(x))


# In[122]:


plot_cat('emp_length_bin')


# # MultiVariate Analysis

# For the multivariate analysis, we will segment the loan applications across the purpose of the loan, 
# since that is a variable affecting many other variables - the type of applicant, interest rate, income, 
# and finally the default rate

# In[123]:


plt.figure(figsize=(16, 6))
plot_cat('purpose')


# From the above, we can determine that the top four purposes who default on the loan are
# - Small Business
# - Renewable Energy
# - House
# - Educational

# In[130]:


# lets first look at the number of loans for each type (purpose) of the loan
#plt.figure(figsize=(16, 6))
#sns.countplot(x='purpose', data=df)
#plt.show()

#ax = sns.countplot(x="purpose", data=df)
#ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
#plt.figure(figsize=(16, 6))
#plt.tight_layout()
#plt.show()

plt.figure(figsize=(16,6))
chart = sns.countplot(x="purpose",data=df)

plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)
# plt.tight_layout()
plt.show()


# In[131]:


# Value Counts based on purpose
df['purpose'].value_counts()


# Among the top 5 purposes choosing the following 4 purposes for the multivariate analysis
# - Debt Consolidation
# - credit_card
# - home_improvement
# - major_purchase

# In[132]:


# filtering the df for the 4 types of loans mentioned above
main_purposes = ["credit_card","debt_consolidation","home_improvement","major_purchase"]
df4 = df[df['purpose'].isin(main_purposes)]
df4['purpose'].value_counts()


# In[135]:


# Create a function which takes a categorical variable and plots the default rate
# segmented by purpose 

def plot_segmented(cat_var):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=cat_var, y='loan_status', hue='purpose', data=df4)
    plt.show()


# In[136]:


# Plot term
plot_segmented('term')


# In[137]:


plot_segmented('grade')


# In[138]:


# home ownership
plot_segmented('home_ownership')


# We see that debt consolidation is the major reason for default in all the above 

# We shall see for 5 more variables (year, loan_amount, int_rate, dti, annual_income)

# In[140]:


#year
plot_segmented('year')


# In[143]:


# loan_amount
plot_segmented('loan_amnt_bin')


# In[144]:


# int_rate
plot_segmented('int_rate_bin')


# In[145]:


# dti
plot_segmented('dti_bin')


# In[147]:


# annual_income
plot_segmented('annual_inc_bin')


# Again in all the above 5 variables, debt_consolidation is the major factor for defaulting of loan

# In[150]:


# A function which takes in a categorical variable and computes the average 
# default rate across the categories
# It also computes the 'difference between the highest and the lowest default rate' across the 
# categories, which is a metric indicating the effect of the varaible on default rate

def diff_rate(cat_var):
    default_rates = df4.groupby(cat_var).loan_status.mean().sort_values(ascending=False)
    return (round(default_rates, 4), round(default_rates[0] - default_rates[-1], 4))


# In[151]:


default_rates, diff = diff_rate('annual_inc_bin')
print(default_rates) 
print(diff)


# There is an increase of 6% of loan defaulting rate, as we move from high to low in "annual income"

# In[152]:


# filtering all the object type variables
df_categorical = df.loc[:, df.dtypes == object]
df_categorical['loan_status'] = df['loan_status']


# In[153]:


# storing the diff of default rates for each column
d = {key: diff_rate(key)[1]*100 for key in df_categorical.columns if key != 'loan_status'}
print(d)


# In[ ]:




