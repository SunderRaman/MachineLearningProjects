#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[2]:


# Set the working directory & Read the data
os.chdir("C:\\Sunder\\DataScience\\MachineLearning\\Projects\\DataAnalytics\\FinancialInvestment")


# In[3]:


# Read the .csv files
rounds = pd.read_csv("rounds2.csv", encoding = "ISO-8859-1")
companies = pd.read_csv("companies.txt", sep="\t", encoding = "ISO-8859-1")


# In[4]:


# Number of records in rounds
print(rounds.shape)


# In[5]:


# Number of records in companies
print(companies.shape)


# In[6]:


# Column information of rounds
rounds.info()


# There are null values in **funding_round_code** and **raised_amount_usd** columns       

# In[7]:


# Column information for companies
companies.info()


# In[8]:


# identify the unique number of permalinks in companies which could be the unique key (Primary key) for each record
len(companies.permalink.unique())


# In[9]:


# converting all permalinks to lowercase
companies['permalink'] = companies['permalink'].str.lower()
companies.head()


# Let's check whether these permalink values are present in the rounds datafrme

# In[10]:


# find the number of unique permalink values in the rounds dataframe
len(rounds.company_permalink.unique())


# There are more unique permalinks than 66368 which shows that they are either additional permalink data 
# or they could be due to lowercase/uppercase combinations of a given permalink

# In[11]:


# Convert permalink to lowercase
rounds['company_permalink'] = rounds['company_permalink'].str.lower()
len(rounds.company_permalink.unique())


# In[12]:


# Identify the two extra permalinks in rounds
rounds.loc[~rounds['company_permalink'].isin(companies['permalink']), :]


# From above, it can be seen that there is some weird characters in the **company_permalink** field. We need to termine 
# the character encoding while reading the .csv file 

# In[14]:


import chardet

rawdata = open('rounds2.csv', 'rb').read()
result = chardet.detect(rawdata)
print(result['encoding'])


# WE will try with utf-8 encoding and then decode with ascii

# In[15]:


rounds['company_permalink'] = rounds.company_permalink.str.encode('utf-8').str.decode('ascii', 'ignore')
rounds.loc[~rounds['company_permalink'].isin(companies['permalink']), :]


# In[16]:


# Let's check for unique company_permalink in rounds df
len(rounds.company_permalink.unique())


# There are same number of unique companies in both round and companies df. We can check, 
# if the encoding problem exists in the **companies** df 

# In[17]:


companies.loc[~companies['permalink'].isin(rounds['company_permalink']), :]


# In[18]:


# Removing the special characters from **companies** df

companies['permalink'] = companies.permalink.str.encode('utf-8').str.decode('ascii', 'ignore')


# In[19]:


companies.loc[~companies['permalink'].isin(rounds['company_permalink']), :]


# The data is now clean for these 2 dataframes and we shall write it out to a file before proceeding for further cleansing

# In[20]:


# write rounds file
rounds.to_csv("rounds_clean.csv", sep=',', index=False)

# write companies file
companies.to_csv("companies_clean.csv", sep='\t', index=False)


# # Data Cleaning (Part-2)

# In[22]:


# read the new, decoded csv files
rounds = pd.read_csv("rounds_clean.csv", encoding = "ISO-8859-1")
companies = pd.read_csv("companies_clean.csv", sep="\t", encoding = "ISO-8859-1")

# Check for the unique permalink values
print("Companies in companies df ", len(companies.permalink.unique()))
print("Companies in rounds df ",len(rounds.company_permalink.unique()))

# Companies in rounds and not in companies df (Ideally should be zero)
print("Companies in rounds but not in companies df ", len(rounds.loc[~rounds['company_permalink'].isin(companies['permalink']), :]))


# ## Identifying Missing Values

# In[23]:


# missing values in companies df
companies.isnull().sum()


# In[24]:


# missing values in rounds df
rounds.isnull().sum()


# In[25]:


# We shall merge the two dataframes based on permalink and company_permalink
master = pd.merge(companies, rounds, how="inner", left_on="permalink", right_on="company_permalink")
master.head()


# In[26]:


# drop the duplicate column "company_permalink"
master =  master.drop(['company_permalink'], axis=1) 


# In[28]:


# Identifying the null values in columns
master.isnull().sum()


# In[29]:


# summing up the missing values (column-wise) and displaying percentage of NaNs 
round(100*(master.isnull().sum()/len(master.index)), 2)


# The columns ```funding_round_code``` is useless (with about 73% missing values). 
# Also, from the business objectives given, the columns ```homepage_url```, ```founded_at```, 
# ```state_code```, ```region``` and ```city``` need not be used and we shall drop these columns.

# In[30]:


drop_columns = ['funding_round_code', 'homepage_url', 'founded_at', 'state_code', 'region', 'city']
master = master.drop(drop_columns, axis=1)


# In[31]:


master.head()


# In[32]:


# summing up the missing values (column-wise) and displaying percentage of NaNs for the remaining columns
round(100*(master.isnull().sum()/len(master.index)), 2)


# - The column ```raised_amount_usd``` is an important column, since that is the number we want to analyse (compare, means, sum etc.). 
# and hence this column needs to be carefully treated 
# - The column ```country_code``` will be used for country-wise analysis, and 
# - The column ```category_list``` will be used to merge the dataframe with the main categories.

# Identifying the missing values in ```raised_amount_usd```.

# In[33]:


# summary stats of raised_amount_usd
master['raised_amount_usd'].describe()


# - The mean is somewhere around USD 10 million, while the median is only about USD 100m. 
# - The min and max values are also quite wide apart (22 billion)
# - Hence, we will not be able to impute and hence removing the NaN values

# In[34]:


# removing NaNs in raised_amount_usd
master = master[~np.isnan(master['raised_amount_usd'])]
round(100*(master.isnull().sum()/len(master.index)), 2)


# In[35]:


country_codes = master['country_code'].astype('category')

# displaying frequencies of each category
country_codes.value_counts()


# Now, we can either delete the rows having country_code missing (about 6% rows), or we can impute them by USA. 
# Since the number 6 is quite small, better to just remove the rows.

# In[36]:


# removing rows with missing country_codes
master = master[~pd.isnull(master['country_code'])]

# look at missing values
round(100*(master.isnull().sum()/len(master.index)), 2)


# Since the category_list has only very small % of misisng values, we can remove those rows

# In[37]:


# removing rows with missing category_list values
master = master[~pd.isnull(master['category_list'])]

# look at missing values
round(100*(master.isnull().sum()/len(master.index)), 2)


# In[38]:


# Write the cleaned data frame to a .csv file
master.to_csv("master_df.csv", sep=',', index=False)


# In[39]:


master.info()


# In[40]:


# Finding how many rows have been retained
100*(len(master.index) / len(rounds.index))


# # Funding Type Analysis

# We shall compare the funding amounts across the funding types. We would also need to impose the constraint that the investment amount should be between 5 and 15 million USD.
# We will choose the funding type such that the average investment amount falls in this range.

# In[43]:


master['funding_round_type'].unique()


# Among the above funding types, we shall be considering only 4 of them (**venture**, **angel**, **seed**, **private_equity**)

# In[44]:


df = master[(master.funding_round_type == "venture") | 
        (master.funding_round_type == "angel") | 
        (master.funding_round_type == "seed") | 
        (master.funding_round_type == "private_equity") ]


# We have to compute a representative value of the funding amount for each type of invesstment. 
# We can either choose the mean or the median - We shall take that call  after we have a look at the
# distribution of **raised_amount_usd** to get a sense of the distribution of data.

# In[45]:


# distribution of raised_amount_usd
sns.boxplot(y=df['raised_amount_usd'])
plt.yscale('log')
plt.show()


# There are few extreme values. We shall find the actual median and mean values through summary metrics

# In[47]:


# Summary Metrics
df['raised_amount_usd'].describe()


# The mean is 9.5 million and the median is 2.0 million which is a significant difference. We shall check the 
# mean and median for each individual **funding type**

# In[48]:


# comparing summary stats across four categories
sns.boxplot(x='funding_round_type', y='raised_amount_usd', data=df)
plt.yscale('log')
plt.show()


# In[49]:


# compare the mean and median values across categories
df.pivot_table(values='raised_amount_usd', columns='funding_round_type', aggfunc=[np.median, np.mean])


# The mean and median for each funding_type is varying largely. For eg:- For **Private Equity**, the mean is 74 million 
# and the median is 20 million. 

# We shall go ahead for the **median** as the statistic, since there are many values which are extreme on the 
# higher end thereby pulling up the mean

# In[51]:


# compare the median investment amount across the types
df.groupby('funding_round_type')['raised_amount_usd'].median().sort_values(ascending=False)


# Among the various funding types, only one of them, **Venture** falls in the range of 5 million to 15 million 
# which is the need

# ## Country Analysis
# 
# We shall compare the total investment amounts across countries. We will filter the data for only the **venture** type investments and then compare the **total investment** across countries

# In[53]:


# filter the df for private equity type investments
df = df[df.funding_round_type=="venture"]

# group by country codes and compare the total funding amounts
country_wise_total = df.groupby('country_code')['raised_amount_usd'].sum().sort_values(ascending=False)
print(country_wise_total)


# We shall take the top 5 countries and in that we will filter out non-English speaking countries

# In[54]:


top5countries = country_wise_total[:5]
top5countries


# Amongthe above, China is a non-English speaking country and hence, we shall consider the other top 3. 
# They are **US**, **GBR** and **India**

# In[55]:


# filtering for the top three countries
df = df[(df.country_code=='USA') | (df.country_code=='GBR') | (df.country_code=='IND')]
df.head()


# In[56]:


# boxplot to see distributions of funding amount across countries
plt.figure(figsize=(10, 10))
sns.boxplot(x='country_code', y='raised_amount_usd', data=df)
plt.yscale('log')
plt.show()


# ## Sector Analysis

# We need to extract the main sector using the column category_list. We shall create a new column "main category". As stated in the document, the first token before the seperator "|" shall be considered as the main category

# In[58]:


# extracting the main category
df.loc[:, 'main_category'] = df['category_list'].apply(lambda x: x.split("|")[0])
df.head()


# In[59]:


# drop the category_list column
df = df.drop('category_list', axis=1)
df.head()


# Now, we'll read the ```mapping.csv``` file and merge the main categories with its corresponding column. 

# In[61]:


# read mapping file
mapping = pd.read_csv("mapping.csv", sep=",")
mapping.head()


# In[62]:


# missing values in mapping file
mapping.isnull().sum()


# In[63]:


# remove the row with missing values
mapping = mapping[~pd.isnull(mapping['category_list'])]
mapping.isnull().sum()


# Since we need to merge the mapping file with the main dataframe (df), We shall convert the common column to lowercase in both.

# In[64]:


# converting common columns to lowercase
mapping['category_list'] = mapping['category_list'].str.lower()
df['main_category'] = df['main_category'].str.lower()


# To be able to merge all the ```main_category``` values with the mapping file's ```category_list``` column, all the values in the  ```main_category``` column should be present in the ```category_list``` column of the mapping file.

# In[65]:


# values in main_category column in df which are not in the category_list column in mapping file
df[~df['main_category'].isin(mapping['category_list'])]


# Ideally, the above value should have been zero. We shall look at the values which are present in the mapping file but not in the main dataframe df.

# In[66]:


# values in the category_list column which are not in main_category column 
mapping[~mapping['category_list'].isin(df['main_category'])]


# From above we see that the value **analytics** is misspelled as **a0lytics**. Similary for alternative medicine also

# In[67]:


# replacing '0' with 'na'
mapping['category_list'] = mapping['category_list'].apply(lambda x: x.replace('0', 'na'))
print(mapping['category_list'])


# Once again check for missing values between df and mapping file (for categories)

# In[129]:


df[~df['main_category'].isin(mapping['category_list'])]


# We shall be dropping the above 6 records and proceed ahead with the rest of the records and merge the dataframes (df & mapping)

# In[138]:


# merge the dfs
df = pd.merge(df, mapping, how='inner', left_on='main_category', right_on='category_list')
df.head()


# In[139]:


# let's drop the category_list column since it is the same as main_category
df = df.drop('category_list', axis=1)
df.head()


# In[140]:


df.info()


# WE shall now merge the last 9 columns to a single column "Sub Category"

# In[141]:


# store the value and id variables in two separate arrays

# store the value variables in one Series
value_vars = df.columns[9:18]

# take the setdiff() to get the rest of the variables
id_vars = np.setdiff1d(df.columns, value_vars)

print(value_vars, "\n")
print(id_vars)


# In[142]:


# convert into long
long_df = pd.melt(df, 
        id_vars=list(id_vars), 
        value_vars=list(value_vars))

long_df.head()


# Since the value of "0" is not useful for us, we shll only use those rows where the value is "1".
# Latter we shll drop the value column 

# In[143]:


long_df = long_df[long_df['value']==1]
long_df = long_df.drop('value', axis=1)


# In[144]:


len(long_df)


# In[145]:


# renaming the 'variable' column
long_df = long_df.rename(columns={'variable': 'sector'})


# The dataframe now contains only venture type investments in countries USA, IND and GBR, and 
# we have mapped each company to one of the eight main sectors (named 'sector' in the dataframe). 
# 
# We can now compute the sector-wise number and the amount of investment in the three countries.

# In[146]:


# We are interested only for investment range between 5 and 15m. Hence filtering for them
df = long_df[(long_df['raised_amount_usd'] >= 5000000) & (long_df['raised_amount_usd'] <= 15000000)]


# In[147]:


# groupby country, sector and compute the count and sum
df.groupby(['country_code', 'sector']).raised_amount_usd.agg(['count', 'sum'])


# In[155]:


# plotting sector-wise count and sum of investments in the three countries
plt.figure(figsize=(16, 25))

plt.subplot(2, 1, 1)
df['raised_amount_usd_million'] = df['raised_amount_usd'].div(1000000)
p = sns.barplot(x='sector', y='raised_amount_usd_million', hue='country_code', data=df, estimator=np.sum)
p.set_xticklabels(p.get_xticklabels(),rotation=30)
plt.title('Total Invested Amount (USD) (million)')

plt.subplot(2, 1, 2)
q = sns.countplot(x='sector', hue='country_code', data=df)
q.set_xticklabels(q.get_xticklabels(),rotation=45)
plt.title('Number of Investments')


plt.show()


# Thus, the top country in terms of the number of investments (and the total amount invested) is the USA. 
# The sectors 'Others', 'Social, Finance, Analytics and Advertising' and 'Cleantech/Semiconductors' are the most heavily invested ones.
# 
# In case you don't want to consider 'Others' as a sector, 'News, Search and Messaging' is the next best sector.

# ## Rough Work Area

# In[136]:


mapping[mapping['category_list'].str.contains("rac")]


# In[127]:


mapping.loc[(mapping.category_list == 'enterprise 2.na'),'category_list']='enterprise 2.0'


# In[128]:


df.loc[(df.main_category == 'specialty retail'),'main_category'] = 'custom retail'

